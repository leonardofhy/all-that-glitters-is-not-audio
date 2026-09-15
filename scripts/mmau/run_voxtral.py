#!/usr/bin/env python3
"""
MMAU Benchmark Inference Script for Voxtral-Mini-3B
Runs on MMAU-Test-Mini (1000 samples).
"""

import argparse
import os
import json
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mmau.utils import (
    extract_mmau_mini_audio,
    parse_other_attributes,
    extract_sample_id,
    format_mcq_prompt,
    get_mini_audio_filename,
    chunk_audio,
)

def load_audio_file(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file from disk."""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return None

    try:
        audio, _ = librosa.load(audio_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Voxtral-Mini-3B Inference on MMAU-Mini")
    parser.add_argument("--model_path", type=str, default="mistralai/Voxtral-Mini-3B-2507", help="Model path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--output_path", type=str, default="results/mmau/voxtral_mini_3b/mini_full.jsonl")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--audio_condition", type=str, default="full", choices=["full", "none"])
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from generation_params.yaml)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override top_p (default: from generation_params.yaml)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: from generation_params.yaml)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--model_id", type=str, default=None,
                        help="Alias for --model_path (for run_chunked.sh compatibility)")
    parser.add_argument("--use_mini", action="store_true", default=True,
                        help="Ignored (always uses mini). Accepted for run_chunked.sh compatibility.")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    args = parser.parse_args()

    # --model_id is an alias for --model_path
    if args.model_id is not None:
        args.model_path = args.model_id

    # Generation config (yaml defaults, CLI overrides)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmau", model_id=args.model_path)
    args.temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    args.top_p = args.top_p if args.top_p is not None else std_config.get("top_p", 1.0)
    args.max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    print(f"Using Generation Config -> max_tokens: {args.max_tokens}, temperature: {args.temperature}")

    # Validate chunking args
    if args.num_chunks is not None:
        if args.chunk_index is None:
            parser.error("--chunk_index is required when --num_chunks is set")
        if args.num_chunks < 2:
            parser.error("--num_chunks must be >= 2")
        if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
            parser.error(f"--chunk_index must be in [0, {args.num_chunks})")
        if args.audio_condition == "none":
            parser.error("--num_chunks is incompatible with --audio_condition none")

    # Ensure audio data available
    audio_dir = None
    if args.audio_condition != "none":
        print("Ensuring audio extraction from mini dataset...")
        audio_dir = extract_mmau_mini_audio()

    # Load Dataset
    print(f"Loading MMAU test-mini dataset...")
    try:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
    except Exception:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini")['test']

    # DROP audio-related columns to prevent automatic decoding/torchcodec requirements
    for col in ["audio", "context"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    if args.limit > 0:
        print(f"Limiting dataset to first {args.limit} samples")
        dataset = dataset.select(range(args.limit))

    # Ensure output directory exists and file is truncated
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w"): pass

    # Initialize vLLM
    print(f"Loading Model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768,
        limit_mm_per_prompt={"audio": 1},
        tokenizer_mode="mistral",
    )

    tokenizer = llm.get_tokenizer()

    prompts = []
    audios = []
    ids = []
    metadata_list = []

    print("Preparing prompts...")
    for i, item in enumerate(tqdm(dataset)):
        try:
            question = item.get('instruction', '')
            choices = item.get('choices', [])

            # Parse attributes first (needed for ID)
            attrs = parse_other_attributes(item.get('other_attributes', {}))
            user_content = format_mcq_prompt(question, choices)
            sample_id = extract_sample_id(item, attrs, i)
            meta = {
                'question': question,
                'choices': choices,
                'ground_truth': item.get('answer'),
                'task': attrs.get('task', ''),
                'category': attrs.get('category', ''),
                'sub_category': attrs.get('sub-category', ''),
                'difficulty': attrs.get('difficulty', ''),
                'dataset': attrs.get('dataset', ''),
            }

            messages = [{"role": "user", "content": user_content}]

            # Audio Handling
            audio_input = None
            if args.audio_condition != "none":
                filename = get_mini_audio_filename(item, attrs, i)
                full_path = str(audio_dir / filename)

                audio_input = load_audio_file(full_path)

                if audio_input is None:
                    print(f"Warning: Audio load failed for sample {i}, skipping")
                    err_res = meta.copy()
                    err_res['id'] = sample_id
                    err_res['prediction'] = "ERROR: Audio completely unloadable"
                    with open(args.output_path, "a") as f: f.write(json.dumps(err_res) + '\n')
                    continue

                if args.num_chunks is not None:
                    chunked = chunk_audio(audio_input, args.num_chunks, args.chunk_index)
                    if chunked is None:
                        print(f"Warning: Audio too short to chunk into {args.num_chunks}, skipping")
                        err_res = meta.copy()
                        err_res['id'] = sample_id
                        err_res['prediction'] = "ERROR: Audio too short to chunk"
                        with open(args.output_path, "a") as f: f.write(json.dumps(err_res) + '\n')
                        continue
                    audio_input = chunked
            else:
                audio_input = None

            prompts.append(messages)
            audios.append(audio_input)
            ids.append(sample_id)
            metadata_list.append(meta)
        except Exception as e:
            print(f"Error preparing sample {i}: {e}")
            sample_id = extract_sample_id(item, parse_other_attributes(item.get('other_attributes', {})), i)
            attrs = parse_other_attributes(item.get('other_attributes', {}))
            error_res = {
                'id': sample_id,
                'question': item.get('instruction', ''),
                'choices': item.get('choices', []),
                'ground_truth': item.get('answer'),
                'task': attrs.get('task', ''),
                'prediction': f"ERROR: {e}",
            }
            with open(args.output_path, 'a') as ef:
                ef.write(json.dumps(error_res) + '\n')
            continue

    # Voxtral requires audio input via mistral_common
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.chunk import TextChunk, AudioChunk, RawAudio
    from mistral_common.tokens.tokenizers.audio import Audio as MistralAudio

    mistral_tokenizer = tokenizer.mistral
    final_inputs = []
    survived_indices = []
    sampling_rate = 16000

    print("Encoding prompts with mistral_common...")
    for idx, (msg, aud) in enumerate(zip(prompts, audios)):
        try:
            text_content = msg[0]["content"]

            content_chunks = [TextChunk(text=text_content)]
            if aud is not None:
                audio_obj = MistralAudio(
                    audio_array=aud,
                    sampling_rate=sampling_rate,
                    format="wav"
                )
                audio_chunk = AudioChunk(input_audio=RawAudio.from_audio(audio_obj))
                content_chunks = [audio_chunk, TextChunk(text=text_content)]

            request = ChatCompletionRequest(
                messages=[
                    UserMessage(content=content_chunks)
                ]
            )

            encoded = mistral_tokenizer.encode_chat_completion(request)
            prompt_tokens = encoded.tokens

            input_record = {"prompt_token_ids": prompt_tokens}
            if aud is not None:
                processed_audios = [a.audio_array for a in encoded.audios] if encoded.audios else [aud]
                input_record["multi_modal_data"] = {"audio": processed_audios[0] if processed_audios else aud}

            final_inputs.append(input_record)
            survived_indices.append(idx)
        except Exception as e:
            print(f"Error encoding sample {idx}: {e}")
            meta = metadata_list[idx]
            error_res = meta.copy()
            error_res['id'] = ids[idx]
            error_res['prediction'] = f"ERROR: encoding_error: {e}"
            with open(args.output_path, 'a') as ef:
                ef.write(json.dumps(error_res) + '\n')
            continue

    if len(final_inputs) == 0:
        print("No valid samples.")
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"Running vLLM generate on {len(final_inputs)} samples...")
    outputs = llm.generate(final_inputs, sampling_params=sampling_params)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'a') as f:
        for output, sample_idx in zip(outputs, survived_indices):
            meta = metadata_list[sample_idx]
            id_val = ids[sample_idx]
            pred = output.outputs[0].text
            res = meta.copy()
            res['id'] = id_val
            res['prediction'] = pred
            f.write(json.dumps(res) + '\n')

    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
