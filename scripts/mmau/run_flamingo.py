#!/usr/bin/env python3
"""
MMAU Benchmark Inference Script for Audio-Flamingo-3
Runs on MMAU-Test-Mini (1000 samples).
"""

import argparse
import os
import json
import tempfile
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

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

def main():
    parser = argparse.ArgumentParser(description="Run Audio-Flamingo-3 Inference on MMAU-Mini")
    parser.add_argument("--model_id", type=str, default="nvidia/audio-flamingo-3-hf", help="Model path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--output_path", type=str, default="results/mmau/audio_flamingo_3/mini_full.jsonl")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--audio_condition", type=str, default="full", choices=["full", "none"])
    parser.add_argument("--use_mini", action="store_true", default=True,
                        help="Ignored (always uses mini). Accepted for run_chunked.sh compatibility.")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    args = parser.parse_args()

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

    # Load Dataset
    print(f"Loading MMAU test-mini dataset...")
    try:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
    except Exception:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini")['test']

    # Drop audio columns to prevent torchcodec decoding requirement
    for col in ["audio", "context"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    # Ensure audio data available (before LLM init to get the path)
    audio_dir = None
    if args.audio_condition != "none":
        print("Ensuring audio extraction from mini dataset...")
        audio_dir = extract_mmau_mini_audio()

    tmp_ctx = tempfile.TemporaryDirectory() if args.num_chunks is not None and audio_dir is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={args.num_chunks}, K={args.chunk_index}, tmpdir={chunks_dir}")

    # Initialize vLLM
    print(f"Loading Model: {args.model_id}")
    llm_kwargs = dict(
        model=args.model_id,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        dtype="bfloat16",
    )
    if audio_dir is not None:
        llm_kwargs["allowed_local_media_path"] = str(audio_dir)
    llm = LLM(**llm_kwargs)

    # Ensure output directory exists and file is truncated
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w"):
        pass

    # Generation config (yaml defaults)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmau", model_id=args.model_id)
    print(f"Using Generation Config -> max_tokens: {std_config['max_tokens']}, temperature: {std_config['temperature']}")

    sampling_params = SamplingParams(
        max_tokens=std_config["max_tokens"],
        temperature=std_config["temperature"],
        repetition_penalty=std_config.get("repetition_penalty", 1.0)
    )

    if args.limit > 0:
        print(f"Limiting dataset to first {args.limit} samples")
        dataset = dataset.select(range(args.limit))

    prompts = []
    ids = []
    metadata_list = []

    print("Preparing prompts...")
    for i, item in enumerate(tqdm(dataset)):
        try:
            question = item.get('instruction', '')
            choices = item.get('choices', [])

            # Parse attributes first (needed for ID and audio path)
            attrs = parse_other_attributes(item.get('other_attributes', {}))
            full_text = format_mcq_prompt(question, choices)
            sample_id = extract_sample_id(item, attrs, i)
            
            meta_item = {
                'question': question,
                'choices': choices,
                'ground_truth': item.get('answer'),
                'task': attrs.get('task', ''),
                'category': attrs.get('category', ''),
                'sub_category': attrs.get('sub-category', ''),
                'difficulty': attrs.get('difficulty', ''),
                'dataset': attrs.get('dataset', ''),
            }

            # Audio Handling
            if args.audio_condition == "full":
                content = []
                filename = get_mini_audio_filename(item, attrs, i)
                full_path = str(audio_dir / filename)

                if not os.path.exists(full_path):
                    print(f"Warning: Audio file not found: {full_path}, skipping sample {i}")
                    err_res = meta_item.copy()
                    err_res['id'] = sample_id
                    err_res['prediction'] = "ERROR: Audio missing"
                    with open(args.output_path, "a") as f: f.write(json.dumps(err_res) + '\n')
                    continue
                elif args.num_chunks is not None:
                    # Load, chunk, and save to temp file
                    audio_data, sr = sf.read(full_path)
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    chunked = chunk_audio(audio_data, args.num_chunks, args.chunk_index)
                    if chunked is None:
                        print(f"Warning: Audio too short to chunk into {args.num_chunks}, skipping")
                        err_res = meta_item.copy()
                        err_res['id'] = sample_id
                        err_res['prediction'] = "ERROR: Audio too short to chunk"
                        with open(args.output_path, "a") as f: f.write(json.dumps(err_res) + '\n')
                        continue
                    chunk_path = str(chunks_dir / filename)
                    sf.write(chunk_path, chunked, sr)
                    content.append({
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{chunk_path}"}
                    })
                else:
                    content.append({
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{full_path}"}
                    })

                content.append({
                    "type": "text",
                    "text": full_text
                })
                prompts.append([{"role": "user", "content": content}])
            else:
                # Text-only: use plain string to avoid vLLM multimodal input validation issues
                prompts.append([{"role": "user", "content": full_text}])
            
            ids.append(sample_id)
            metadata_list.append(meta_item)

        except Exception as e:
            print(f"Skipping sample {i}: {e}")
            continue

    if len(prompts) == 0:
        print("No valid samples.")
        return

    if args.audio_condition == "full":
        print(f"Running vLLM chat on {len(prompts)} samples...")
        outputs = llm.chat(prompts, sampling_params)
    else:
        # Text-only: use generate() with pre-applied chat template to avoid
        # vLLM 0.15 bug where chat() fails on text-only input for multimodal models
        tokenizer = llm.get_tokenizer()
        text_prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in prompts
        ]
        print(f"Running vLLM generate on {len(text_prompts)} text-only samples...")
        outputs = llm.generate(text_prompts, sampling_params)

    with open(args.output_path, 'a') as f:
        for output, meta, id_val in zip(outputs, metadata_list, ids):
            pred = output.outputs[0].text.strip()
            res = meta.copy()
            res['id'] = id_val
            res['prediction'] = pred
            f.write(json.dumps(res) + '\n')

    print(f"Saved to {args.output_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()

if __name__ == "__main__":
    main()
