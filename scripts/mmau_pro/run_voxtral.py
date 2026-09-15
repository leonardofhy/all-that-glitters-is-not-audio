#!/usr/bin/env python3
"""
MMAU-Pro Benchmark Inference Script for Voxtral-Mini-3B

Uses vLLM with Mistral tokenization (``mistral_common``).
Two-phase: (1) prepare & encode all prompts, (2) single ``llm.generate()`` call.

For the ``none`` condition, audio is completely omitted (no silence placeholder).
See commit 185c107 for the rationale behind the none-omit approach.
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from scripts.mmau_pro.utils import (
    download_mmau_audio,
    load_audio_file,
    chunk_audio,
    format_prompt,
    get_sample_audio_path,
    build_result_dict,
    setup_output_file,
    append_result_jsonl,
    load_completed_ids,
    add_common_args,
    load_mmau_pro_dataset,
    load_gen_config,
)


def main():
    parser = argparse.ArgumentParser(description="Run Voxtral-Mini-3B Inference on MMAU-Pro")
    add_common_args(parser)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens from generation config")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature from generation config")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override top_p from generation config")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Max audio duration in seconds (None = no limit)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")
    parser.set_defaults(
        model_id="mistralai/Voxtral-Mini-3B-2507",
        output_path="results/mmau_pro/voxtral_mini_3b/full.jsonl",
    )
    args = parser.parse_args()

    # Validate chunk args
    if args.num_chunks is not None:
        if args.audio_condition == "none":
            parser.error("--num_chunks is incompatible with --audio_condition none")
        if args.chunk_index is None:
            parser.error("--chunk_index required when --num_chunks is set")
        if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
            parser.error(f"--chunk_index must be in [0, {args.num_chunks})")
        if args.num_chunks < 2:
            parser.error("--num_chunks must be >= 2")

    # Generation config (centralized yaml, CLI overrides)
    std_config = load_gen_config(args.model_id)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    top_p = args.top_p if args.top_p is not None else std_config.get("top_p", 1.0)
    print(f"Using Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")

    # Dataset + audio
    dataset = load_mmau_pro_dataset(
        num_samples=args.num_samples,
        category=args.category,
    )
    data_dir = download_mmau_audio()

    # vLLM engine
    print(f"Loading Model: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        limit_mm_per_prompt={"audio": 1},
        tokenizer_mode="mistral",
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output_path)
        if completed_ids:
            print(f"Resuming: skipping {len(completed_ids)} already-processed samples")

    output_path = setup_output_file(args.output_path, resume=args.resume)

    # Phase 1: Prepare prompts + audio
    prompts = []
    audios = []
    prepared_samples = []
    sampling_rate = 16000

    print("Preparing prompts...")
    for item in tqdm(dataset):
        if item["id"] in completed_ids:
            continue
        try:
            user_content = format_prompt(
                item["question"], item.get("choices", []), item.get("category", "")
            )

            audio_ref = get_sample_audio_path(item)
            if not audio_ref:
                print(f"Warning: No audio path for {item['id']}")
                result = build_error_result_dict(item, "Missing audio_path")
                append_result_jsonl(output_path, result)
                continue

            max_dur = None if args.num_chunks is not None else args.max_duration
            audio_input = load_audio_file(audio_ref, data_dir, target_sr=sampling_rate, max_duration_sec=max_dur)
            if audio_input is None:
                print(f"Warning: Audio load failed for {item['id']}")
                result = build_error_result_dict(item, "Audio completely unloadable")
                append_result_jsonl(output_path, result)
                continue

            # Apply chunking if requested (before silence override)
            if args.num_chunks is not None:
                audio_input = chunk_audio(audio_input, args.num_chunks, args.chunk_index)
                if audio_input is None:
                    print(f"Skipping {item['id']}: audio too short for {args.num_chunks} chunks")
                    result = build_error_result_dict(item, "Audio too short to chunk")
                    append_result_jsonl(output_path, result)
                    continue

            # none-omit: do not pass any audio for text-only condition
            if args.audio_condition == "none":
                audio_input = None

            prompts.append([{"role": "user", "content": user_content}])
            audios.append(audio_input)
            prepared_samples.append(item)
        except Exception as e:
            print(f"Skipping {item.get('id', 'unknown')}: {e}")
            result = build_error_result_dict(item, f"Error preparing prompt: {e}")
            append_result_jsonl(output_path, result)
            continue

    # Phase 2: Encode via mistral_common
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.chunk import TextChunk, AudioChunk, RawAudio
    from mistral_common.tokens.tokenizers.audio import Audio

    mistral_tokenizer = tokenizer.mistral

    final_inputs = []
    survived_indices = []

    for idx, (msg, aud) in enumerate(zip(prompts, audios)):
        try:
            text_content = msg[0]["content"]

            content_chunks = [TextChunk(text=text_content)]
            if aud is not None:
                audio_obj = Audio(audio_array=aud, sampling_rate=sampling_rate, format="wav")
                audio_chunk = AudioChunk(input_audio=RawAudio.from_audio(audio_obj))
                content_chunks = [audio_chunk, TextChunk(text=text_content)]

            request = ChatCompletionRequest(
                messages=[UserMessage(content=content_chunks)]
            )
            encoded = mistral_tokenizer.encode_chat_completion(request)

            input_record = {"prompt_token_ids": encoded.tokens}
            if aud is not None:
                processed_audios = (
                    [a.audio_array for a in encoded.audios] if encoded.audios else [aud]
                )
                input_record["multi_modal_data"] = {
                    "audio": processed_audios[0] if processed_audios else aud
                }

            final_inputs.append(input_record)
            survived_indices.append(idx)
        except Exception as e:
            print(f"Error encoding sample {idx}: {e}", flush=True)
            result = build_error_result_dict(prepared_samples[idx], f"Encoding error: {e}")
            append_result_jsonl(output_path, result)
            continue

    if not final_inputs:
        print("No valid samples after encoding.")
        return

    # Inference
    print(f"Running vLLM generate on {len(final_inputs)} samples...")
    outputs = llm.generate(final_inputs, sampling_params=sampling_params)

    # Save
    for output, sample_idx in zip(outputs, survived_indices):
        pred = output.outputs[0].text
        result = build_result_dict(prepared_samples[sample_idx], prediction=pred)
        append_result_jsonl(output_path, result)

    print(f"Saved {len(survived_indices)} predictions to {output_path}")


if __name__ == "__main__":
    main()
