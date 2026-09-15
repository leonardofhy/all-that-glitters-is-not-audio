#!/usr/bin/env python3
"""
MMAU-Pro Benchmark Inference Script for Qwen-series Audio Models

Supports Qwen2-Audio-7B-Instruct, Qwen2.5-Omni-7B, Qwen3-Omni-30B
via the shared ``src.inference.AudioLLMEngine`` (vLLM wrapper).
Batched inference with incremental JSONL saving.
"""

import argparse
from tqdm import tqdm

from scripts.mmau_pro.utils import (
    download_mmau_audio,
    load_audio_file,
    chunk_audio,
    format_prompt,
    get_sample_audio_path,
    build_result_dict,
    build_error_result_dict,
    setup_output_file,
    load_completed_ids,
    append_result_jsonl,
    add_common_args,
    load_mmau_pro_dataset,
    load_gen_config,
)
from src.inference import AudioLLMEngine


def main():
    parser = argparse.ArgumentParser(
        description="Run MMAU-Pro benchmark inference with Qwen-series Audio models"
    )
    add_common_args(parser)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: from generation_params.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from generation_params.yaml)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph compilation (helps with TP on some GPUs)")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Max audio duration in seconds (None = no limit)")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")
    parser.set_defaults(
        model_id="Qwen/Qwen2-Audio-7B-Instruct",
        output_path="results/mmau_pro/qwen2_audio_7b_instruct/full.jsonl",
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

    # Generation config (yaml defaults, CLI overrides)
    std_config = load_gen_config(args.model_id)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    print(f"Using Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Dataset + audio
    dataset = load_mmau_pro_dataset(
        num_samples=args.num_samples,
        category=args.category,
    )
    data_dir = download_mmau_audio() if args.audio_condition != "none" else None

    # Engine
    print(f"\nInitializing AudioLLMEngine with model: {args.model_id}")
    engine = AudioLLMEngine(
        model_id=args.model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output_path)
        if completed_ids:
            print(f"Resuming: skipping {len(completed_ids)} already-processed samples")
        output_path = setup_output_file(args.output_path, resume=True)
    else:
        output_path = setup_output_file(args.output_path)

    chunk_info = f", chunk {args.chunk_index}/{args.num_chunks}" if args.num_chunks else ""
    print(f"\nRunning inference on {len(dataset)} samples (condition: {args.audio_condition}{chunk_info})...")
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Inference"):
        batch_samples = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]

        prompts, audios, valid_samples = [], [], []
        for sample in batch_samples:
            if sample["id"] in completed_ids:
                continue
            prompt = format_prompt(
                sample["question"], sample["choices"], sample.get("category", "")
            )

            if args.audio_condition == "none":
                prompts.append(prompt)
                audios.append(None)
                valid_samples.append(sample)
                continue

            audio_ref = get_sample_audio_path(sample)
            if not audio_ref:
                print(f"Warning: Sample {sample['id']} has no audio_path, logging error instead of skipping")
                result = build_error_result_dict(sample, "Missing audio_path in dataset")
                append_result_jsonl(output_path, result)
                continue

            max_dur = None if args.num_chunks is not None else args.max_duration
            audio_array = load_audio_file(
                audio_ref, data_dir,
                max_duration_sec=max_dur,
            )
            if audio_array is None:
                print(f"Logging error for sample {sample['id']}: audio not loadable")
                result = build_error_result_dict(sample, "Audio completely unloadable")
                append_result_jsonl(output_path, result)
                continue

            # Apply chunking if requested
            if args.num_chunks is not None:
                audio_array = chunk_audio(audio_array, args.num_chunks, args.chunk_index)
                if audio_array is None:
                    print(f"Logging error for {sample['id']}: audio too short for {args.num_chunks} chunks")
                    result = build_error_result_dict(sample, "Audio too short to chunk")
                    append_result_jsonl(output_path, result)
                    continue

            prompts.append(prompt)
            audios.append([audio_array])
            valid_samples.append(sample)

        if not prompts:
            continue

        try:
            responses = engine.generate(
                prompts=prompts,
                audios=audios if args.audio_condition != "none" else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for sample, response in zip(valid_samples, responses):
                result = build_result_dict(sample, prediction=response)
                append_result_jsonl(output_path, result)
        except Exception as e:
            print(f"Error during inference: {e}")
            for sample in valid_samples:
                result = build_error_result_dict(sample, e)
                append_result_jsonl(output_path, result)

    print(f"\nDone — results saved to {output_path}")


if __name__ == "__main__":
    main()
