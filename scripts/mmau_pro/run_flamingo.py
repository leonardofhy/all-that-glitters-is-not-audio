#!/usr/bin/env python3
"""
MMAU-Pro Benchmark Inference Script for Audio-Flamingo-3 (vLLM)

Uses vLLM with ``audio_url`` content blocks (``file://`` URLs).
Batch inference: all prompts built first, then a single ``llm.chat()`` call.
"""

import argparse
import os
import json
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from scripts.mmau_pro.utils import (
    MMAU_CACHE_DIR,
    download_mmau_audio,
    load_audio_file,
    chunk_audio,
    format_prompt,
    get_sample_audio_path,
    resolve_audio_path,
    build_result_dict,
    setup_output_file,
    add_common_args,
    load_mmau_pro_dataset,
    load_gen_config,
)

# Allow long max model length as per NVIDIA example
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def main():
    parser = argparse.ArgumentParser(description="Run Audio-Flamingo-3 inference on MMAU-Pro")
    add_common_args(parser)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.set_defaults(
        model_id="nvidia/audio-flamingo-3-hf",
        output_path="results/mmau_pro/audio_flamingo_3/full.jsonl",
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

    # Dataset + audio
    dataset = load_mmau_pro_dataset(
        num_samples=args.num_samples,
        category=args.category,
    )
    data_dir = download_mmau_audio()

    # Truncate and prep output file at the start so we can append errors incrementally
    output_path = setup_output_file(args.output_path)

    # Set up temp dir for chunked audio files (cleaned up after inference)
    tmp_ctx = tempfile.TemporaryDirectory() if args.num_chunks is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={args.num_chunks}, K={args.chunk_index}, tmpdir={chunks_dir}")

    # Generation config
    std_config = load_gen_config(args.model_id)

    # Initialize vLLM
    print(f"Loading Model: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
        allowed_local_media_path=str(Path.home()),
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        dtype="bfloat16",
    )

    print(f"Using Generation Config -> max_tokens: {std_config['max_tokens']}, "
          f"temperature: {std_config['temperature']}")

    sampling_params = SamplingParams(
        max_tokens=std_config["max_tokens"],
        temperature=std_config["temperature"],
        repetition_penalty=std_config.get("repetition_penalty", 1.0),
    )

    # Build all prompts
    prompts = []
    survived_items = []  # Track which dataset items made it into prompts
    for item in tqdm(dataset, desc="Building prompts"):
        full_text = format_prompt(
            item["question"], item.get("choices", []), item.get("category", "")
        )

        if args.audio_condition == "full":
            content = []
            audio_ref = get_sample_audio_path(item)
            if audio_ref:
                audio_path_obj = resolve_audio_path(audio_ref, data_dir)
                if audio_path_obj:
                    if args.num_chunks is not None:
                        # Load, chunk, save to temp file
                        audio_data, sr = sf.read(str(audio_path_obj))
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        chunked = chunk_audio(audio_data, args.num_chunks, args.chunk_index)
                        if chunked is None:
                            print(f"Warning: Audio too short to chunk for {item.get('id', 'unknown')}")
                            result = build_error_result_dict(item, "Audio too short to chunk")
                            append_result_jsonl(output_path, result)
                            continue
                        chunk_path = chunks_dir / audio_path_obj.name
                        sf.write(str(chunk_path), chunked, sr)
                        content.append({
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{chunk_path.absolute()}"},
                        })
                    else:
                        content.append({
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{audio_path_obj.absolute()}"},
                        })
                else:
                    print(f"Warning: Audio file not found for {item.get('id', 'unknown')}")
                    result = build_error_result_dict(item, "Missing audio_path or file not found")
                    append_result_jsonl(output_path, result)
                    continue
            content.append({"type": "text", "text": full_text})
            prompts.append([{"role": "user", "content": content}])
        else:
            # Text-only: use plain string to avoid vLLM 0.15 multimodal input validation bug
            prompts.append([{"role": "user", "content": full_text}])

        survived_items.append(item)

    # Inference
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

    # Save successful results append-style
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        result = build_result_dict(survived_items[i], prediction=generated_text)
        append_result_jsonl(output_path, result)

    print(f"Results saved to {output_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
