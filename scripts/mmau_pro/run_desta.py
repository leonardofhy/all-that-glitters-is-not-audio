#!/usr/bin/env python3
"""
MMAU-Pro Benchmark Inference Script for DeSTA 2.5 Audio

Uses native HuggingFace DeSTA API.  Audio is passed as a file path string
(no librosa needed — DeSTA loads internally).
"""

import argparse
import tempfile
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Shared utilities (also sets up sys.path → project root)
from scripts.mmau_pro.utils import (
    download_mmau_audio,
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

# Try to import DeSTA (only available in am_desta environment)
try:
    from desta import DeSTA25AudioModel
except ImportError:
    print("Error: 'DeSTA' package not found. Make sure you are in the 'am_desta' environment.")
    import sys
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run DeSTA-2.5 inference on MMAU-Pro")
    add_common_args(parser)
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")
    parser.set_defaults(
        model_id="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        output_path="results/mmau_pro/desta2.5/full.jsonl",
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

    # Load dataset
    dataset = load_mmau_pro_dataset(
        num_samples=args.num_samples,
        category=args.category,
    )

    # Download audio (unless text-only)
    data_dir = download_mmau_audio() if args.audio_condition != "none" else None

    # Set up temp dir for chunked audio files (cleaned up after inference)
    tmp_ctx = tempfile.TemporaryDirectory() if args.num_chunks is not None and data_dir is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={args.num_chunks}, K={args.chunk_index}, tmpdir={chunks_dir}")

    # Load model
    print(f"Initializing DeSTA model: {args.model_id}")

    # Generation config (centralized yaml)
    std_config = load_gen_config(args.model_id)
    max_new_tokens = std_config.get("max_tokens", 128)  # yaml uses vLLM key; map to HF
    temperature = std_config.get("temperature", 1.0)
    print(f"Using Generation Config -> max_new_tokens: {max_new_tokens}, temperature: {temperature}")

    try:
        model = DeSTA25AudioModel.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output_path)
        if completed_ids:
            print(f"Resuming: skipping {len(completed_ids)} already-processed samples")
        output_path = setup_output_file(args.output_path, resume=True)
    else:
        output_path = setup_output_file(args.output_path)

    print(f"Running inference on {len(dataset)} samples (condition: {args.audio_condition})...")
    for idx, sample in enumerate(tqdm(dataset)):
        if sample["id"] in completed_ids:
            continue
        prompt_text = format_prompt(
            sample["question"], sample["choices"], sample.get("category", "")
        )

        # Build DeSTA message
        if args.audio_condition == "none" or data_dir is None:
            messages = [{"role": "user", "content": prompt_text}]
        else:
            audio_rel = get_sample_audio_path(sample)
            if audio_rel:
                full_audio_path = str(data_dir / Path(audio_rel).name)
                if args.num_chunks is not None:
                    # Load, chunk, save to temp file
                    try:
                        audio_data, sr = sf.read(full_audio_path)
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        chunked = chunk_audio(audio_data, args.num_chunks, args.chunk_index)
                        if chunked is None:
                            print(f"Warning: Audio too short to chunk for {sample['id']}")
                            result = build_error_result_dict(sample, "Audio too short to chunk")
                            append_result_jsonl(output_path, result)
                            continue
                        chunk_path = str(chunks_dir / Path(audio_rel).name)
                        sf.write(chunk_path, chunked, sr)
                        full_audio_path = chunk_path
                    except Exception as e:
                        print(f"Error chunking audio for {sample['id']}: {e}")
                        result = build_error_result_dict(sample, e)
                        append_result_jsonl(output_path, result)
                        continue
                messages = [{
                    "role": "user",
                    "content": "<|AUDIO|>\n" + prompt_text,
                    "audios": [{"audio": full_audio_path, "text": None}],
                }]
            else:
                messages = [{"role": "user", "content": prompt_text}]

        try:
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                top_p=std_config.get("top_p", 1.0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            prediction = outputs.text
            if isinstance(prediction, list):
                prediction = prediction[0] if prediction else ""
            result = build_result_dict(sample, prediction=prediction)
        except Exception as e:
            print(f"Error on sample {sample['id']}: {e}")
            result = build_error_result_dict(sample, e)

        append_result_jsonl(output_path, result)

    print(f"Done — results saved to {output_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
