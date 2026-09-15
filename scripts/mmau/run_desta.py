#!/usr/bin/env python3
"""
MMAU Benchmark Inference Script for DeSTA 2.5 Audio
Runs on MMAU-Test-Mini (1000 samples).
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import load_dataset
from scripts.mmau.utils import (
    extract_mmau_mini_audio,
    parse_other_attributes,
    extract_sample_id,
    build_mmau_result,
    format_mcq_prompt,
    get_mini_audio_filename,
    chunk_audio,
)

# Try to import DeSTA
try:
    from desta import DeSTA25AudioModel
except ImportError:
    print("Error: 'DeSTA' package not found. Make sure you are in the 'am_desta' environment.")
    sys.exit(1)


def run_desta_inference(
    model_id: str,
    num_samples: int,
    output_path: str,
    audio_condition: str = "full",
    num_chunks: int = None,
    chunk_index: int = None,
):
    print(f"Loading MMAU test-mini dataset...")
    try:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
    except Exception:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini")['test']

    # Drop audio columns to prevent torchcodec decoding requirement
    for col in ["audio", "context"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    audio_dir = None
    if audio_condition != "none":
        print("Ensuring audio extraction for mini dataset...")
        audio_dir = extract_mmau_mini_audio()

    tmp_ctx = tempfile.TemporaryDirectory() if num_chunks is not None and audio_dir is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={num_chunks}, K={chunk_index}, tmpdir={chunks_dir}")

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Generation config (yaml defaults)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmau", model_id=model_id)
    max_new_tokens = std_config["max_tokens"]
    gen_temperature = std_config.get("temperature", 1.0)
    print(f"Using Generation Config -> max_new_tokens: {max_new_tokens}, temperature: {gen_temperature}")

    print(f"Initializing DeSTA model: {model_id}")
    try:
        model = DeSTA25AudioModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []
    print(f"Running inference on {len(dataset)} samples...")

    for i, sample in enumerate(tqdm(dataset)):
        question = sample.get('instruction', '')
        choices = sample.get('choices', [])

        # Parse attributes first (needed for ID and audio path)
        attrs = parse_other_attributes(sample.get('other_attributes', {}))

        prompt_text = format_mcq_prompt(question, choices)

        messages = []

        if audio_condition == "none":
            messages = [{"role": "user", "content": prompt_text}]
        else:
            filename = get_mini_audio_filename(sample, attrs, i)
            full_audio_path = str(audio_dir / filename)

            if not os.path.exists(full_audio_path):
                print(f"Warning: Audio file {full_audio_path} not found.")
                messages = [{"role": "user", "content": prompt_text}]
            elif num_chunks is not None:
                # Load, chunk, and save to temp file
                audio_data, sr = sf.read(full_audio_path)
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                chunked = chunk_audio(audio_data, num_chunks, chunk_index)
                if chunked is None:
                    print(f"Warning: Audio too short to chunk into {num_chunks}, logging error for {i}")
                    raise ValueError(f"Audio too short to chunk into {num_chunks}")
                chunk_path = str(chunks_dir / filename)
                sf.write(chunk_path, chunked, sr)
                messages = [{
                    "role": "user",
                    "content": "<|AUDIO|>\n" + prompt_text,
                    "audios": [{
                        "audio": chunk_path,
                        "text": None
                    }]
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": "<|AUDIO|>\n" + prompt_text,
                    "audios": [{
                        "audio": full_audio_path,
                        "text": None
                    }]
                }]

        try:
            outputs = model.generate(
                messages=messages,
                do_sample=False,
                top_p=1.0,
                temperature=gen_temperature,
                max_new_tokens=max_new_tokens
            )
            prediction = outputs.text
            if isinstance(prediction, list):
                prediction = prediction[0] if prediction else ""

            result = build_mmau_result(sample, attrs, prediction, i)
            results.append(result)

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            result = build_mmau_result(sample, attrs, f"ERROR: {e}", i)
            results.append(result)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"Saved predictions to {output_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="results/mmau/desta2.5/mini_full.jsonl")
    parser.add_argument("--audio_condition", type=str, default="full")
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

    run_desta_inference(
        model_id=args.model_id,
        num_samples=args.num_samples,
        output_path=args.output_path,
        audio_condition=args.audio_condition,
        num_chunks=args.num_chunks,
        chunk_index=args.chunk_index,
    )

if __name__ == "__main__":
    main()
