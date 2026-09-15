#!/usr/bin/env python3
"""
MMAR Benchmark Inference Script for DeSTA 2.5 Audio
Uses native HuggingFace DeSTA API (not vLLM).
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Try to import DeSTA
try:
    from desta import DeSTA25AudioModel
except ImportError:
    print("Error: 'DeSTA' package not found. Make sure you are in the 'am_desta' environment.")
    sys.exit(1)

MMAR_CACHE_DIR = Path.home() / ".cache" / "mmar"


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    """Safely extract tar contents without allowing path traversal."""
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if member_path != destination and not str(member_path).startswith(f"{destination}{os.sep}"):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(destination)


def download_mmar_audio(cache_dir: Path = MMAR_CACHE_DIR) -> Path:
    """Download and extract MMAR audio files."""
    cache_dir = Path(cache_dir)
    audio_dir = cache_dir / "audio"

    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        return audio_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MMAR audio data...")
    tar_path = hf_hub_download(
        repo_id="BoJack/MMAR",
        filename="mmar-audio.tar.gz",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    with tarfile.open(tar_path, 'r:gz') as tar:
        safe_extract_tar(tar, cache_dir)
    return audio_dir


def chunk_audio(audio: np.ndarray, num_chunks: int, chunk_index: int) -> Optional[np.ndarray]:
    """Split audio into N equal-duration chunks, return chunk at chunk_index."""
    total_samples = len(audio)
    chunk_size = total_samples // num_chunks
    if chunk_size == 0:
        return None
    start = chunk_index * chunk_size
    end = total_samples if chunk_index == num_chunks - 1 else start + chunk_size
    return audio[start:end]

def normalize_choices(choices: Any) -> List[str]:
    if isinstance(choices, dict):
        return [str(v) for _, v in sorted(choices.items())]
    if isinstance(choices, (list, tuple)):
        return [str(c) for c in choices]
    return []


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    if not choices:
        return f"{question}\n\nProvide your answer:"

    options_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer with the full option text (you may include the letter)."
    )


def main():
    parser = argparse.ArgumentParser(description="Run DeSTA-2.5 Inference on MMAR")
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--audio_condition", type=str, default="full", choices=["full", "none", "silence"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Unused, kept for CLI compat")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
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

    # Generation config
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmar", model_id=args.model_id)
    max_new_tokens = std_config.get("max_tokens", 128)
    temperature = std_config.get("temperature", 1.0)
    print(f"Using Generation Config -> max_new_tokens: {max_new_tokens}, temperature: {temperature}")

    print("Loading MMAR dataset...")
    dataset = load_dataset("BoJack/MMAR", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    audio_dir = None
    if args.audio_condition in ("full", "silence"):
        audio_dir = download_mmar_audio()

    # Set up temp dir for chunked audio files (cleaned up after inference)
    tmp_ctx = tempfile.TemporaryDirectory() if args.num_chunks is not None and audio_dir is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={args.num_chunks}, K={args.chunk_index}, tmpdir={chunks_dir}")

    print(f"Initializing DeSTA model: {args.model_id}")
    try:
        model = DeSTA25AudioModel.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results: List[Dict[str, Any]] = []
    print(f"Running inference on {len(dataset)} samples (condition: {args.audio_condition})...")

    for idx, sample in enumerate(tqdm(dataset)):
        question = sample.get("question") or sample.get("text", "")
        choices = normalize_choices(sample.get("choices") or sample.get("options"))
        prompt_text = format_mcq_prompt(question, choices)

        audio_source = "text_only"
        messages = []

        if args.audio_condition == "none":
            messages = [{"role": "user", "content": prompt_text}]
        elif args.audio_condition == "silence":
            # DeSTA text-only: no <|AUDIO|> token, just text prompt
            messages = [{"role": "user", "content": prompt_text}]
            audio_source = "silence"
        elif args.audio_condition == "full" and audio_dir:
            audio_path = sample.get("audio_path", "")
            if audio_path:
                filename = Path(audio_path).name
                full_path = audio_dir / filename
                if full_path.exists():
                    target_path = full_path
                    if args.num_chunks is not None:
                        # Load, chunk, save to temp file
                        audio_data, sr = sf.read(str(full_path))
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        chunked = chunk_audio(audio_data, args.num_chunks, args.chunk_index)
                        if chunked is None:
                            messages = [{"role": "user", "content": prompt_text}]
                            audio_source = "silence_fallback_audio_too_short"
                        else:
                            chunk_path = chunks_dir / filename
                            sf.write(str(chunk_path), chunked, sr)
                            target_path = chunk_path
                    if audio_source != "silence_fallback_audio_too_short":
                        messages = [{
                            "role": "user",
                            "content": "<|AUDIO|>\n" + prompt_text,
                            "audios": [{"audio": str(target_path), "text": None}],
                        }]
                        audio_source = "audio_file"
                else:
                    messages = [{"role": "user", "content": prompt_text}]
                    audio_source = "missing_audio_text_only"
            else:
                messages = [{"role": "user", "content": prompt_text}]
                audio_source = "missing_audio_text_only"
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

            results.append({
                "id": sample.get("id", idx),
                "question": question,
                "choices": choices,
                "answer": sample.get("answer", ""),
                "modality": sample.get("modality", "unknown"),
                "category": sample.get("category", "unknown"),
                "sub-category": sample.get("sub-category") or sample.get("sub_category"),
                "model_prediction": prediction,
                "model_id": args.model_id,
                "audio_condition": args.audio_condition,
                "audio_source": audio_source,
            })
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            results.append({
                "id": sample.get("id", idx),
                "question": question,
                "choices": choices,
                "answer": sample.get("answer", ""),
                "modality": sample.get("modality", "unknown"),
                "category": sample.get("category", "unknown"),
                "sub-category": sample.get("sub-category") or sample.get("sub_category"),
                "model_prediction": "",
                "model_id": args.model_id,
                "audio_condition": args.audio_condition,
                "audio_source": audio_source,
                "error": f"generation_error: {e}",
            })

    print(f"Finished inference for {len(results)} samples.")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
