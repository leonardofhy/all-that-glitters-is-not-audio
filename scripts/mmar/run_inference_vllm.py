#!/usr/bin/env python3
"""
MMAR Benchmark Inference Script

Runs vLLM-based audio models on MMAR dataset.
Outputs predictions in JSON (official MMAR eval compatible) or JSONL.

Dataset: BoJack/MMAR (1000 test samples, 7 modalities, 4 categories)
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import librosa

# Add project root to path (scripts/mmar -> scripts -> project root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.inference import AudioLLMEngine

# Default cache directory for MMAR audio
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
    """Download and extract MMAR audio files from mmar-audio.tar.gz."""
    cache_dir = Path(cache_dir)
    audio_dir = cache_dir / "audio"

    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        num_files = len(list(audio_dir.glob("*.wav")))
        print(f"MMAR audio already extracted at {audio_dir} ({num_files} files)")
        return audio_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MMAR audio data (mmar-audio.tar.gz)...")
    tar_path = hf_hub_download(
        repo_id="BoJack/MMAR",
        filename="mmar-audio.tar.gz",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    print(f"Extracting audio files to {cache_dir}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        safe_extract_tar(tar, cache_dir)

    num_files = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    print(f"Extracted {num_files} audio files to {audio_dir}")
    return audio_dir


def normalize_choices(choices: Any) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, dict):
        items = list(choices.items())
        try:
            items = sorted(items, key=lambda x: x[0])
        except Exception:
            pass
        return [str(v) for _, v in items]
    if isinstance(choices, (list, tuple)):
        return [str(c) for c in choices]
    return [str(choices)]


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format MCQ prompt aligned with official MMAR eval (token matching)."""
    if not choices:
        return f"{question}\n\nProvide your answer:"

    options_str = "\n".join([
        f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)
    ])

    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer with the full option text (you may include the letter)."
    )


def extract_text_field(sample: Dict[str, Any]) -> str:
    for key in ["question", "query", "prompt", "instruction", "text"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


def chunk_audio(
    audio: np.ndarray,
    num_chunks: int,
    chunk_index: int,
) -> Optional[np.ndarray]:
    """Split audio into N equal-duration chunks, return chunk at chunk_index."""
    if num_chunks < 1 or chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(
            f"Invalid chunk params: num_chunks={num_chunks}, chunk_index={chunk_index}"
        )
    if num_chunks == 1:
        return audio
    total_samples = len(audio)
    chunk_size = total_samples // num_chunks
    if chunk_size == 0:
        return None
    start = chunk_index * chunk_size
    end = total_samples if chunk_index == num_chunks - 1 else start + chunk_size
    return audio[start:end]


def load_audio_from_path(
    audio_path: str,
    audio_dir: Path,
    target_sr: int = 16000,
    max_duration_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Load audio file from MMAR audio directory."""
    filename = Path(audio_path).name
    full_path = audio_dir / filename

    if not full_path.exists():
        print(f"Warning: Audio file not found: {full_path}")
        return None
    try:
        audio, _ = librosa.load(str(full_path), sr=target_sr, duration=max_duration_sec)
        return audio
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None


def resolve_output_format(output_path: Path, output_format: str) -> str:
    if output_format != "auto":
        return output_format
    return "jsonl" if output_path.suffix.lower() == ".jsonl" else "json"


def main():
    parser = argparse.ArgumentParser(description="Run MMAR inference with vLLM-based audio models")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    parser.add_argument("--output_path", type=str, default="results/mmar/inference.json")
    parser.add_argument("--output_format", type=str, default="auto", choices=["auto", "json", "jsonl"],
                        help="Output format. 'auto' infers from extension ('.jsonl' -> jsonl, else json)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: from generation_params.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from generation_params.yaml)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--audio_condition", type=str, default="full",
                        choices=["full", "none", "silence"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph compilation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in output file (append mode)")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Max audio duration in seconds (None = no limit)")
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

    # Generation config (yaml defaults, CLI overrides)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmar", model_id=args.model_id)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    print(f"Using Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Load dataset
    print("Loading MMAR dataset...")
    dataset = load_dataset("BoJack/MMAR", split=args.split)
    print(f"Loaded {len(dataset)} samples")

    # Download audio files if needed
    audio_dir = None
    if args.audio_condition == "full":
        audio_dir = download_mmar_audio()

    if args.limit > 0 and args.limit < len(dataset):
        dataset = dataset.select(range(args.limit))
        print(f"Selected first {args.limit} samples")

    # Initialize engine
    print(f"Initializing AudioLLMEngine with model: {args.model_id}")
    engine = AudioLLMEngine(
        model_id=args.model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"audio": 1},
    )

    # Setup output file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = resolve_output_format(output_path, args.output_format)

    batch_prompts: List[str] = []
    batch_audios: List[List[np.ndarray]] = []
    batch_meta: List[Dict[str, Any]] = []
    json_results: List[Dict[str, Any]] = []

    # Resume: load existing results to skip already-processed samples
    existing_ids: set = set()
    if args.resume and output_path.exists():
        try:
            if output_format == "jsonl":
                with open(output_path) as _rf:
                    for _line in _rf:
                        _line = _line.strip()
                        if _line:
                            try:
                                _row = json.loads(_line)
                                existing_ids.add(str(_row.get("id", "")))
                                json_results.append(_row)
                            except json.JSONDecodeError:
                                pass
            else:  # json array
                _existing = json.loads(output_path.read_text())
                if isinstance(_existing, list):
                    for _row in _existing:
                        existing_ids.add(str(_row.get("id", "")))
                    json_results.extend(_existing)
            print(f"Resume: {len(existing_ids)} existing results in {output_path}")
        except Exception as _e:
            print(f"Resume: could not load existing results ({_e}), starting fresh")
            existing_ids.clear()
            json_results.clear()
    elif output_format == "jsonl":
        with open(output_path, "w"):  # truncate for fresh run
            pass
    silence_audio = np.zeros(16000, dtype=np.float32)

    def save_record(record: Dict[str, Any]) -> None:
        if output_format == "jsonl":
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json_results.append(record)

    def flush_batch() -> None:
        nonlocal batch_prompts, batch_audios, batch_meta
        if not batch_prompts:
            return
        try:
            outputs = engine.generate(
                prompts=batch_prompts,
                audios=batch_audios if args.audio_condition != "none" else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            err_msg = str(e)
            for meta in batch_meta:
                record = {
                    **meta,
                    "model_prediction": "",
                    "model_id": args.model_id,
                    "audio_condition": args.audio_condition,
                    "error": f"generation_error: {err_msg}",
                }
                save_record(record)
            batch_prompts, batch_audios, batch_meta = [], [], []
            return

        for i, meta in enumerate(batch_meta):
            prediction = outputs[i] if i < len(outputs) else ""
            record = {
                **meta,
                "model_prediction": prediction,
                "model_id": args.model_id,
                "audio_condition": args.audio_condition,
            }
            if i >= len(outputs):
                record["error"] = "missing_generation_output"
            save_record(record)
        batch_prompts, batch_audios, batch_meta = [], [], []

    print(f"\nRunning inference on {len(dataset)} samples (condition: {args.audio_condition})...")

    for idx, sample in enumerate(tqdm(dataset, desc="Running inference")):
        # Skip already-processed samples when resuming
        if existing_ids and str(sample.get("id", idx)) in existing_ids:
            continue

        question = extract_text_field(sample)
        choices = normalize_choices(sample.get("choices") or sample.get("options"))
        answer = sample.get("answer", "")
        modality = sample.get("modality", "unknown")
        category = sample.get("category", "unknown")
        sub_category = sample.get("sub-category") or sample.get("sub_category")

        prompt = format_mcq_prompt(question, choices)

        audio_array = None
        audio_source = "text_only"
        if args.audio_condition == "full" and audio_dir is not None:
            audio_path = sample.get("audio_path", "")
            if audio_path:
                max_dur = None if args.num_chunks is not None else args.max_duration
                audio_array = load_audio_from_path(audio_path, audio_dir, max_duration_sec=max_dur)
        if args.audio_condition == "silence":
            audio_array = silence_audio
            audio_source = "silence"
        elif args.audio_condition == "full":
            if audio_array is None:
                audio_array = silence_audio
                audio_source = "silence_fallback_missing_audio"
            else:
                audio_source = "audio_file"

        # Apply audio chunking if requested
        if audio_array is not None and args.num_chunks is not None:
            audio_array = chunk_audio(audio_array, args.num_chunks, args.chunk_index)
            if audio_array is None:
                audio_array = silence_audio
                audio_source = "silence_fallback_audio_too_short"

        batch_prompts.append(prompt)
        if args.audio_condition != "none":
            batch_audios.append([audio_array])
        batch_meta.append({
            "id": sample.get("id", idx),
            "question": question,
            "choices": choices,
            "answer": answer,
            "modality": modality,
            "category": category,
            "sub-category": sub_category,
            "audio_source": audio_source,
        })

        if len(batch_prompts) >= args.batch_size:
            flush_batch()

    # Flush remaining batch
    flush_batch()

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        n_results = len(json_results)
    else:
        with open(output_path) as f:
            n_results = sum(1 for _ in f)
    print(f"\nDone — {n_results} predictions saved to {output_path}")


if __name__ == "__main__":
    main()
