#!/usr/bin/env python3
"""
MMAU Benchmark Inference Script

Runs Audio-LLM inference on MMAU benchmark samples using vLLM.
MMAU is MCQ-only (simpler than MMAU-Pro).

Datasets:
  - Mini: gamma-lab-umd/MMAU-test-mini (1000 samples, embedded audio)
  - Full: gamma-lab-umd/MMAU-test (9000 samples, external audio files)

Prompt format aligned with official MMAU eval (string_match token matching).
"""

import argparse
import ast
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
import librosa
import soundfile as sf
from src.inference import AudioLLMEngine
from scripts.mmau.utils import chunk_audio

# Constants
MMAU_CACHE_DIR = Path.home() / ".cache" / "mmau"
MMAU_DATASET_FULL = "gamma-lab-umd/MMAU-test"
MMAU_DATASET_MINI = "gamma-lab-umd/MMAU-test-mini"


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    """Safely extract tar contents without allowing path traversal."""
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if member_path != destination and not str(member_path).startswith(
            f"{destination}{os.sep}"
        ):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(destination)


def download_mmau_audio(cache_dir: Path = MMAU_CACHE_DIR) -> Path:
    """Download and extract MMAU audio files from test-audios.tar.gz."""
    cache_dir = Path(cache_dir)
    audio_dir = cache_dir / "test-audios"

    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        num_files = len(list(audio_dir.glob("*.wav")))
        print(f"Audio data already extracted at {audio_dir} ({num_files} files)")
        return audio_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MMAU audio data (test-audios.tar.gz)...")
    tar_path = hf_hub_download(
        repo_id=MMAU_DATASET_FULL,
        filename="test-audios.tar.gz",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    print(f"Extracting audio files to {cache_dir}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        safe_extract_tar(tar, cache_dir)

    num_files = len(list(audio_dir.glob("*.wav")))
    print(f"Extracted {num_files} audio files to {audio_dir}")
    return audio_dir


def load_audio_file(
    audio_id: str,
    audio_dir: Path,
    target_sr: int = 16000,
    max_duration_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Load audio file from disk (full dataset)."""
    filename = Path(audio_id).name
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


def load_embedded_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = 16000,
    max_duration_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Decode embedded audio bytes from MMAU mini dataset."""
    try:
        with io.BytesIO(audio_bytes) as buffer:
            audio, sr = sf.read(buffer, dtype="float32", always_2d=False)
        if audio is None:
            return None
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        return audio
    except Exception as e:
        print(f"Error decoding embedded audio: {e}")
        return None


def format_mcq_prompt(question: str, choices: list) -> str:
    """Format MCQ prompt aligned with official MMAU eval (token matching)."""
    if not choices:
        return f"{question}\n\nProvide your answer:"

    options_str = "\n".join([
        f"{chr(65 + i)}: {choice}"
        for i, choice in enumerate(choices)
    ])

    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with the full option text (you may include the letter)."
    )


def _parse_other_attributes(attrs):
    """Parse the other_attributes field from MMAU-mini dataset."""
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (json.JSONDecodeError, TypeError):
            try:
                attrs = ast.literal_eval(attrs) if isinstance(attrs, str) else {}
            except (ValueError, SyntaxError):
                attrs = {}
    if not isinstance(attrs, dict):
        attrs = {}
    return attrs


def _build_result(sample: dict, prediction: str, is_mini: bool, idx: int = 0) -> dict:
    """Build a standardized result dict from a dataset sample."""
    if is_mini:
        attrs = _parse_other_attributes(sample.get('other_attributes', {}))
        return {
            "id": attrs.get('id', sample.get('id', str(idx))),
            "task": attrs.get('task', ''),
            "category": attrs.get('category', ''),
            "sub_category": attrs.get('sub-category', ''),
            "difficulty": attrs.get('difficulty', ''),
            "dataset": attrs.get('dataset', ''),
            "question": sample.get('instruction', ''),
            "choices": sample.get('choices', []),
            "ground_truth": sample.get('answer', ''),
            "prediction": prediction,
        }
    else:
        return {
            "id": sample.get('id', str(idx)),
            "audio_id": sample.get('audio_id', ''),
            "task": sample.get('task', ''),
            "category": sample.get('category', ''),
            "sub_category": sample.get('sub-category', ''),
            "difficulty": sample.get('difficulty', ''),
            "dataset": sample.get('dataset', ''),
            "question": sample.get('question', ''),
            "choices": sample.get('choices', []),
            "ground_truth": sample.get('answer', ''),
            "prediction": prediction,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run MMAU benchmark inference with Audio-LLM"
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=0, help="0 for all")
    parser.add_argument("--output_path", type=str, default="results/mmau/predictions.jsonl")
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
                        help="Disable CUDA graph compilation")
    parser.add_argument("--audio_condition", type=str, default="full",
                        choices=["full", "none"])
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Max audio duration in seconds (None = no limit)")
    parser.add_argument("--use_mini", action="store_true",
                        help="Use mini test set (1000 samples with embedded audio)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in output file (append mode)")
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

    # Generation config (yaml defaults, CLI overrides)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmau", model_id=args.model_id)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    print(f"Using Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Load dataset
    is_mini = args.use_mini
    if is_mini:
        print(f"Loading MMAU mini dataset from {MMAU_DATASET_MINI}...")
        dataset = load_dataset(MMAU_DATASET_MINI, split="test")
        dataset = dataset.cast_column("context", Audio(decode=False))
    else:
        print(f"Loading MMAU full dataset from {MMAU_DATASET_FULL}...")
        dataset = load_dataset(MMAU_DATASET_FULL, split="test")
    print(f"Loaded {len(dataset)} samples, columns: {dataset.column_names}")

    # Download audio files (full dataset only, mini has embedded audio)
    audio_dir = None
    if args.audio_condition != "none" and not is_mini:
        audio_dir = download_mmau_audio()

    # Limit samples
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
        print(f"Selected first {args.num_samples} samples")

    # Initialize engine
    print(f"\nInitializing AudioLLMEngine with model: {args.model_id}")
    engine = AudioLLMEngine(
        model_id=args.model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    # Setup output file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing IDs to skip already-processed samples
    existing_ids: set = set()
    if args.resume and output_path.exists():
        with open(output_path) as _rf:
            for _line in _rf:
                _line = _line.strip()
                if _line:
                    try:
                        existing_ids.add(str(json.loads(_line).get("id", "")))
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(existing_ids)} existing results in {output_path}")
    else:
        with open(output_path, "w"):  # truncate for fresh run
            pass

    print(f"\nRunning inference on {len(dataset)} samples (condition: {args.audio_condition})...")

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Inference"):
        batch_indices = list(range(i, min(i + args.batch_size, len(dataset))))
        batch_samples = [dataset[j] for j in batch_indices]

        prompts, audios, valid_samples, valid_indices = [], [], [], []

        for j, sample in zip(batch_indices, batch_samples):
            # Skip already-processed samples when resuming
            if existing_ids:
                if is_mini:
                    attrs = _parse_other_attributes(sample.get("other_attributes", {}))
                    _sid = str(attrs.get("id", sample.get("id", str(j))))
                else:
                    _sid = str(sample.get("id", str(j)))
                if _sid in existing_ids:
                    continue

            # Extract question and choices (different column names for mini vs full)
            if is_mini:
                question = sample.get('instruction', '')
                choices = sample.get('choices', [])
            else:
                question = sample.get('question', '')
                choices = sample.get('choices', [])

            prompt = format_mcq_prompt(question, choices)

            if args.audio_condition == "none":
                prompts.append(prompt)
                audios.append(None)
                valid_samples.append(sample)
                valid_indices.append(j)
                continue

            # Load audio
            audio_array = None
            max_dur = None if args.num_chunks is not None else args.max_duration
            if is_mini:
                audio_data = sample.get('context', {})
                if isinstance(audio_data, dict) and audio_data.get('bytes'):
                    audio_array = load_embedded_audio_bytes(
                        audio_data['bytes'], max_duration_sec=max_dur,
                    )
                    if audio_array is None:
                        print("Warning: Failed to decode embedded audio, skipping")
                else:
                    print("Warning: Sample has no embedded audio bytes, skipping")
            else:
                audio_id = sample.get('audio_id', '')
                audio_array = load_audio_file(audio_id, audio_dir, max_duration_sec=max_dur)
                if audio_array is None:
                    print(f"Skipping sample {sample.get('id', j)}: audio not loadable")

            if audio_array is None:
                result = _build_result(sample, "ERROR: Audio completely unloadable", is_mini, idx=j)
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue

            # Apply audio chunking if requested
            if args.num_chunks is not None:
                audio_array = chunk_audio(audio_array, args.num_chunks, args.chunk_index)
                if audio_array is None:
                    print(f"Warning: Audio too short to chunk into {args.num_chunks}, skipping")
                    result = _build_result(sample, "ERROR: Audio too short to chunk", is_mini, idx=j)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    continue

            prompts.append(prompt)
            audios.append([audio_array])
            valid_samples.append(sample)
            valid_indices.append(j)

        if not prompts:
            continue

        try:
            responses = engine.generate(
                prompts=prompts,
                audios=audios if args.audio_condition != "none" else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for sample, idx, response in zip(valid_samples, valid_indices, responses):
                result = _build_result(sample, response, is_mini, idx=idx)
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"Error during inference: {e}")
            for sample, idx in zip(valid_samples, valid_indices):
                result = _build_result(sample, f"ERROR: {e}", is_mini, idx=idx)
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

    # Count results
    with open(output_path) as f:
        n_results = sum(1 for _ in f)
    print(f"\nDone — {n_results} predictions saved to {output_path}")


if __name__ == "__main__":
    main()
