"""
Shared utilities for MMAU-Pro inference scripts.

Provides:
- Task-type-aware prompt formatting (MCQ / free-form / AIF)
- MMAU-Pro audio download & extraction
- Audio file loading via librosa
- Audio path resolution (for models that need file:// URLs or path strings)
- Standardized result dict construction
- Incremental JSONL saving with file truncation
- Common CLI argument definitions

Phase 4 consolidation — eliminates ~300 lines of duplication across 6 scripts.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from huggingface_hub import hf_hub_download

# ============================================================
# Constants
# ============================================================

MMAU_CACHE_DIR = Path.home() / ".cache" / "mmau_pro"
MMAU_DATASET_ID = "gamma-lab-umd/MMAU-Pro"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ensure project root is on sys.path so `from src.…` imports work
# regardless of which directory the script is launched from.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Prompt Formatting
# ============================================================

def format_prompt(question: str, choices: list, category: str = "") -> str:
    """
    Format a prompt based on the task category.

    Args:
        question: The question text from the dataset.
        choices: List of answer choices (may be empty for open-ended/AIF).
        category: The sample's category field (e.g., "music", "speech",
                  "open", "instruction following").

    Returns:
        Formatted prompt string appropriate for the task type.
    """
    category_lower = (category or "").strip().lower()

    # --- Open-ended: free-form QA, no letter options ---
    if category_lower == "open":
        return (
            f"{question}\n\n"
            f"Provide a detailed answer:"
        )

    # --- AIF (instruction following): pass the question through directly ---
    # The question itself already contains the instruction constraints.
    if category_lower == "instruction following":
        return question

    # --- Closed-ended (MCQ): letter options with standard format ---
    if choices:
        options_str = "\n".join(
            f"{chr(65 + i)}: {choice}"
            for i, choice in enumerate(choices)
        )
        return (
            f"{question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Answer with the correct option text (you may include the letter)."
        )

    # Fallback for unknown category with no choices
    return f"{question}\n\nProvide your answer:"


# ============================================================
# Audio Download & Loading
# ============================================================

def download_mmau_audio(cache_dir: Path = MMAU_CACHE_DIR) -> Path:
    """
    Download and extract MMAU-Pro audio files from HuggingFace.

    Returns:
        Path to the extracted ``data/`` directory containing .wav files.
    """
    cache_dir = Path(cache_dir)
    data_dir = cache_dir / "data"

    if data_dir.exists() and any(data_dir.glob("*.wav")):
        print(f"Audio data already extracted at {data_dir}")
        return data_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MMAU-Pro audio data...")
    zip_path = hf_hub_download(
        repo_id=MMAU_DATASET_ID,
        filename="data.zip",
        repo_type="dataset",
        local_dir=cache_dir,
    )

    print(f"Extracting audio files to {cache_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)

    n_files = len(list(data_dir.glob("*.wav")))
    print(f"Extracted {n_files} audio files")
    return data_dir


def load_audio_file(
    audio_path: str,
    data_dir: Path,
    target_sr: int = 16000,
    max_duration_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Load an audio file as a numpy array via librosa.

    Args:
        audio_path: Relative path from the dataset (e.g. ``"data/xxx.wav"``).
                    Only the filename component is used.
        data_dir:   Base directory containing extracted audio files.
        target_sr:  Target sample rate.
        max_duration_sec: Maximum duration to load (seconds).  ``None`` = full.

    Returns:
        1-D float32 numpy array, or ``None`` on failure.
    """
    import librosa  # deferred so scripts that don't need audio can skip it

    filename = Path(audio_path).name
    full_path = data_dir / filename

    if not full_path.exists():
        print(f"Warning: Audio file not found: {full_path}")
        return None

    try:
        audio, _ = librosa.load(str(full_path), sr=target_sr, duration=max_duration_sec)
        return audio
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None


def chunk_audio(
    audio: np.ndarray,
    num_chunks: int,
    chunk_index: int,
) -> Optional[np.ndarray]:
    """
    Split audio into N equal-duration chunks, return chunk at chunk_index.

    Args:
        audio: 1-D float32 numpy array (from load_audio_file).
        num_chunks: Total number of equal-duration chunks (N).
        chunk_index: 0-indexed chunk to extract (0 <= chunk_index < num_chunks).

    Returns:
        1-D float32 numpy array for the requested chunk,
        or None if the audio is too short to split.
    """
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


def resolve_audio_path(
    audio_path: str,
    data_dir: Path,
) -> Optional[Path]:
    """
    Resolve an audio reference from the dataset to an absolute ``Path``.

    Tries the filename directly, then probes common extensions.

    Returns:
        Absolute ``Path`` to the file, or ``None`` if not found.
    """
    fname = Path(audio_path).name
    candidate = data_dir / fname
    if candidate.exists():
        return candidate

    stem = Path(audio_path).stem
    for ext in (".wav", ".mp3", ".flac"):
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def get_sample_audio_path(sample: dict) -> Optional[str]:
    """
    Extract the first audio-path string from a dataset sample.

    Handles both list and scalar ``audio_path`` fields.
    """
    audio_paths = sample.get("audio_path", [])
    if not audio_paths:
        return None
    if isinstance(audio_paths, list):
        return audio_paths[0] if audio_paths else None
    return audio_paths


# ============================================================
# Result Dict Construction
# ============================================================

_RESULT_KEYS = (
    "id", "category", "question", "choices", "ground_truth",
    "prediction", "task_identifier", "kwargs", "prompt_transcription",
)


def build_result_dict(sample: dict, prediction: str, **extra) -> dict:
    """
    Build a standardized result dict from a dataset sample.

    Always includes the full schema so that ``evaluate.py`` can consume
    any script's output without missing-column errors.
    """
    result = {
        "id": sample.get("id", ""),
        "category": sample.get("category", ""),
        "question": sample.get("question", ""),
        "choices": sample.get("choices", []),
        "ground_truth": sample.get("answer", ""),
        "prediction": prediction,
        "task_identifier": sample.get("task_identifier"),
        "kwargs": sample.get("kwargs"),
        "prompt_transcription": sample.get("prompt_transcription"),
    }
    result.update(extra)
    return result


def build_error_result_dict(sample: dict, error: Exception) -> dict:
    """Convenience wrapper: builds a result dict with an ERROR prediction."""
    return build_result_dict(sample, prediction=f"ERROR: {error}")


# ============================================================
# Output File Helpers
# ============================================================

def setup_output_file(output_path: str, resume: bool = False) -> Path:
    """
    Create parent directories and prepare the output file.

    Args:
        output_path: Path to write results to.
        resume: If *True*, keep existing content for resumption.
                If *False* (default), truncate so re-runs don't append
                duplicates.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        # Truncate so that re-runs don't append duplicates
        with open(p, "w"):
            pass
    return p


def load_completed_ids(output_path: str) -> set:
    """Load the set of already-completed sample IDs from an output file.

    Also performs cleanup:
    1. Strips any corrupt (truncated) last line from a preempted job.
    2. Deduplicates by sample ``id``, keeping the last occurrence.

    Returns:
        A set of sample ID strings that have already been processed.
    """
    p = Path(output_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    with open(p, "r") as f:
        lines = f.readlines()
    if not lines:
        return set()
    # Validate last line is complete JSON
    try:
        json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"Warning: corrupt last line in {output_path}, stripping it")
        lines = lines[:-1]
    # Deduplicate by sample id, keeping last occurrence
    seen = {}
    for line in lines:
        try:
            row = json.loads(line)
            seen[row["id"]] = line
        except (json.JSONDecodeError, KeyError):
            continue
    if len(seen) < len(lines):
        print(f"Dedup: {output_path} had {len(lines)} lines, deduplicated to {len(seen)}")
        with open(p, "w") as f:
            f.writelines(seen.values())
    return set(seen.keys())


def count_existing_lines(output_path: str) -> int:
    """Count completed lines in an existing output file for resume support.

    .. deprecated:: Use :func:`load_completed_ids` instead for correct
       ID-based resume that handles skipped samples.

    Returns:
        Number of valid, unique lines (0 if the file does not exist or is empty).
    """
    return len(load_completed_ids(output_path))


def append_result_jsonl(output_path, result: dict) -> None:
    """Append a single result dict as a JSON line."""
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


# ============================================================
# Common CLI Arguments
# ============================================================

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the CLI arguments shared by all MMAU-Pro inference scripts.

    Model-specific arguments (e.g. ``--max_model_len``, ``--gpus``)
    should be added by each script after calling this function.
    """
    parser.add_argument(
        "--model_id", type=str, required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save JSONL results",
    )
    parser.add_argument(
        "--audio_condition", type=str, default="full",
        choices=["full", "none"],
        help="Audio condition: full (with audio) or none (text only)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Max samples to process (0 = all)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter to a single category",
    )


# ============================================================
# Dataset Loading
# ============================================================

def load_mmau_pro_dataset(
    num_samples: int = 0,
    category: Optional[str] = None,
):
    """
    Load the MMAU-Pro dataset, optionally filtering and limiting.

    Returns:
        A HuggingFace ``Dataset`` object.
    """
    from datasets import load_dataset

    print("Loading MMAU-Pro dataset...")
    dataset = load_dataset(MMAU_DATASET_ID, split="test")
    print(f"Loaded {len(dataset)} samples")

    if category:
        dataset = dataset.filter(lambda x: x["category"] == category)
        print(f"Filtered to {len(dataset)} samples with category='{category}'")

    if num_samples > 0 and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"Selected first {num_samples} samples")

    return dataset


# ============================================================
# Generation Config Helper
# ============================================================

def load_gen_config(model_id: str, benchmark: str = "mmau_pro") -> dict:
    """
    Load generation parameters from ``configs/generation_params.yaml``.

    Thin wrapper around ``src.config_utils.load_generation_config`` that
    handles the import so callers don't need to.
    """
    from src.config_utils import load_generation_config
    return load_generation_config(benchmark=benchmark, model_id=model_id)
