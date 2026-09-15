
import ast
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, Audio

MMAU_MINI_DATASET = "gamma-lab-umd/MMAU-test-mini"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mmau" / "test_mini_audio"

def extract_mmau_mini_audio(cache_dir: Path = DEFAULT_CACHE_DIR, target_sr: int = 16000) -> Path:
    """
    Extracts embedded audio from MMAU-test-mini to WAV files.
    Returns the directory path containing the WAV files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if seems populated (simple check: >900 files)
    existing_files = list(cache_dir.glob("*.wav"))
    if len(existing_files) >= 1000:
        print(f"MMAU Mini audio already extracted to {cache_dir}")
        return cache_dir
        
    print(f"Loading MMAU Mini dataset to extract audio...")
    dataset = load_dataset(MMAU_MINI_DATASET, split="test")
    dataset = dataset.cast_column("context", Audio(decode=False))
    
    print(f"Extracting {len(dataset)} files to {cache_dir}...")
    for i, sample in enumerate(tqdm(dataset)):
        # Use the same ID logic as get_mini_audio_filename() for consistency
        attrs = parse_other_attributes(sample.get('other_attributes', {}))
        sample_id = extract_sample_id(sample, attrs, i)
        safe_id = "".join(x for x in str(sample_id) if x.isalnum() or x in "._-")
        filename = f"{safe_id}.wav"
        output_path = cache_dir / filename
        
        if output_path.exists():
            continue
            
        audio_data = sample.get('context', {})
        if not audio_data or not audio_data.get('bytes'):
            print(f"Warning: Sample {sample_id} has no audio bytes.")
            continue
            
        try:
            with io.BytesIO(audio_data['bytes']) as buffer:
                audio, sr = sf.read(buffer)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1) # Mono
                
                # Resample if needed? 
                # DeSTA/Flamingo usually handle resampling, but let's save as is or standardize?
                # Using 16k is safer for most audio LLMs.
                if sr != target_sr:
                     import librosa
                     audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                     sr = target_sr
                     
                sf.write(str(output_path), audio, sr)
        except Exception as e:
            print(f"Error extracting sample {sample_id}: {e}")
            
    return cache_dir

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


def parse_other_attributes(attrs):
    """Safely parse the other_attributes field from MMAU-mini dataset.

    Handles str (JSON or Python literal), dict, and None inputs.
    Always returns a dict.
    """
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


def extract_sample_id(item: dict, attrs: dict, index: int) -> str:
    """Extract a stable ID for an MMAU-mini sample.

    MMAU-mini has no top-level 'id' column. The real UUID is inside
    other_attributes['id']. Falls back to the dataset index.
    """
    return attrs.get('id', item.get('id', str(index)))


def build_mmau_result(item: dict, attrs: dict, prediction: str, index: int = 0) -> dict:
    """Build a standardized MMAU result dict from a mini dataset sample.

    Output schema matches what evaluate.py expects.
    """
    return {
        "id": extract_sample_id(item, attrs, index),
        "task": attrs.get('task', ''),
        "category": attrs.get('category', ''),
        "sub_category": attrs.get('sub-category', ''),
        "difficulty": attrs.get('difficulty', ''),
        "dataset": attrs.get('dataset', ''),
        "question": item.get('instruction', ''),
        "choices": item.get('choices', []),
        "ground_truth": item.get('answer', ''),
        "prediction": prediction,
    }


def format_mcq_prompt(question: str, choices: list) -> str:
    """Format MCQ prompt aligned with official MMAU eval (token matching)."""
    if not choices:
        return f"{question}\n\nProvide your answer:"
    options_str = "\n".join(
        f"{chr(65 + i)}: {choice}"
        for i, choice in enumerate(choices)
    )
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with the full option text (you may include the letter)."
    )


def get_mini_audio_filename(item: dict, attrs: dict, index: int) -> str:
    """Get the WAV filename for a mini dataset sample.

    Uses the same ID-to-filename logic as extract_mmau_mini_audio().
    """
    sample_id = extract_sample_id(item, attrs, index)
    safe_id = "".join(x for x in str(sample_id) if x.isalnum() or x in "._-")
    return f"{safe_id}.wav"


if __name__ == "__main__":
    extract_mmau_mini_audio()
