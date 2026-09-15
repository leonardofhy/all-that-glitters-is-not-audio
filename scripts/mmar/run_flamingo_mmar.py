#!/usr/bin/env python3
"""
MMAR Benchmark Inference Script for Audio-Flamingo-3
Outputs predictions to JSON format compatible with scripts/mmar/evaluation.py.
"""

import argparse
import json
import os
import tarfile
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

    options_str = "\n".join(
        [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)]
    )

    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer with the full option text (you may include the letter)."
    )

def main():
    parser = argparse.ArgumentParser(description="Run Audio-Flamingo-3 Inference on MMAR")
    parser.add_argument("--model_id", type=str, default="nvidia/audio-flamingo-3-hf")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--audio_condition", type=str, default="full", choices=["full", "none", "silence"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
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

    print("Loading MMAR dataset...")
    dataset = load_dataset("BoJack/MMAR", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    audio_dir = None
    if args.audio_condition == "full":
        audio_dir = download_mmar_audio()

    # Set up temp dir for chunked audio files (cleaned up after inference)
    tmp_ctx = tempfile.TemporaryDirectory() if args.num_chunks is not None and audio_dir is not None else None
    chunks_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None
    if chunks_dir:
        print(f"Chunking: N={args.num_chunks}, K={args.chunk_index}, tmpdir={chunks_dir}")

    print(f"Initializing vLLM with {args.model_id}")
    # Load standard config
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmar", model_id=args.model_id)

    print(f"Using Generation Config -> max_tokens: {std_config['max_tokens']}, temperature: {std_config['temperature']}")

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        allowed_local_media_path="/",
    )

    sampling_params = SamplingParams(
        max_tokens=std_config["max_tokens"],
        temperature=std_config["temperature"],
        repetition_penalty=std_config.get("repetition_penalty", 1.2)
    )

    prompts: List[Any] = []
    metadata: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    
    silent_audio_path = None
    if args.audio_condition == "silence":
        mmar_cache = Path(MMAR_CACHE_DIR)
        mmar_cache.mkdir(parents=True, exist_ok=True)
        silent_audio_path = mmar_cache / "silence_100ms.wav"
        if not silent_audio_path.exists():
            print("Generating silent audio for 'silence' condition...")
            import wave
            import struct
            
            sample_rate = 16000
            duration = 0.1  # seconds
            num_frames = int(sample_rate * duration)
            
            with wave.open(str(silent_audio_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                # Write zeros
                data = struct.pack('<' + 'h'*num_frames, *([0]*num_frames))
                wav_file.writeframes(data)

    def flush_batch() -> None:
        nonlocal prompts, metadata
        if not prompts:
            return

        try:
            if args.audio_condition == "none":
                outputs = llm.generate(prompts, sampling_params)
            else:
                outputs = llm.chat(prompts, sampling_params)
        except Exception as e:
            err_msg = str(e)
            for meta in metadata:
                results.append({
                    **meta,
                    "model_prediction": "",
                    "model_id": args.model_id,
                    "audio_condition": args.audio_condition,
                    "error": f"generation_error: {err_msg}",
                })
            prompts, metadata = [], []
            return

        for i, meta in enumerate(metadata):
            prediction = ""
            output_missing = i >= len(outputs) or not outputs[i].outputs
            if not output_missing:
                prediction = outputs[i].outputs[0].text.strip()
            record = {
                **meta,
                "model_prediction": prediction,
                "model_id": args.model_id,
                "audio_condition": args.audio_condition,
            }
            if output_missing:
                record["error"] = "missing_generation_output"
            results.append(record)

        prompts, metadata = [], []

    print("Preparing prompts...")
    for idx, sample in enumerate(tqdm(dataset)):
        question = sample.get("question") or sample.get("text", "")
        choices = normalize_choices(sample.get("choices") or sample.get("options"))
        prompt_text = format_mcq_prompt(question, choices)
        
        content: List[Dict[str, Any]] = []
        audio_source = "text_only"
        
        # Audio handling
        target_audio_path = None
        if args.audio_condition == "full" and audio_dir:
            audio_path = sample.get("audio_path", "")
            if audio_path:
                filename = Path(audio_path).name
                full_path = audio_dir / filename
                if full_path.exists():
                    if args.num_chunks is not None:
                        # Load, chunk, save to temp file
                        audio_data, sr = sf.read(str(full_path))
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        chunked = chunk_audio(audio_data, args.num_chunks, args.chunk_index)
                        if chunked is None:
                            audio_source = "silence_fallback_audio_too_short"
                        else:
                            chunk_path = chunks_dir / filename
                            sf.write(str(chunk_path), chunked, sr)
                            target_audio_path = chunk_path
                            audio_source = "audio_file"
                    else:
                        target_audio_path = full_path
                        audio_source = "audio_file"
                else:
                    audio_source = "missing_audio_text_only"
            else:
                audio_source = "missing_audio_text_only"
        elif args.audio_condition == "silence" and silent_audio_path:
            target_audio_path = silent_audio_path
            audio_source = "silence"
            
        if target_audio_path:
            content = [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{target_audio_path}"}
                },
                {"type": "text", "text": prompt_text},
            ]
        else:
            content = [{"type": "text", "text": prompt_text}]

        if args.audio_condition == "none":
            prompts.append(prompt_text)
        else:
            prompts.append([{"role": "user", "content": content}])
        
        metadata.append({
            "id": sample.get("id", idx),
            "question": question,
            "choices": choices,
            "answer": sample.get("answer", ""),
            "modality": sample.get("modality", "unknown"),
            "category": sample.get("category", "unknown"),
            "sub-category": sample.get("sub-category") or sample.get("sub_category"),
            "audio_source": audio_source,
        })

        if len(prompts) >= batch_size:
            flush_batch()

    flush_batch()

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
