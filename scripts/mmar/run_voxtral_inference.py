#!/usr/bin/env python3
"""
MMAR Benchmark Inference Script for Voxtral-Mini-3B
Specialized script using mistral_common for tokenization.
"""

import argparse
import os
import json
import tarfile
import numpy as np
import librosa
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import mistral-common specific classes
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.chunk import TextChunk, AudioChunk, RawAudio
from mistral_common.tokens.tokenizers.audio import Audio as MistralAudio

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

def load_audio_from_path(audio_path: str, audio_dir: Path, target_sr: int = 16000) -> Optional[np.ndarray]:
    filename = Path(audio_path).name
    full_path = audio_dir / filename
    if not full_path.exists():
        return None
    try:
        audio, _ = librosa.load(str(full_path), sr=target_sr)
        return audio
    except Exception:
        return None

def normalize_choices(choices: Any) -> List[str]:
    if isinstance(choices, dict):
        return [str(v) for _, v in sorted(choices.items())]
    if isinstance(choices, list):
        return [str(c) for c in choices]
    return []

def format_mcq_prompt(question, choices):
    if not choices:
        return f"{question}\n\nProvide your answer:"
    options_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
    return f"{question}\n\nOptions:\n{options_str}\n\nAnswer with the full option text (you may include the letter)."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--audio_condition", type=str, default="full", choices=["full", "none", "silence"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in output file (append mode)")
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

    # Generation config (yaml defaults)
    from src.config_utils import load_generation_config
    std_config = load_generation_config(benchmark="mmar", model_id=args.model_id)
    print(f"Using Generation Config -> max_tokens: {std_config['max_tokens']}, temperature: {std_config['temperature']}")

    print("Loading MMAR dataset...")
    dataset = load_dataset("BoJack/MMAR", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    audio_dir = None
    if args.audio_condition == "full":
        audio_dir = download_mmar_audio()

    print(f"Initializing vLLM with {args.model_id}")
    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        max_model_len=32768,
        tokenizer_mode="mistral",
    )
    tokenizer = llm.get_tokenizer()
    mistral_tokenizer = tokenizer.mistral

    batch_inputs: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    silence_audio = np.zeros(16000, dtype=np.float32)

    # Resume: load existing results to skip already-processed samples
    existing_ids: set = set()
    if args.resume and Path(args.output_path).exists():
        try:
            _existing = json.loads(Path(args.output_path).read_text())
            if isinstance(_existing, list):
                for _row in _existing:
                    existing_ids.add(str(_row.get("id", "")))
                results.extend(_existing)
            print(f"Resume: {len(existing_ids)} existing results in {args.output_path}")
        except Exception as _e:
            print(f"Resume: could not load existing results ({_e}), starting fresh")
            existing_ids.clear()
            results.clear()

    sampling_params = SamplingParams(
        max_tokens=std_config["max_tokens"],
        temperature=std_config["temperature"],
        top_p=std_config.get("top_p", 1.0),
    )

    def flush_batch() -> None:
        nonlocal batch_inputs, metadata
        if not batch_inputs:
            return
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        except Exception as e:
            err_msg = str(e)
            for meta in metadata:
                results.append({
                    **meta,
                    "model_prediction": "",
                    "error": f"generation_error: {err_msg}",
                })
            batch_inputs, metadata = [], []
            return

        for i, meta in enumerate(metadata):
            prediction = ""
            output_missing = i >= len(outputs) or not outputs[i].outputs
            if not output_missing:
                prediction = outputs[i].outputs[0].text.strip()
            record = {**meta, "model_prediction": prediction}
            if output_missing:
                record["error"] = "missing_generation_output"
            results.append(record)
        batch_inputs, metadata = [], []
    
    print("Preparing inputs...")
    for idx, sample in enumerate(tqdm(dataset)):
        # Skip already-processed samples when resuming
        if existing_ids and str(sample.get("id", idx)) in existing_ids:
            continue

        question = sample.get("question") or sample.get("text", "")
        choices = normalize_choices(sample.get("choices") or sample.get("options"))
        prompt_text = format_mcq_prompt(question, choices)
        
        audio = None
        audio_source = "text_only"
        if args.audio_condition == "full" and audio_dir:
            audio_path = sample.get("audio_path", "")
            if audio_path:
                audio = load_audio_from_path(audio_path, audio_dir)
                if audio is not None:
                    # Apply chunking if requested
                    if args.num_chunks is not None:
                        audio = chunk_audio(audio, args.num_chunks, args.chunk_index)
                        if audio is None:
                            audio = silence_audio
                            audio_source = "silence_fallback_audio_too_short"
                        else:
                            audio_source = "audio_file"
                    else:
                        audio_source = "audio_file"
                else:
                    audio = silence_audio
                    audio_source = "silence_fallback_missing_audio"
            else:
                audio = silence_audio
                audio_source = "silence_fallback_missing_audio"
        elif args.audio_condition == "silence":
            audio = silence_audio
            audio_source = "silence"

        # Tokenize with mistral_common
        try:
            content_chunks = [TextChunk(text=prompt_text)]
            if audio is not None:
                audio_obj = MistralAudio(audio_array=audio, sampling_rate=16000, format="wav")
                audio_chunk = AudioChunk(input_audio=RawAudio.from_audio(audio_obj))
                content_chunks = [audio_chunk, TextChunk(text=prompt_text)]
            
            # Construct message
            request = ChatCompletionRequest(
                messages=[UserMessage(content=content_chunks)]
            )
            encoded = mistral_tokenizer.encode_chat_completion(request)
            
            input_record: Dict[str, Any] = {
                "prompt_token_ids": encoded.tokens,
            }
            if audio is not None:
                processed_audio = encoded.audios[0].audio_array if encoded.audios else audio
                input_record["multi_modal_data"] = {"audio": processed_audio}

            batch_inputs.append(input_record)
            
            metadata.append({
                "id": sample.get("id", idx),
                "question": question,
                "choices": choices,
                "answer": sample.get("answer", ""),
                "modality": sample.get("modality", "unknown"),
                "category": sample.get("category", "unknown"),
                "sub-category": sample.get("sub-category") or sample.get("sub_category"),
                "audio_condition": args.audio_condition,
                "model_id": args.model_id,
                "audio_source": audio_source,
            })

            if len(batch_inputs) >= batch_size:
                flush_batch()
            
        except Exception as e:
            print(f"Error encoding sample {idx}: {e}")
            results.append({
                "id": sample.get("id", idx),
                "question": question,
                "choices": choices,
                "answer": sample.get("answer", ""),
                "modality": sample.get("modality", "unknown"),
                "category": sample.get("category", "unknown"),
                "sub-category": sample.get("sub-category") or sample.get("sub_category"),
                "audio_condition": args.audio_condition,
                "model_id": args.model_id,
                "audio_source": audio_source,
                "model_prediction": "",
                "error": f"encoding_error: {e}",
            })

    flush_batch()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} samples to {args.output_path}")

if __name__ == "__main__":
    main()
