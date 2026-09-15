#!/usr/bin/env python3
"""
MMAR Benchmark Inference Script for Phi-4-Multimodal (HuggingFace Transformers)

Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True,
which correctly loads MoLoRA adapters (speech/vision) via PEFT.
The vLLM backend skips LoRA weights, so this HF version is required
for correct multimodal inference.

Single-sample loop with incremental JSON saving.
"""

import argparse
import functools
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config_utils import load_generation_config

# ---------------------------------------------------------------------------
# Patch DynamicCache for transformers>=4.50 compatibility with Phi-4's custom code
# ---------------------------------------------------------------------------
from transformers import DynamicCache

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
        self.get_seq_length(layer_idx)
    )

# ---------------------------------------------------------------------------
# MMAR audio cache
# ---------------------------------------------------------------------------
MMAR_CACHE_DIR = Path.home() / ".cache" / "mmar"


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
    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract_tar(tar, cache_dir)

    num_files = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    print(f"Extracted {num_files} audio files to {audio_dir}")
    return audio_dir


# ---------------------------------------------------------------------------
# MMAR helpers (copied from run_inference_vllm.py)
# ---------------------------------------------------------------------------

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

    options_str = "\n".join(
        [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)]
    )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Phi-4-Multimodal HF Inference on MMAR"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Alias for --model_id",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/mmar/phi4_multimodal/full.json",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="Output format. 'auto' infers from extension ('.jsonl' -> jsonl, else json)",
    )
    parser.add_argument(
        "--audio_condition",
        type=str,
        default="full",
        choices=["full", "none", "silence"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Split audio into N equal chunks (N>=2)",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=None,
        help="0-indexed chunk to use (required with --num_chunks)",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Max audio duration in seconds (None = no limit)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in output file (append mode)",
    )
    args = parser.parse_args()

    if args.model_path is not None:
        args.model_id = args.model_path

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
    std_config = load_generation_config(benchmark="mmar", model_id=args.model_id)
    temperature = (
        args.temperature if args.temperature is not None else std_config["temperature"]
    )
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    )
    print(f"Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Load dataset
    print("Loading MMAR dataset...")
    dataset = load_dataset("BoJack/MMAR", split=args.split)
    print(f"Loaded {len(dataset)} samples")

    # Download audio files if needed
    audio_dir = None
    if args.audio_condition in ("full", "silence"):
        audio_dir = download_mmar_audio()

    if args.limit > 0 and args.limit < len(dataset):
        dataset = dataset.select(range(args.limit))
        print(f"Selected first {args.limit} samples")

    # ------------------------------------------------------------------
    # Load model (HF with MoLoRA via PEFT)
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Phi-4's modeling code calls get_peft_model(self.model, ...) which expects
    # prepare_inputs_for_generation on the inner model. Patch it before loading.
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"],
        args.model_id,
        trust_remote_code=True,
    )

    # Patch the inner model class to have prepare_inputs_for_generation
    inner_model_module = sys.modules.get(model_class.__module__)
    if inner_model_module:
        inner_model_cls = getattr(inner_model_module, "Phi4MMModel", None)
        if inner_model_cls and not hasattr(
            inner_model_cls, "prepare_inputs_for_generation"
        ):
            inner_model_cls.prepare_inputs_for_generation = (
                lambda self, *args, **kwargs: {}
            )

    model = model_class.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation="eager",
    )
    model.eval()

    # Patch forward to handle num_logits_to_keep=None (transformers >=4.50 compat)
    _orig_forward = model.forward

    @functools.wraps(_orig_forward)
    def _patched_forward(*args, **kwargs):
        if kwargs.get("num_logits_to_keep") is None:
            kwargs["num_logits_to_keep"] = 1
        return _orig_forward(*args, **kwargs)

    model.forward = _patched_forward

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = resolve_output_format(output_path, args.output_format)

    json_results: List[Dict[str, Any]] = []
    existing_ids: set = set()

    # Resume: load existing results to skip already-processed samples
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
        with open(output_path, "w"):
            pass  # truncate for fresh run

    silence_audio = np.zeros(16000, dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference loop (single-sample, no batching for HF generate)
    # ------------------------------------------------------------------
    print(
        f"\nRunning inference on {len(dataset)} samples "
        f"(condition: {args.audio_condition})..."
    )

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

        user_msg = format_mcq_prompt(question, choices)

        # Audio loading
        audio_array = None
        audio_source = "text_only"

        if args.audio_condition == "full" and audio_dir is not None:
            audio_path = sample.get("audio_path", "")
            if audio_path:
                max_dur = None if args.num_chunks is not None else args.max_duration
                audio_array = load_audio_from_path(
                    audio_path, audio_dir, max_duration_sec=max_dur
                )

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

        # Build Phi-4 prompt
        if args.audio_condition != "none" and audio_array is not None:
            prompt = f"<|user|><|audio_1|>{user_msg}<|end|><|assistant|>"
        else:
            prompt = f"<|user|>{user_msg}<|end|><|assistant|>"

        # Prepare audio tuple for processor
        audio_tuple = None
        if args.audio_condition != "none" and audio_array is not None:
            audio_tuple = [(audio_array, 16000)]

        try:
            # Process inputs
            inputs = processor(
                text=prompt,
                audios=audio_tuple,
                return_tensors="pt",
            ).to(model.device)

            # Cast float tensors to model dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=model.dtype)

            # Generate
            with torch.inference_mode():
                generate_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                )
                if temperature > 0:
                    generate_kwargs["temperature"] = temperature
                outputs = model.generate(**generate_kwargs)

            # Decode (skip input tokens)
            pred = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error on sample {idx}: {e}")
            pred = f"[ERROR] {e}"

        record = {
            "id": sample.get("id", idx),
            "question": question,
            "choices": choices,
            "answer": answer,
            "modality": modality,
            "category": category,
            "sub-category": sub_category,
            "audio_source": audio_source,
            "model_prediction": pred,
            "model_id": args.model_id,
            "audio_condition": args.audio_condition,
        }

        # Incremental save
        if output_format == "jsonl":
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json_results.append(record)
            # Write full JSON array each time for crash resilience
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)

    # Final save for JSON format (ensures clean output)
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        n_results = len(json_results)
    else:
        with open(output_path) as f:
            n_results = sum(1 for _ in f)

    print(f"\nDone -- {n_results} predictions saved to {output_path}")


if __name__ == "__main__":
    main()
