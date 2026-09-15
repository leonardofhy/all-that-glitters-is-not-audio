#!/usr/bin/env python3
"""
MMAU Benchmark Inference Script for Phi-4-Multimodal (HuggingFace Transformers)

Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True,
which correctly loads MoLoRA adapters (speech/vision) via PEFT.
The vLLM backend skips LoRA weights, so this HF version is required
for correct multimodal inference.

Single-sample loop with incremental JSONL saving.
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mmau.utils import (
    extract_mmau_mini_audio,
    parse_other_attributes,
    extract_sample_id,
    format_mcq_prompt,
    get_mini_audio_filename,
    chunk_audio,
)
from src.config_utils import load_generation_config

# Patch DynamicCache for transformers>=4.50 compatibility with Phi-4's custom code
from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
        self.get_seq_length(layer_idx)
    )


def load_audio_file(audio_path: str, target_sr: int = 16000) -> np.ndarray | None:
    if not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=target_sr)
        return audio
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Phi-4-Multimodal HF Inference on MMAU-Mini"
    )
    parser.add_argument("--model_id", type=str,
                        default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--output_path", type=str,
                        default="results/mmau/phi4_multimodal/full.jsonl")
    parser.add_argument("--audio_condition", type=str, default="full",
                        choices=["full", "none"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--chunk_index", type=int, default=None)
    parser.add_argument("--use_mini", action="store_true", default=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Alias for --model_id")
    args = parser.parse_args()

    if args.model_path is not None:
        args.model_id = args.model_path

    # Generation config
    std_config = load_generation_config(benchmark="mmau", model_id=args.model_id)
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    print(f"Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Validate chunking
    if args.num_chunks is not None:
        if args.chunk_index is None:
            parser.error("--chunk_index required when --num_chunks is set")
        if args.num_chunks < 2:
            parser.error("--num_chunks must be >= 2")
        if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
            parser.error(f"--chunk_index must be in [0, {args.num_chunks})")
        if args.audio_condition == "none":
            parser.error("--num_chunks is incompatible with --audio_condition none")

    # Audio
    audio_dir = None
    if args.audio_condition != "none":
        print("Ensuring audio extraction...")
        audio_dir = extract_mmau_mini_audio()

    # Dataset
    print("Loading MMAU test-mini dataset...")
    try:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
    except Exception:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini")["test"]

    for col in ["audio", "context"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    # Model (HF with MoLoRA via PEFT)
    # Phi-4's modeling code calls get_peft_model(self.model, ...) which expects
    # prepare_inputs_for_generation on the inner model. Patch it before loading.
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # Import the custom model module to patch it
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"],
        args.model_id,
        trust_remote_code=True,
    )
    # Patch the inner model class to have prepare_inputs_for_generation
    inner_model_module = sys.modules.get(model_class.__module__)
    if inner_model_module:
        inner_model_cls = getattr(inner_model_module, "Phi4MMModel", None)
        if inner_model_cls and not hasattr(inner_model_cls, "prepare_inputs_for_generation"):
            inner_model_cls.prepare_inputs_for_generation = lambda self, *args, **kwargs: {}

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
    import functools
    @functools.wraps(_orig_forward)
    def _patched_forward(*args, **kwargs):
        if kwargs.get("num_logits_to_keep") is None:
            kwargs["num_logits_to_keep"] = 1
        return _orig_forward(*args, **kwargs)
    model.forward = _patched_forward

    # Output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(dataset)} samples...")
    with open(output_path, "w") as f:
        pass  # truncate

    for i, item in enumerate(tqdm(dataset)):
        try:
            question = item.get("instruction", "")
            choices = item.get("choices", [])
            attrs = parse_other_attributes(item.get("other_attributes", {}))
            user_msg = format_mcq_prompt(question, choices)

            # Build prompt
            if args.audio_condition != "none":
                prompt = f"<|user|><|audio_1|>{user_msg}<|end|><|assistant|>"
            else:
                prompt = f"<|user|>{user_msg}<|end|><|assistant|>"

            # Audio loading
            audio_tuple = None
            if args.audio_condition != "none":
                filename = get_mini_audio_filename(item, attrs, i)
                full_path = str(audio_dir / filename)
                audio_array = load_audio_file(full_path)

                if audio_array is None:
                    audio_array = np.zeros(16000, dtype=np.float32)
                elif args.num_chunks is not None:
                    chunked = chunk_audio(audio_array, args.num_chunks, args.chunk_index)
                    if chunked is None:
                        print(f"Warning: Audio too short to chunk, logging error for sample {i}")
                        raise ValueError("Audio too short to chunk")
                    audio_array = chunked

                audio_tuple = [(audio_array, 16000)]

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
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error on sample {i}: {e}")
            pred = f"[ERROR] {e}"

        result = {
            "id": extract_sample_id(item, attrs, i),
            "question": question,
            "choices": choices,
            "ground_truth": item.get("answer"),
            "task": attrs.get("task", ""),
            "category": attrs.get("category", ""),
            "sub_category": attrs.get("sub-category", ""),
            "difficulty": attrs.get("difficulty", ""),
            "dataset": attrs.get("dataset", ""),
            "prediction": pred,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
