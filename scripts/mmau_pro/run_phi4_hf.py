#!/usr/bin/env python3
"""
MMAU-Pro Benchmark Inference Script for Phi-4-Multimodal (HuggingFace Transformers)

Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True,
which correctly loads MoLoRA adapters (speech/vision) via PEFT.
The vLLM backend skips LoRA weights, so this HF version is required
for correct multimodal inference.

Single-sample loop with incremental JSONL saving.
"""

import functools
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from scripts.mmau_pro.utils import (
    download_mmau_audio,
    load_audio_file,
    chunk_audio,
    format_prompt,
    get_sample_audio_path,
    build_result_dict,
    build_error_result_dict,
    setup_output_file,
    append_result_jsonl,
    load_completed_ids,
    add_common_args,
    load_mmau_pro_dataset,
    load_gen_config,
)

# ---------------------------------------------------------------------------
# Patch DynamicCache for transformers>=4.50 compatibility with Phi-4's
# custom modelling code which calls the removed get_usable_length method.
# ---------------------------------------------------------------------------
from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
        self.get_seq_length(layer_idx)
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MMAU-Pro inference for Phi-4-Multimodal (HF)"
    )
    add_common_args(parser)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: from generation_params.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from generation_params.yaml)")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Max audio duration in seconds (None = no limit)")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Split audio into N equal chunks (N>=2)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="0-indexed chunk to use (required with --num_chunks)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file")
    parser.set_defaults(
        model_id="microsoft/Phi-4-multimodal-instruct",
        output_path="results/mmau_pro/phi4_multimodal/full.jsonl",
    )
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

    # Generation config (yaml defaults, CLI overrides)
    std_config = load_gen_config(args.model_id)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    print(f"Using Generation Config -> max_tokens: {max_tokens}, temperature: {temperature}")

    # Dataset + audio
    dataset = load_mmau_pro_dataset(
        num_samples=args.num_samples,
        category=args.category,
    )
    data_dir = download_mmau_audio() if args.audio_condition != "none" else None

    # ------------------------------------------------------------------
    # Model loading (HF with MoLoRA via PEFT)
    #
    # Phi-4's modelling code calls get_peft_model(self.model, ...) which
    # expects prepare_inputs_for_generation on the inner model.  We patch
    # the class before from_pretrained so PEFT can wrap it.
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

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
            inner_model_cls.prepare_inputs_for_generation = lambda self, *a, **kw: {}

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
    # Resume support
    # ------------------------------------------------------------------
    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output_path)
        if completed_ids:
            print(f"Resuming: skipping {len(completed_ids)} already-processed samples")
    output_path = setup_output_file(args.output_path, resume=args.resume)

    chunk_info = f", chunk {args.chunk_index}/{args.num_chunks}" if args.num_chunks else ""
    print(f"\nRunning inference on {len(dataset)} samples "
          f"(condition: {args.audio_condition}{chunk_info})...")

    for i in tqdm(range(len(dataset)), desc="Inference"):
        sample = dataset[i]

        if sample["id"] in completed_ids:
            continue

        try:
            prompt_text = format_prompt(
                sample["question"], sample["choices"], sample.get("category", "")
            )

            # Build Phi-4 chat-template prompt
            if args.audio_condition != "none":
                prompt = f"<|user|><|audio_1|>{prompt_text}<|end|><|assistant|>"
            else:
                prompt = f"<|user|>{prompt_text}<|end|><|assistant|>"

            # Audio loading
            audio_tuple = None
            if args.audio_condition != "none":
                audio_ref = get_sample_audio_path(sample)
                if not audio_ref:
                    print(f"Warning: Sample {sample['id']} has no audio_path, skipping")
                    result = build_error_result_dict(sample, "Missing audio_path")
                    append_result_jsonl(output_path, result)
                    continue

                max_dur = None if args.num_chunks is not None else args.max_duration
                audio_array = load_audio_file(
                    audio_ref, data_dir, target_sr=16000,
                    max_duration_sec=max_dur,
                )
                if audio_array is None:
                    print(f"Skipping sample {sample['id']}: audio not loadable")
                    result = build_error_result_dict(sample, "Audio completely unloadable")
                    append_result_jsonl(output_path, result)
                    continue

                # Apply chunking if requested
                if args.num_chunks is not None:
                    audio_array = chunk_audio(audio_array, args.num_chunks, args.chunk_index)
                    if audio_array is None:
                        print(f"Skipping {sample['id']}: audio too short for {args.num_chunks} chunks")
                        result = build_error_result_dict(sample, "Audio too short to chunk")
                        append_result_jsonl(output_path, result)
                        continue

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
            print(f"Error on sample {sample['id']}: {e}")
            result = build_error_result_dict(sample, e)
            append_result_jsonl(output_path, result)
            continue

        result = build_result_dict(sample, prediction=pred)
        append_result_jsonl(output_path, result)

    print(f"\nDone — results saved to {output_path}")


if __name__ == "__main__":
    main()
