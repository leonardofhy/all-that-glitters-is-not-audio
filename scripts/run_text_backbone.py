#!/usr/bin/env python3
"""
Text Backbone LLM Inference Script

Runs pure text LLMs (no audio capability) on MMAU, MMAU-Pro, and MMAR benchmarks.
Used to measure raw text-only performance of backbone LLMs that underlie audio models.

Uses vLLM directly (not AudioLLMEngine) since these are standard text models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Benchmark-specific loaders and formatters
# ============================================================

def load_mmau(num_samples: int = 0):
    """Load MMAU-test-mini dataset (1000 MCQ samples, matching existing results)."""
    from datasets import load_dataset

    dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test")
    # Drop audio column to avoid torchcodec/ffmpeg decoding (text-only inference)
    if "context" in dataset.column_names:
        dataset = dataset.remove_columns(["context"])
    print(f"Loaded MMAU-mini: {len(dataset)} samples")
    if num_samples > 0 and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"Selected first {num_samples} samples")
    return dataset


def load_mmau_pro(num_samples: int = 0):
    """Load MMAU-Pro dataset (5305 samples: MCQ + open-ended + AIF)."""
    from scripts.mmau_pro.utils import load_mmau_pro_dataset

    return load_mmau_pro_dataset(num_samples=num_samples)


def load_mmar(num_samples: int = 0):
    """Load MMAR dataset (1000 MCQ samples)."""
    from datasets import load_dataset

    dataset = load_dataset("BoJack/MMAR", split="test")
    print(f"Loaded MMAR: {len(dataset)} samples")
    if num_samples > 0 and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"Selected first {num_samples} samples")
    return dataset


def format_and_build_mmau(dataset, idx: int, sample: dict):
    """Format prompt and build result dict for MMAU-mini."""
    from scripts.mmau.utils import (
        format_mcq_prompt, parse_other_attributes, extract_sample_id,
        build_mmau_result,
    )

    question = sample.get("instruction", "")
    choices = sample.get("choices", [])
    prompt = format_mcq_prompt(question, choices)

    attrs = parse_other_attributes(sample.get("other_attributes", {}))
    meta = build_mmau_result(sample, attrs=attrs, prediction="", index=idx)
    # Remove the empty prediction — will be filled in after generation
    meta.pop("prediction", None)
    return prompt, meta


def format_and_build_mmau_pro(dataset, idx: int, sample: dict):
    """Format prompt and build result dict for MMAU-Pro."""
    from scripts.mmau_pro.utils import format_prompt

    question = sample.get("question", "")
    choices = sample.get("choices", [])
    category = sample.get("category", "")
    prompt = format_prompt(question, choices, category)
    return prompt, {
        "id": sample.get("id", ""),
        "category": category,
        "question": question,
        "choices": choices,
        "ground_truth": sample.get("answer", ""),
        "task_identifier": sample.get("task_identifier"),
        "kwargs": sample.get("kwargs"),
        "prompt_transcription": sample.get("prompt_transcription"),
    }


def format_and_build_mmar(dataset, idx: int, sample: dict):
    """Format prompt and build result dict for MMAR."""
    # Inline the helpers to avoid importing from run_inference_vllm.py
    # (which has side effects from AudioLLMEngine import)
    question = ""
    for key in ["question", "query", "prompt", "instruction", "text"]:
        if key in sample and sample[key] is not None:
            question = str(sample[key])
            break

    raw_choices = sample.get("choices") or sample.get("options")
    if raw_choices is None:
        choices = []
    elif isinstance(raw_choices, dict):
        items = sorted(raw_choices.items(), key=lambda x: x[0])
        choices = [str(v) for _, v in items]
    elif isinstance(raw_choices, (list, tuple)):
        choices = [str(c) for c in raw_choices]
    else:
        choices = [str(raw_choices)]

    if not choices:
        prompt = f"{question}\n\nProvide your answer:"
    else:
        options_str = "\n".join(
            f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)
        )
        prompt = (
            f"{question}\n\n"
            f"Options:\n{options_str}\n\n"
            "Answer with the full option text (you may include the letter)."
        )

    return prompt, {
        "id": sample.get("id", idx),
        "question": question,
        "choices": choices,
        "answer": sample.get("answer", ""),
        "modality": sample.get("modality", "unknown"),
        "category": sample.get("category", "unknown"),
        "sub-category": sample.get("sub-category") or sample.get("sub_category"),
        "audio_source": "text_only",
    }


# ============================================================
# Benchmark dispatch
# ============================================================

BENCHMARK_LOADERS = {
    "mmau": load_mmau,
    "mmau_pro": load_mmau_pro,
    "mmar": load_mmar,
}

BENCHMARK_FORMATTERS = {
    "mmau": format_and_build_mmau,
    "mmau_pro": format_and_build_mmau_pro,
    "mmar": format_and_build_mmar,
}

# MMAR uses JSON array; MMAU and MMAU-Pro use JSONL
BENCHMARK_FORMAT = {
    "mmau": "jsonl",
    "mmau_pro": "jsonl",
    "mmar": "json",
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run text backbone LLM inference on audio benchmarks (no audio)"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["mmau", "mmau_pro", "mmar"],
                        help="Benchmark to run")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save results")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: from generation_params.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from generation_params.yaml)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph compilation")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode: use thinking-model generation params "
                             "(temp=0.6, max_tokens=4096) and don't suppress thinking in "
                             "chat template. Auto-inferred from model_id if it contains 'thinking'.")
    args = parser.parse_args()

    # Auto-infer thinking mode from model ID
    if not args.enable_thinking and "thinking" in args.model_id.lower():
        args.enable_thinking = True
        print("Auto-detected thinking model: enabling thinking mode")

    # Generation config (yaml defaults, CLI overrides)
    # When thinking is enabled, keep "thinking" in model ID so config_utils applies
    # thinking model-type overrides (temp=0.6, max_tokens=4096).
    # When disabled, strip "thinking" to get standard non-thinking params.
    from src.config_utils import load_generation_config
    model_id_for_config = args.model_id
    if not args.enable_thinking and "thinking" in args.model_id.lower():
        model_id_for_config = args.model_id.replace("-Thinking", "").replace("-thinking", "")
    std_config = load_generation_config(benchmark=args.benchmark, model_id=model_id_for_config)
    max_tokens = args.max_tokens if args.max_tokens is not None else std_config["max_tokens"]
    temperature = args.temperature if args.temperature is not None else std_config["temperature"]
    print(f"Generation config: max_tokens={max_tokens}, temperature={temperature}")

    # Load dataset
    dataset = BENCHMARK_LOADERS[args.benchmark](num_samples=args.num_samples)
    formatter = BENCHMARK_FORMATTERS[args.benchmark]
    output_format = BENCHMARK_FORMAT[args.benchmark]

    # Initialize vLLM (text-only, no multimodal)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\nLoading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Initializing vLLM: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
    )

    # Prepare all prompts
    print(f"\nPreparing {len(dataset)} prompts...")
    all_prompts = []
    all_meta = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        prompt_text, meta = formatter(dataset, idx, sample)

        # Apply chat template
        messages = [{"role": "user", "content": prompt_text}]
        if tokenizer.chat_template is not None:
            template_kwargs = dict(add_generation_prompt=True, tokenize=False)
            if not args.enable_thinking:
                template_kwargs["enable_thinking"] = False
            try:
                full_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # enable_thinking not supported by this tokenizer
                template_kwargs.pop("enable_thinking", None)
                full_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            # Fallback: manual ChatML for models without chat_template (e.g. Qwen-7B-Chat)
            full_prompt = (
                f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        all_prompts.append(full_prompt)
        all_meta.append(meta)

    # Generate all at once — vLLM's continuous batching handles scheduling
    print(f"\nRunning inference on {len(all_prompts)} prompts...")
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    all_responses = [o.outputs[0].text.strip() for o in outputs]

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        with open(output_path, "w") as f:
            for meta, response in zip(all_meta, all_responses):
                meta["prediction"] = response
                f.write(json.dumps(meta) + "\n")
    else:
        # JSON array (MMAR)
        results = []
        for meta, response in zip(all_meta, all_responses):
            meta["model_prediction"] = response
            meta["model_id"] = args.model_id
            meta["audio_condition"] = "none"
            results.append(meta)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_responses)} results to {output_path}")


if __name__ == "__main__":
    main()
