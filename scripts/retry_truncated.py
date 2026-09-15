#!/usr/bin/env python3
"""
Selective retry for truncated thinking-model predictions.

Loads an existing results file, identifies entries where the model hit max_tokens
(no </think> closing tag), reruns only those entries with a higher token budget,
and patches the results back into the original file.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/retry_truncated.py \
        --results_file results/mmar/qwen3_omni_30b_a3b_thinking/none.json \
        --model_id "Qwen/Qwen3-Omni-30B-A3B-Thinking" \
        --benchmark mmar \
        --max_tokens 8192 \
        --max_model_len 16384
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_utils import load_generation_config


def identify_truncated(data, pred_key="prediction"):
    """Return indices of entries missing </think> (truncated thinking chain)."""
    truncated = []
    for i, d in enumerate(data):
        pred = d.get(pred_key, "")
        if pred and "</think>" not in pred:
            truncated.append(i)
    return truncated


def load_results(path):
    """Load results file (JSON array or JSONL)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()], "jsonl"
    else:
        with open(path) as f:
            return json.load(f), "json"


def save_results(data, path, fmt):
    """Save results file."""
    path = Path(path)
    if fmt == "jsonl":
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def format_prompt_mmar(sample):
    """Format MMAR prompt — identical to run_inference_vllm.py:format_mcq_prompt."""
    question = sample["question"]
    choices = sample["choices"]
    if not choices:
        return f"{question}\n\nProvide your answer:"
    options_str = "\n".join(
        f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer with the full option text (you may include the letter)."
    )


def format_prompt_mmau(sample):
    """Format MMAU prompt — identical to scripts/mmau/utils.py:format_mcq_prompt."""
    question = sample.get("question", "")
    choices = sample.get("choices", [])
    if not choices:
        return f"{question}\n\nProvide your answer:"
    options_str = "\n".join(
        f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with the full option text (you may include the letter)."
    )


def format_prompt_mmau_pro(sample):
    """Format MMAU-Pro prompt — identical to scripts/mmau_pro/run_qwen.py prompt format."""
    question = sample.get("question", "")
    choices = sample.get("choices", [])
    if not choices:
        return f"{question}\n\nProvide your answer:"
    options_str = "\n".join(
        f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with the full option text (you may include the letter)."
    )


FORMATTERS = {
    "mmar": format_prompt_mmar,
    "mmau": format_prompt_mmau,
    "mmau_pro": format_prompt_mmau_pro,
}

PRED_KEYS = {
    "mmar": "model_prediction",
    "mmau": "prediction",
    "mmau_pro": "prediction",
}


def main():
    parser = argparse.ArgumentParser(description="Retry truncated thinking-model predictions")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to existing results file (JSON or JSONL)")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["mmau", "mmau_pro", "mmar"])
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="New max_tokens for retried entries (default: 8192)")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="vLLM max model context length (default: 16384)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only identify truncated entries, don't rerun")
    args = parser.parse_args()

    pred_key = PRED_KEYS[args.benchmark]
    formatter = FORMATTERS[args.benchmark]

    # Load existing results
    print(f"Loading results from: {args.results_file}")
    data, fmt = load_results(args.results_file)
    print(f"Loaded {len(data)} entries (format: {fmt})")

    # Identify truncated entries
    trunc_indices = identify_truncated(data, pred_key)
    print(f"Found {len(trunc_indices)} truncated entries ({len(trunc_indices)/len(data)*100:.1f}%)")

    if not trunc_indices:
        print("No truncated entries found. Nothing to do.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would retry these entries:")
        for i in trunc_indices[:10]:
            pred = data[i].get(pred_key, "")
            print(f"  idx={i}, id={data[i].get('id','?')}, pred_len={len(pred)}")
        if len(trunc_indices) > 10:
            print(f"  ... and {len(trunc_indices) - 10} more")
        return

    # Backup original file
    backup_path = Path(args.results_file).with_suffix(
        ".pre_retry" + Path(args.results_file).suffix
    )
    save_results(data, backup_path, fmt)
    print(f"Backed up original to: {backup_path}")

    # Get generation config (temperature from yaml, max_tokens from CLI)
    gen_config = load_generation_config(benchmark=args.benchmark, model_id=args.model_id)
    temperature = gen_config["temperature"]  # 0.6 for thinking models

    print(f"\nRetry config: max_tokens={args.max_tokens}, temperature={temperature}")

    # Use AudioLLMEngine (same as original inference) — handles chat template via AutoProcessor
    from src.inference import AudioLLMEngine

    print(f"Initializing AudioLLMEngine: {args.model_id}")
    engine = AudioLLMEngine(
        model_id=args.model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"audio": 1},
    )

    # Prepare prompts for truncated entries only
    print(f"\nPreparing {len(trunc_indices)} prompts for retry...")
    prompts = []
    for idx in trunc_indices:
        sample = data[idx]
        prompts.append(formatter(sample))

    # Run inference via AudioLLMEngine (audios=None for text-only/none condition)
    print(f"Running inference on {len(prompts)} truncated entries...")
    responses = engine.generate(
        prompts=prompts,
        audios=None,
        temperature=temperature,
        max_tokens=args.max_tokens,
    )

    # Patch results
    patched = 0
    still_truncated = 0
    for idx, response in zip(trunc_indices, responses):
        data[idx][pred_key] = response
        if "</think>" in response:
            patched += 1
        else:
            still_truncated += 1

    # Save patched results
    save_results(data, args.results_file, fmt)
    print(f"\nResults saved to: {args.results_file}")
    print(f"Patched: {patched}/{len(trunc_indices)} entries now have </think>")
    if still_truncated:
        print(f"Still truncated: {still_truncated} entries (may need even higher max_tokens)")


if __name__ == "__main__":
    main()
