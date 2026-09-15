#!/usr/bin/env python3
"""
MMAU Benchmark Evaluation Script

Evaluates MMAU inference results (MCQ-only).
Implements official MMAU string-match scoring with task/difficulty/sub-category breakdown.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def remove_thinking_process(text):
    """Remove <think>...</think> blocks from model output"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
        
    # Remove standard <think> tags (and variations just in case)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Handle leaked/truncated <think> without closing tag (endpoint cutoff)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle </think> without opening <think> (vLLM strips opening tag for some models)
    cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2


def evaluate_mmau(input_file: str, output_file: str = None, output_key: str = "prediction"):
    """Evaluate MMAU inference results with official string-match logic.

    Supports both JSON (official format: list of dicts) and JSONL (our format: one dict per line).
    """
    with open(input_file) as f:
        content = f.read().strip()

    # Auto-detect JSON vs JSONL
    if content.startswith("["):
        results = json.loads(content)
    else:
        results = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"Loaded {len(results)} results from {input_file}")

    corr = 0
    total = 0
    no_pred_count = 0

    task_metrics = defaultdict(lambda: [0, 0])
    diff_metrics = defaultdict(lambda: [0, 0])
    subcat_metrics = defaultdict(lambda: [0, 0])

    for sample in results:
        if output_key not in sample:
            continue

        prediction = sample.get(output_key, "") or ""
        prediction = remove_thinking_process(prediction)
        
        if isinstance(prediction, list):
            prediction = " ".join(str(x) for x in prediction)
        if not prediction:
            no_pred_count += 1

        answer = sample.get("ground_truth", sample.get("answer", ""))
        task = sample.get("task", "unknown")
        difficulty = sample.get("difficulty", "unknown")
        choices = sample.get("choices", [])
        subcat = sample.get("sub_category", sample.get("sub-category", None))

        match_result = string_match(answer, prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            corr += 1
            sample["match"] = 1
        else:
            sample["match"] = 0

        total += 1
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    print("*" * 30)
    print("Task-wise Accuracy:")
    for task in sorted(task_metrics.keys()):
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    for diff in sorted(diff_metrics.keys()):
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat in sorted(subcat_metrics.keys()):
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    total_acc = (corr / total) * 100 if total > 0 else 0
    print(f"Total Accuracy: {total_acc:.2f}% over {total} samples")
    print("*" * 30)
    print(f"No prediction count: {no_pred_count}")

    output_data = {
        "overall_accuracy": total_acc / 100,
        "overall_correct": corr,
        "overall_total": total,
        "task_accuracy": {
            task: {
                "accuracy": (vals[0] / vals[1]) if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for task, vals in task_metrics.items()
        },
        "difficulty_accuracy": {
            diff: {
                "accuracy": (vals[0] / vals[1]) if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for diff, vals in diff_metrics.items()
        },
        "sub_category_accuracy": {
            subcat: {
                "accuracy": (vals[0] / vals[1]) if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for subcat, vals in subcat_metrics.items()
        },
        "no_prediction_count": no_pred_count,
    }

    if output_file is None:
        output_file = str(Path(input_file).with_suffix('')) + "_results.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMAU inference results")
    parser.add_argument("input_file", help="Path to inference JSONL file")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--output_key", type=str, default="prediction", help="Key containing model output")
    
    args = parser.parse_args()
    evaluate_mmau(args.input_file, args.output_file, args.output_key)


if __name__ == "__main__":
    main()
