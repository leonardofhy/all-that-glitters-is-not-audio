#!/usr/bin/env python3
"""
MMAR Benchmark Evaluation Script

Evaluates MMAR inference results (MCQ-only).
string_match() is identical to official MMAR eval.
Supports both JSON and JSONL input formats.
"""

import argparse
import json
import re
from pathlib import Path


def remove_thinking_process(text):
    """Remove <think>...</think> blocks from model output."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Handle </think> without opening <think> (vLLM strips opening tag for some models)
    cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def string_match(answer, prediction, choices):
    """Official MMAR string-match scoring (identical to official eval)."""
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


def load_input_data(input_path: str) -> list:
    """Load input data from either JSON or JSONL format."""
    with open(input_path, 'r') as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # JSON array format
            return json.load(f)
        else:
            # JSONL format (one JSON object per line)
            return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate MMAR inference results.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON/JSONL file')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON results file')

    args = parser.parse_args()

    input_data = load_input_data(args.input)
    print(f"Loaded {len(input_data)} samples from {args.input}")

    corr, total = 0, 0

    # Track metrics for different categories:
    modality_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0], 'mix-sound-music': [0, 0], 'mix-sound-speech': [0, 0], 'mix-music-speech': [0, 0], 'mix-sound-music-speech': [0, 0]}
    category_metrics = {'Signal Layer': [0, 0], 'Perception Layer': [0, 0], 'Semantic Layer': [0, 0], 'Cultural Layer': [0, 0]}

    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'model_prediction'  # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    for idx, sample in enumerate(input_data):

        # If there's no model output key, skip
        if output_key not in sample:
            no_pred_count += 1
            continue

        raw_prediction = sample.get(output_key, "")
        _prediction = remove_thinking_process(raw_prediction)
        if not str(raw_prediction).strip() or not _prediction:
            no_pred_count += 1

        _answer = sample['answer']
        modality = sample['modality']
        category = sample['category']
        choices = sample['choices']

        # Get the sub-category
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    # Print results:
    print("*" * 30)
    print("Modality-wise Accuracy:")
    for modality in modality_metrics:
        n_correct, n_total = modality_metrics[modality]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{modality} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Category-wise Accuracy:")
    for category in category_metrics:
        n_correct, n_total = category_metrics[category]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{category} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    total_acc = (corr / total) * 100 if total > 0 else 0
    print(f"Total Accuracy: {total_acc:.2f}% over {total} samples")
    print("*" * 30)
    print(f"No prediction count: {no_pred_count}")

    # Save structured results to JSON
    output_data = {
        "overall_accuracy": total_acc / 100,
        "overall_correct": corr,
        "overall_total": total,
        "modality_accuracy": {
            mod: {
                "accuracy": (vals[0] / vals[1]) if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for mod, vals in modality_metrics.items()
        },
        "category_accuracy": {
            cat: {
                "accuracy": (vals[0] / vals[1]) if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for cat, vals in category_metrics.items()
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

    if args.output_file is None:
        output_file = str(Path(args.input).with_suffix('')) + "_results.json"
    else:
        output_file = args.output_file

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
