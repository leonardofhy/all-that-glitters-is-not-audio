#!/usr/bin/env python3
"""
Standalone Open-Ended Evaluation Script

Evaluates Open-Ended (free text) model responses using Qwen2.5-7B-Instruct as a Judge LLM.
This script runs in isolation to avoid CUDA context conflicts with NV-Embed.

Usage:
    python scripts/evaluate_openended.py \
        --input_file results/mmau_pro__xxx.jsonl \
        --output_file results/mmau_pro__xxx_openended_results.json
"""

import argparse
import json
import torch
import pandas as pd

from scripts.mmau_pro.eval_utils import (
    remove_thinking_process,
    load_qwen_model,
    evaluate_openended_with_qwen,
    calculate_openended_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Standalone Open-Ended Evaluation Script")
    parser.add_argument('--input_file', required=True, help='JSONL file with model predictions')
    parser.add_argument('--output_file', required=True, help='Output JSON file for results')
    parser.add_argument('--model_output_column', default='prediction', help='Column name for model output')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for Judge LLM')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = pd.read_json(args.input_file, lines=True)
    
    # Filter to open-ended samples only
    open_df = df[df['category'] == 'open']
    print(f"Found {len(open_df)} Open-Ended samples")
    
    if len(open_df) == 0:
        print("No Open-Ended samples found. Exiting.")
        results = {"type": "openended", "count": 0, "metrics": {}}
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return
    
    # Load Judge LLM
    model, tokenizer = load_qwen_model()
    
    # Prepare data — strip <think> blocks so judge evaluates only the final answer
    questions = open_df['question'].tolist()
    reference_answers = open_df['ground_truth'].fillna("").tolist()
    model_responses = [remove_thinking_process(r) for r in open_df[args.model_output_column].fillna("").tolist()]
    task_types = ['open'] * len(open_df)
    
    # Evaluate
    all_scores = evaluate_openended_with_qwen(
        model, tokenizer,
        questions, reference_answers, model_responses, task_types,
        batch_size=args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_openended_metrics(all_scores)
    
    # Save results
    results = {
        "type": "openended",
        "count": len(open_df),
        "metrics": metrics,
        "per_sample_scores": all_scores
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Open-Ended Evaluation Results ===")
    print(f"Samples: {len(open_df)}")
    print(f"Avg Overall: {metrics.get('avg_overall', 0):.2f}/5.0")
    print(f"Good Response Rate (>=4.0): {metrics.get('good_response_rate', 0)*100:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
