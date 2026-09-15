#!/usr/bin/env python3
"""
Batched MMAU-Pro Evaluation — GPU-optimized variant of evaluate.py

Key differences from evaluate.py:
- Loads NVEmbed + Qwen judge ONCE and reuses across all input files
- Qwen judge uses batched inference (left-padding, configurable batch_size)
  instead of one sample at a time — 8-16x faster on a single large GPU
- Output format identical to evaluate.py (_comprehensive_results.json)

Usage:
    python scripts/mmau_pro/evaluate_batch.py results/mmau_pro/*/n*.jsonl
    python scripts/mmau_pro/evaluate_batch.py results/mmau_pro/phi4_multimodal/n*.jsonl --judge_batch_size 32
"""

import pandas as pd
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
import torch
import torch.nn.functional as F
import json
import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from scripts.mmau_pro.eval_utils import (
    remove_thinking_process,
    extract_scores_from_evaluation,
    create_evaluation_prompt,
    calculate_openended_metrics,
)
from scripts.mmau_pro.evaluate import (
    _patch_nvembed_for_transformers_compat,
    evaluate_aif_sample,
    calculate_metrics,
    calculate_weighted_performance,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore")


# ================================
# Batched Qwen Judge
# ================================

def evaluate_openended_with_qwen_batched(
    model, tokenizer, questions, reference_answers, model_responses, task_types,
    batch_size=16,
):
    """Batched Qwen2.5 judge evaluation.

    Uses left-padding so all sequences in a batch align on the right (causal LM).
    Output format identical to eval_utils.evaluate_openended_with_qwen.
    """
    all_scores = []
    detailed_evaluations = []

    # Build all prompt texts upfront
    texts = []
    for question, ref_answer, model_response, task_type in zip(
        questions, reference_answers, model_responses, task_types
    ):
        eval_prompt = create_evaluation_prompt(question, ref_answer, model_response, task_type)
        messages = [
            {"role": "system", "content": "You are a helpful and objective evaluator."},
            {"role": "user", "content": eval_prompt},
        ]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Switch to left-padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    print(f"Performing Qwen 2.5 LLM judge evaluation (batch_size={batch_size}, {len(texts)} samples)...")

    for batch_start in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, min(batch_start + batch_size, len(texts))))

        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Slice off prompt tokens
            new_ids = [
                out[len(inp):]
                for inp, out in zip(model_inputs.input_ids, generated_ids)
            ]
            evaluation_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: Batch {batch_start//batch_size} failed ({e}), using defaults")
            evaluation_texts = [""] * len(batch_texts)

        for i, (idx, evaluation_text) in enumerate(zip(batch_indices, evaluation_texts)):
            try:
                scores = extract_scores_from_evaluation(evaluation_text)
            except Exception:
                scores = {
                    'correctness': 3.0, 'relevance': 3.0,
                    'completeness': 3.0, 'clarity': 3.0, 'overall': 3.0,
                }
            all_scores.append(scores)
            detailed_evaluations.append({
                'question': questions[idx],
                'reference_answer': reference_answers[idx],
                'model_response': model_responses[idx],
                'evaluation': evaluation_text,
                'scores': scores,
                'task_type': task_types[idx],
            })

    tokenizer.padding_side = original_padding_side
    return all_scores, detailed_evaluations


# ================================
# Per-file evaluation (models pre-loaded)
# ================================

def evaluate_file(filepath, nvembed_model, qwen_model, qwen_tokenizer,
                  judge_batch_size=16, model_output_column="model_output"):
    """Evaluate a single JSONL/JSON/parquet file using pre-loaded models."""
    res_path = os.path.join(
        os.path.dirname(os.path.abspath(filepath)),
        os.path.splitext(os.path.basename(filepath))[0] + "_comprehensive_results.json",
    )
    if os.path.exists(res_path):
        print(f"  SKIP (eval exists): {res_path}")
        return

    print(f"\n{'='*70}")
    print(f"Evaluating: {filepath}")
    print(f"{'='*70}")

    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif file_ext in ['.jsonl', '.json']:
        df = pd.read_json(filepath, lines=(file_ext == '.jsonl'))
        if 'prediction' in df.columns and model_output_column not in df.columns:
            df[model_output_column] = df['prediction']
    else:
        print(f"  ERROR: unsupported format {file_ext}")
        return

    if 'ground_truth' in df.columns and 'answer' not in df.columns:
        df['answer'] = df['ground_truth']
    if model_output_column not in df.columns:
        df[model_output_column] = df.get('answer', '')

    print(f"Loaded {len(df)} samples")
    input_filename = os.path.splitext(os.path.basename(filepath))[0]
    category_results = {}

    # ── Open-ended ────────────────────────────────────────────────────────
    open_df = df[df['category'] == 'open'].copy()
    if len(open_df) > 0:
        print(f"\nOpen-ended: {len(open_df)} samples")
        questions = open_df['question'].tolist()
        reference_answers = open_df['answer'].tolist()
        model_responses = [
            remove_thinking_process(r) for r in open_df[model_output_column].fillna("").tolist()
        ]
        task_types = ['open'] * len(open_df)

        openended_scores, openended_detailed = evaluate_openended_with_qwen_batched(
            qwen_model, qwen_tokenizer, questions, reference_answers,
            model_responses, task_types, batch_size=judge_batch_size,
        )
        openended_metrics = calculate_openended_metrics(openended_scores)
        category_results['open'] = {
            'type': 'openended',
            'count': len(open_df),
            'metrics': openended_metrics,
            'scores': openended_scores,
        }
        print(f"  Avg Overall: {openended_metrics.get('avg_overall', 0):.3f}/5.0")

    # ── AIF ───────────────────────────────────────────────────────────────
    aif_df = df[df['category'] == 'instruction following'].copy()
    if len(aif_df) > 0:
        print(f"\nAIF: {len(aif_df)} samples")
        aif_results = []
        for _, row in tqdm(aif_df.iterrows(), total=len(aif_df)):
            sample_data = {
                'task_identifier': row.get('task_identifier'),
                'kwargs': row.get('kwargs'),
                'prompt_transcription': row.get('question', ''),
            }
            aif_results.append(evaluate_aif_sample(
                str(row.get(model_output_column, '')), sample_data
            ))
        success_rate = np.mean([float(r) for r in aif_results])
        category_results['instruction following'] = {
            'type': 'aif',
            'count': len(aif_df),
            'success_rate': success_rate,
            'results': aif_results,
        }
        print(f"  Success rate: {success_rate:.3f}")

    # ── Closed-ended (NVEmbed) ────────────────────────────────────────────
    closed_categories = [c for c in df['category'].unique()
                         if c not in ['open', 'instruction following']]
    closed_df = df[df['category'].isin(closed_categories)].copy()
    closed_df = closed_df[closed_df['choices'].notna()].copy()
    closed_df = closed_df[closed_df['choices'].apply(
        lambda x: len(x) > 1 if hasattr(x, '__len__') else False
    )].copy()

    if len(closed_df) > 0:
        print(f"\nClosed-ended: {len(closed_df)} samples")
        questions = closed_df['question'].tolist()
        ground_truth_answers = closed_df['answer'].tolist()
        choices_list = [list(c) if hasattr(c, '__iter__') else [str(c)]
                        for c in closed_df['choices'].tolist()]
        model_predictions = closed_df[model_output_column].fillna('').tolist()
        task_types = closed_df['category'].tolist()

        predictions = []
        for question, choices, gt_answer, model_prediction, task_type in tqdm(
            zip(questions, choices_list, ground_truth_answers, model_predictions, task_types)
        ):
            clean_prediction = remove_thinking_process(model_prediction)
            pred_emb = nvembed_model.encode([clean_prediction], instruction="", max_length=4096)
            pred_emb = F.normalize(pred_emb, p=2, dim=1)
            choice_embs = nvembed_model.encode(choices, instruction="", max_length=4096)
            choice_embs = F.normalize(choice_embs, p=2, dim=1)
            scores = (pred_emb @ choice_embs.T).squeeze()
            predictions.append(choices[torch.argmax(scores).item()])

        overall_metrics = calculate_metrics(ground_truth_answers, predictions)
        for category in closed_categories:
            cat_mask = closed_df['category'] == category
            if cat_mask.sum() > 0:
                cat_gt = [ground_truth_answers[i] for i, m in enumerate(cat_mask) if m]
                cat_pred = [predictions[i] for i, m in enumerate(cat_mask) if m]
                category_results[category] = {
                    'type': 'closed',
                    'count': int(cat_mask.sum()),
                    'metrics': calculate_metrics(cat_gt, cat_pred),
                }
        print(f"  Overall accuracy: {overall_metrics['accuracy']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    overall_weighted_performance, category_scores = calculate_weighted_performance(category_results)

    results_summary = {
        'evaluation_summary': {
            'total_samples': len(df),
            'evaluated_samples': sum(r['count'] for r in category_results.values()),
            'parquet_file': filepath,
            'model_output_column': model_output_column,
            'overall_weighted_performance': overall_weighted_performance,
        },
        'category_results': {},
    }
    for category, result in category_results.items():
        json_result = {
            'type': result['type'],
            'count': int(result['count']),
            'performance_score': category_scores.get(category, 0.0),
        }
        if result['type'] == 'openended':
            json_result['metrics'] = result['metrics']
        elif result['type'] == 'aif':
            json_result['success_rate'] = result['success_rate']
        elif result['type'] == 'closed':
            json_result['metrics'] = result['metrics']
        results_summary['category_results'][category] = json_result

    with open(res_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=float)

    print(f"\nSaved: {res_path}")
    print(f"Overall Weighted Performance: {overall_weighted_performance:.4f}")


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='Batched MMAU-Pro evaluation — loads models once, evaluates many files'
    )
    parser.add_argument('files', nargs='+', help='JSONL/JSON/parquet files to evaluate')
    parser.add_argument('--judge_batch_size', type=int, default=16,
                        help='Batch size for Qwen judge (default: 16)')
    parser.add_argument('--model_output_column', default='model_output')
    args = parser.parse_args()

    # Filter to files that need eval
    files_to_eval = []
    for f in sorted(args.files):
        res = os.path.join(
            os.path.dirname(os.path.abspath(f)),
            os.path.splitext(os.path.basename(f))[0] + "_comprehensive_results.json",
        )
        if not os.path.exists(f):
            print(f"SKIP (not found): {f}")
        elif os.path.exists(res):
            print(f"SKIP (eval exists): {f}")
        else:
            cnt = sum(1 for _ in open(f)) if f.endswith('.jsonl') else -1
            if cnt >= 0 and cnt < 5305:
                print(f"SKIP (incomplete {cnt}/5305): {f}")
            else:
                files_to_eval.append(f)

    if not files_to_eval:
        print("Nothing to evaluate.")
        return

    print(f"\n{len(files_to_eval)} files to evaluate with judge_batch_size={args.judge_batch_size}")

    # Load models once
    print("\nLoading NV-Embed-v2...")
    nvembed_model = AutoModel.from_pretrained(
        'nvidia/NV-Embed-v2', trust_remote_code=True, local_files_only=False
    )
    _patch_nvembed_for_transformers_compat(nvembed_model)
    nvembed_model.to('cuda')
    nvembed_model.eval()
    print("NVEmbed loaded.")

    print("\nLoading Qwen2.5-7B-Instruct (Judge)...")
    torch.cuda.empty_cache()
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    qwen_model.eval()
    print("Qwen judge loaded.")

    # Evaluate all files
    for i, filepath in enumerate(files_to_eval):
        print(f"\n[{i+1}/{len(files_to_eval)}] {filepath}")
        try:
            evaluate_file(
                filepath, nvembed_model, qwen_model, qwen_tokenizer,
                judge_batch_size=args.judge_batch_size,
                model_output_column=args.model_output_column,
            )
        except Exception as e:
            print(f"ERROR evaluating {filepath}: {e}")
            import traceback; traceback.print_exc()

    print("\n=== All evaluations complete ===")


if __name__ == "__main__":
    main()
