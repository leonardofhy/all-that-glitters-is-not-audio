#!/usr/bin/env python3
"""
Comprehensive MMAU-Pro Evaluation Script — Aligned with Official Eval

Aligned with the official gamma-lab-umd/MMAU-Pro evaluation script:
- Closed-ended: NVEmbed-only matching (no regex/keyword fallback)
- Open-ended: In-process Qwen2.5-7B-Instruct LLM judge
- AIF: Rule-based format checking
- Output: {input_filename}_comprehensive_results.json

Additions over official (needed for our pipeline):
- Supports JSONL/JSON input in addition to Parquet
- Strips <think>...</think> blocks before evaluation (for Thinking models)
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
from transformers import AutoTokenizer, AutoModel
from scripts.mmau_pro.eval_utils import (
    remove_thinking_process,
    load_qwen_model,
    evaluate_openended_with_qwen,
    calculate_openended_metrics,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore")

# ================================
# AIF Functions (Identical to official)
# ================================

def count_words(text):
    return len(text.split())

def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def count_paragraphs(text):
    paragraphs = text.split("***")
    return len([p for p in paragraphs if p.strip()])

def count_bullet_points(text):
    bullets = re.findall(r'(?:^|\n)\s*\*\s+', text)
    return len(bullets)

def count_highlighted_sections(text):
    highlights = re.findall(r'\*([^*]+)\*', text)
    return len(highlights)

def count_placeholders(text):
    placeholders = re.findall(r'\[[^\]]+\]', text)
    return len(placeholders)

def count_capital_words(text):
    words = text.split()
    return len([w for w in words if w.isupper()])

def count_keyword_frequency(text, keyword):
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    return len(re.findall(pattern, text.lower()))

def has_title(text):
    return bool(re.search(r'<<[^>]+>>', text))

def has_postscript(text, marker):
    text_alpha = re.sub(r'[^a-zA-Z]', '', text).lower()
    marker_alpha = re.sub(r'[^a-zA-Z]', '', marker).lower()
    return marker_alpha in text_alpha

def starts_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.startswith(phrase_alpha)

def ends_with_phrase(text, phrase):
    text_alpha = re.sub(r'[^a-zA-Z ]', '', text).lower()
    phrase_alpha = re.sub(r'[^a-zA-Z ]', '', phrase).lower()
    return text_alpha.endswith(phrase_alpha)

def is_wrapped_in_quotes(text):
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')

def has_no_commas(text):
    return ',' not in text

def check_sections(text, num_sections, splitter):
    escaped_splitter = re.escape(splitter)
    sections = re.split(rf'\s*{escaped_splitter}\s*', text.strip())
    actual_sections = [s for s in sections if s.strip()]
    return len(actual_sections) == num_sections

def evaluate_aif_sample(response, sample_data):
    """Evaluate Audio Instruction Following sample.

    NOTE: Strips <think>...</think> blocks before evaluation so format/word-count
    checks apply only to the actual response (our addition for Thinking models).
    """
    response = remove_thinking_process(response)
    task_identifier = sample_data.get("task_identifier", "")
    kwargs = sample_data.get("kwargs", {}) or {}

    success = False

    if task_identifier == "Include Keywords":
        keywords = kwargs.get("keywords", "").split(", ")
        success = all(keyword.lower() in response.lower() for keyword in keywords)
    elif task_identifier == "Keyword Frequency":
        keyword = kwargs.get("keyword", "")
        target = kwargs.get("N", 0)
        success = count_keyword_frequency(response, keyword) == target
    elif task_identifier == "Forbidden Words":
        forbidden_words = kwargs.get("forbidden_words", "").split(", ")
        success = not any(word.lower() in response.lower() for word in forbidden_words)
    elif task_identifier == "Number Paragraphs":
        success = count_paragraphs(response) == kwargs.get("N", 0)
    elif task_identifier == "Number Words (at least)":
        success = count_words(response) >= kwargs.get("N", 0)
    elif task_identifier == "Number Words (at most)":
        success = count_words(response) <= kwargs.get("N", 0)
    elif task_identifier == "Number Words (range)":
        success = kwargs.get("N1", 0) <= count_words(response) <= kwargs.get("N2", 999)
    elif task_identifier == "Number Sentences (at least)":
        success = count_sentences(response) >= kwargs.get("N", 0)
    elif task_identifier == "Number Sentences (at most)":
        success = count_sentences(response) <= kwargs.get("N", 0)
    elif task_identifier == "Number Sentences (range)":
        success = kwargs.get("N1", 0) <= count_sentences(response) <= kwargs.get("N2", 999)
    elif task_identifier == "Postscript":
        success = has_postscript(response, kwargs.get("postscript_marker", ""))
    elif task_identifier == "Number Placeholder":
        success = count_placeholders(response) >= kwargs.get("N", 0)
    elif task_identifier == "Number Bullets":
        success = count_bullet_points(response) == kwargs.get("N", 0)
    elif task_identifier == "Title":
        success = has_title(response)
    elif task_identifier == "Minimum Number Highlighted Section":
        success = count_highlighted_sections(response) >= kwargs.get("N", 0)
    elif task_identifier == "Multiple Sections":
        success = check_sections(response, kwargs.get("N", 0), kwargs.get("section_splitter", ""))
    elif task_identifier == "Repeat Prompt":
        # Official: uses question field as prompt_transcription
        original_prompt = sample_data.get("prompt_transcription", "")
        success = response.strip().lower().startswith(original_prompt.strip().lower())
    elif task_identifier == "Two Responses":
        parts = response.split("******")
        success = len(parts) == 2 and parts[0].lower().strip() != parts[1].lower().strip()
    elif task_identifier == "All Uppercase":
        success = response.isupper()
    elif task_identifier == "All Lowercase":
        success = response.islower()
    elif task_identifier == "All-capital Words (at least)":
        success = count_capital_words(response) >= kwargs.get("N", 0)
    elif task_identifier == "All-capital Words (at most)":
        success = count_capital_words(response) <= kwargs.get("N", 0)
    elif task_identifier == "All-capital Words (range)":
        success = kwargs.get("N1", 0) <= count_capital_words(response) <= kwargs.get("N2", 999)
    elif task_identifier == "Start Checker":
        success = starts_with_phrase(response, kwargs.get("start_phrase", ""))
    elif task_identifier == "End Checker":
        success = ends_with_phrase(response, kwargs.get("end_phrase", ""))
    elif task_identifier == "Quotation":
        success = is_wrapped_in_quotes(response)
    elif task_identifier == "No Commas":
        success = has_no_commas(response)

    return success

# ================================
# Closed-ended Evaluation (NVEmbed Only — aligned with official)
# ================================

def _patch_nvembed_for_transformers_compat(model):
    """Monkey-patch NV-Embed-v2's BidirectionalMistralModel to work with transformers>=4.50.

    Newer transformers changed MistralDecoderLayer.forward to require position_embeddings=(cos, sin)
    instead of computing them internally. The NV-Embed-v2 custom code doesn't pass this,
    causing: TypeError: cannot unpack non-iterable NoneType object.

    This patch wraps the embedding_model.forward to compute position_embeddings from the
    model's rotary_emb and pass them to each decoder layer.
    """
    emb_model = model.embedding_model
    if not hasattr(emb_model, 'rotary_emb'):
        return  # Nothing to patch

    import types
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    try:
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
    except ImportError:
        _prepare_4d_attention_mask_for_sdpa = _prepare_4d_attention_mask

    from transformers.modeling_outputs import BaseModelOutputWithPast

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # NVEmbed encoding never uses KV cache
        use_cache = False

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
        else:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # --- KEY FIX: compute position_embeddings for newer transformers ---
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

            # Newer transformers returns a single tensor, older returns a tuple
            if isinstance(layer_outputs, torch.Tensor):
                hidden_states = layer_outputs
            else:
                hidden_states = layer_outputs[0]
                if output_attentions and len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    emb_model.forward = types.MethodType(patched_forward, emb_model)
    print("Applied NVEmbed compatibility patch for transformers>=4.50")


def load_nvembed_model():
    """Load NV-Embed-v2 model (aligned with official: local_files_only=False)."""
    print("Loading NV-Embed-v2 model...")
    model = AutoModel.from_pretrained(
        'nvidia/NV-Embed-v2', trust_remote_code=True, local_files_only=False
    )
    _patch_nvembed_for_transformers_compat(model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("NVEmbed model loaded successfully!")
    return model

def evaluate_closedended_with_nvembed(model, questions, choices_list, ground_truth_answers, predicted_answers, task_types):
    """NVEmbed-only evaluation — aligned with official (no regex/keyword fallback).

    NOTE: Strips <think> blocks before embedding (our addition for Thinking models).
    """
    predictions = []
    confidence_scores = []

    print("Performing NVEmbed evaluation...")
    for question, choices, gt_answer, model_prediction, task_type in tqdm(
        zip(questions, choices_list, ground_truth_answers, predicted_answers, task_types)
    ):
        # Strip thinking blocks so NVEmbed matches the actual answer
        clean_prediction = remove_thinking_process(model_prediction)

        prediction_embedding = model.encode([clean_prediction], instruction="", max_length=4096)
        prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)

        choice_embeddings = model.encode(choices, instruction="", max_length=4096)
        choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

        scores = (prediction_embedding @ choice_embeddings.T) * 100
        scores = scores.squeeze()

        best_choice_idx = torch.argmax(scores).item()
        predictions.append(choices[best_choice_idx])
        confidence_scores.append(torch.max(scores).item())

    return predictions, confidence_scores

# ================================
# Utility Functions
# ================================

def calculate_metrics(ground_truth, predictions):
    if len(ground_truth) == 0 or len(predictions) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def calculate_weighted_performance(category_results):
    """Calculate overall weighted performance (aligned with official)."""
    total_weighted_score = 0.0
    total_samples = 0
    category_scores = {}

    for category, result in category_results.items():
        count = result['count']
        if result['type'] == 'openended':
            score = result['metrics'].get('avg_overall', 3.0) / 5.0
        elif result['type'] == 'aif':
            score = result['success_rate']
        elif result['type'] == 'closed':
            score = result['metrics'].get('accuracy', 0.0)
        else:
            continue

        category_scores[category] = score
        total_weighted_score += score * count
        total_samples += count

    overall = total_weighted_score / total_samples if total_samples > 0 else 0.0
    return overall, category_scores

# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(description='MMAU-Pro evaluation (aligned with official)')
    parser.add_argument('parquet_file', nargs='?', default='test.parquet',
                        help='Path to parquet/jsonl/json file with test data')
    parser.add_argument('--model_output_column', default='model_output',
                        help='Column name containing model outputs')
    args = parser.parse_args()

    parquet_file_path = args.parquet_file
    model_output_column = args.model_output_column

    if not os.path.exists(parquet_file_path):
        print(f"Error: File '{parquet_file_path}' not found.")
        return

    # Load data (supports parquet, jsonl, json)
    print(f"Loading data from: {parquet_file_path}")
    file_ext = os.path.splitext(parquet_file_path)[1].lower()
    if file_ext == '.parquet':
        df = pd.read_parquet(parquet_file_path)
    elif file_ext in ['.jsonl', '.json']:
        df = pd.read_json(parquet_file_path, lines=(file_ext == '.jsonl'))
        # Map our inference output column names to official names
        if 'prediction' in df.columns and model_output_column not in df.columns:
            df[model_output_column] = df['prediction']
    else:
        print(f"Error: Unsupported file format '{file_ext}'")
        return

    # Map column names
    if 'ground_truth' in df.columns and 'answer' not in df.columns:
        df['answer'] = df['ground_truth']
    if model_output_column not in df.columns:
        print(f"Warning: Column '{model_output_column}' not found. Using 'answer' as placeholder.")
        df[model_output_column] = df['answer']

    print(f"Loaded {len(df)} samples")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    input_filename = os.path.splitext(os.path.basename(parquet_file_path))[0]
    category_results = {}

    # ================================
    # 1. Open-ended Evaluation (in-process, aligned with official)
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING OPEN-ENDED QUESTIONS")
    print("=" * 60)

    open_df = df[df['category'] == 'open'].copy()
    if len(open_df) > 0:
        print(f"Found {len(open_df)} open-ended questions")

        qwen_model, qwen_tokenizer = load_qwen_model()

        questions = open_df['question'].tolist()
        reference_answers = open_df['answer'].tolist()
        # Strip thinking blocks so judge evaluates only the final answer
        model_responses = [
            remove_thinking_process(r) for r in open_df[model_output_column].fillna("").tolist()
        ]
        task_types = ['open'] * len(open_df)

        openended_scores, openended_detailed = evaluate_openended_with_qwen(
            qwen_model, qwen_tokenizer, questions, reference_answers, model_responses, task_types
        )

        openended_metrics = calculate_openended_metrics(openended_scores)
        category_results['open'] = {
            'type': 'openended',
            'count': len(open_df),
            'metrics': openended_metrics,
            'scores': openended_scores,
        }

        print(f"Open-ended evaluation completed: {len(openended_scores)} samples")
        print(f"Average Overall Score: {openended_metrics.get('avg_overall', 0.0):.3f}/5.0")

        del qwen_model, qwen_tokenizer
        torch.cuda.empty_cache()
    else:
        print("No open-ended questions found")

    # ================================
    # 2. AIF Evaluation (rule-based, aligned with official)
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING INSTRUCTION FOLLOWING QUESTIONS")
    print("=" * 60)

    aif_df = df[df['category'] == 'instruction following'].copy()
    if len(aif_df) > 0:
        print(f"Found {len(aif_df)} instruction following questions")

        aif_results = []
        for _, row in tqdm(aif_df.iterrows(), total=len(aif_df)):
            model_response = str(row.get(model_output_column, ""))
            # Aligned with official: use question field as prompt_transcription
            sample_data = {
                'task_identifier': row.get('task_identifier'),
                'kwargs': row.get('kwargs'),
                'prompt_transcription': row.get('question', ""),
            }
            success = evaluate_aif_sample(model_response, sample_data)
            aif_results.append(success)

        success_rate = np.mean([float(r) for r in aif_results])
        category_results['instruction following'] = {
            'type': 'aif',
            'count': len(aif_df),
            'success_rate': success_rate,
            'results': aif_results,
        }

        print(f"Instruction following evaluation completed: {len(aif_results)} samples")
        print(f"Success Rate: {success_rate:.3f}")
    else:
        print("No instruction following questions found")

    # ================================
    # 3. Closed-ended Evaluation (NVEmbed only, aligned with official)
    # ================================
    print("\n" + "=" * 60)
    print("EVALUATING CLOSED-ENDED QUESTIONS")
    print("=" * 60)

    closed_categories = [cat for cat in df['category'].unique()
                         if cat not in ['open', 'instruction following']]
    closed_df = df[df['category'].isin(closed_categories)].copy()

    if len(closed_df) > 0:
        print(f"Found {len(closed_df)} closed-ended questions across: {closed_categories}")

        # Filter: valid choices (aligned with official)
        closed_df = closed_df[closed_df['choices'].notna()].copy()
        closed_df = closed_df[closed_df['choices'].apply(
            lambda x: len(x) > 1 if hasattr(x, '__len__') else False
        )].copy()

        print(f"After filtering: {len(closed_df)} samples with valid choices")

        if len(closed_df) > 0:
            nvembed_model = load_nvembed_model()

            questions = closed_df['question'].tolist()
            ground_truth_answers = closed_df['answer'].tolist()
            choices_list = [list(c) if hasattr(c, '__iter__') else [str(c)]
                           for c in closed_df['choices'].tolist()]
            model_predictions = closed_df[model_output_column].fillna("").tolist()
            task_types = closed_df['category'].tolist()

            predictions, confidence_scores = evaluate_closedended_with_nvembed(
                nvembed_model, questions, choices_list, ground_truth_answers,
                model_predictions, task_types
            )

            overall_closed_metrics = calculate_metrics(ground_truth_answers, predictions)

            for category in closed_categories:
                cat_mask = closed_df['category'] == category
                if cat_mask.sum() > 0:
                    cat_gt = [ground_truth_answers[i] for i, mask in enumerate(cat_mask) if mask]
                    cat_pred = [predictions[i] for i, mask in enumerate(cat_mask) if mask]
                    category_results[category] = {
                        'type': 'closed',
                        'count': cat_mask.sum(),
                        'metrics': calculate_metrics(cat_gt, cat_pred),
                    }

            print(f"Closed-ended evaluation completed: {len(predictions)} samples")
            print(f"Overall Accuracy: {overall_closed_metrics['accuracy']:.4f}")

            del nvembed_model
            torch.cuda.empty_cache()
    else:
        print("No closed-ended questions found")

    # ================================
    # Calculate Weighted Performance & Save Results
    # ================================
    overall_weighted_performance, category_scores = calculate_weighted_performance(category_results)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    total_samples = len(df)
    evaluated_samples = sum(result['count'] for result in category_results.values())

    print(f"Total samples: {total_samples}")
    print(f"Successfully evaluated: {evaluated_samples}")
    print(f"Overall Weighted Performance: {overall_weighted_performance:.4f}")

    print("\nBREAKDOWN BY CATEGORY:")
    print("-" * 60)
    for category, result in category_results.items():
        print(f"\n{category.upper()}:")
        print(f"  Type: {result['type']}")
        print(f"  Count: {result['count']}")
        print(f"  Performance Score: {category_scores.get(category, 0.0):.4f}")
        if result['type'] == 'openended':
            metrics = result['metrics']
            print(f"  Avg Overall Score: {metrics.get('avg_overall', 0.0):.3f}/5.0")
            print(f"  Avg Correctness: {metrics.get('avg_correctness', 0.0):.3f}/5.0")
        elif result['type'] == 'aif':
            print(f"  Success Rate: {result['success_rate']:.3f}")
        elif result['type'] == 'closed':
            metrics = result['metrics']
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"  F1 Score: {metrics.get('f1_score', 0.0):.4f}")

    # Save results (aligned with official output format)
    results_summary = {
        'evaluation_summary': {
            'total_samples': total_samples,
            'evaluated_samples': evaluated_samples,
            'parquet_file': parquet_file_path,
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

    output_filename = f'{input_filename}_comprehensive_results.json'
    # Save alongside input file (not in CWD)
    output_dir = os.path.dirname(os.path.abspath(parquet_file_path))
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=float)

    print(f"\nResults saved to '{output_path}'")
    print("\nEvaluation completed successfully!")
    print(f"\nSUMMARY:")
    print(f"   Overall Weighted Performance: {overall_weighted_performance:.4f}")
    print(f"   Total Samples Evaluated: {evaluated_samples}/{total_samples}")

if __name__ == "__main__":
    main()
