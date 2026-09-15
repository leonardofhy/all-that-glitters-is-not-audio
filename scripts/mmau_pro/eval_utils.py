#!/usr/bin/env python3
"""
Shared utilities for MMAU-Pro evaluation scripts.

Contains functions used by both ``evaluate.py`` (main evaluation) and
``evaluate_openended.py`` (standalone subprocess for open-ended scoring).
Extracted to eliminate ~150 lines of code duplication.

NOTE: This module intentionally does NOT import NVEmbed (AutoModel) so it
can be safely imported in the subprocess without triggering CUDA context
conflicts.
"""

import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# DynamicCache Monkey Patch
# ================================
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, input_length, layer_idx=None):
            if layer_idx is None:
                layer_idx = 0
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length
except ImportError:
    pass


# ================================
# Text Cleaning
# ================================

def remove_thinking_process(text):
    """Remove <think>...</think> blocks from model output."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Handle leaked/truncated <think> without closing tag (endpoint cutoff)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Handle </think> without opening <think> (vLLM strips opening tag for some models)
    cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# ================================
# Judge LLM (Qwen 2.5)
# ================================

def load_qwen_model():
    """Load the Qwen 2.5-7B-Instruct Judge model."""
    torch.cuda.empty_cache()
    print("Loading Qwen 2.5-7B-Instruct (Judge LLM)...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Qwen 2.5 model loaded successfully!")
    return model, tokenizer


def create_evaluation_prompt(question, reference_answer, model_response, task_type):
    """Build the LLM-judge prompt for open-ended evaluation."""
    task_context = {
        "sound": "audio content analysis and sound identification",
        "speech": "speech recognition and conversation understanding",
        "music": "music analysis and musical element identification",
        "open": "general open-ended question answering",
    }
    context = task_context.get(task_type, "general question answering")

    return f"""You are an expert evaluator for {context} tasks.

Question: {question}
Reference Answer: {reference_answer}
Model Response: {model_response}

Evaluate on these criteria (1-5 scale):
1. **Correctness**: Factual accuracy vs reference
2. **Relevance**: Addresses the question
3. **Completeness**: Covers important aspects
4. **Clarity**: Clear and well-structured

Format:
CORRECTNESS: [score] - [justification]
RELEVANCE: [score] - [justification]
COMPLETENESS: [score] - [justification]
CLARITY: [score] - [justification]
OVERALL: [average] - [assessment]"""


def extract_scores_from_evaluation(evaluation_text):
    """Parse criterion scores from the judge LLM output."""
    patterns = {
        'correctness': r'CORRECTNESS:\s*(\d+)',
        'relevance': r'RELEVANCE:\s*(\d+)',
        'completeness': r'COMPLETENESS:\s*(\d+)',
        'clarity': r'CLARITY:\s*(\d+)',
        'overall': r'OVERALL:\s*(\d+(?:\.\d+)?)',
    }
    scores = {}
    for criterion, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        scores[criterion] = float(match.group(1)) if match else 3.0
    if scores.get('overall', 3.0) == 3.0:
        scores['overall'] = np.mean([
            scores.get(k, 3.0) for k in ['correctness', 'relevance', 'completeness', 'clarity']
        ])
    return scores


def evaluate_openended_with_qwen(
    model, tokenizer, questions, reference_answers, model_responses, task_types,
    batch_size=128,
):
    """Evaluate open-ended responses using Qwen 2.5 as a judge.

    Aligned with official MMAU-Pro evaluation:
    - do_sample=True, temperature=0.1, max_new_tokens=512
    - Batched generation for throughput (batch_size controls GPU utilization)
    - Returns (all_scores, detailed_evaluations) tuple
    """
    all_scores = []
    detailed_evaluations = []

    print(f"Performing Qwen 2.5 LLM judge evaluation (batch_size={batch_size})...")

    n = len(questions)
    for batch_start in tqdm(range(0, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        batch_questions = questions[batch_start:batch_end]
        batch_refs = reference_answers[batch_start:batch_end]
        batch_responses = model_responses[batch_start:batch_end]
        batch_tasks = task_types[batch_start:batch_end]

        # Prepare all prompts in the batch
        texts = []
        for q, ref, resp, tt in zip(batch_questions, batch_refs, batch_responses, batch_tasks):
            eval_prompt = create_evaluation_prompt(q, ref, resp, tt)
            messages = [
                {"role": "system", "content": "You are a helpful and objective evaluator."},
                {"role": "user", "content": eval_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        # Tokenize with left-padding for batched generation
        tokenizer.padding_side = "left"
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
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
            # Extract only new tokens for each sample
            input_lengths = model_inputs.input_ids.shape[1]
            new_ids = generated_ids[:, input_lengths:]
            evaluation_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: Error evaluating batch {batch_start}-{batch_end}: {e}")
            evaluation_texts = [""] * (batch_end - batch_start)

        # Process each sample in the batch
        for j, (q, ref, resp, tt, eval_text) in enumerate(
            zip(batch_questions, batch_refs, batch_responses, batch_tasks, evaluation_texts)
        ):
            scores = extract_scores_from_evaluation(eval_text) if eval_text else {
                'correctness': 3.0, 'relevance': 3.0,
                'completeness': 3.0, 'clarity': 3.0, 'overall': 3.0,
            }
            all_scores.append(scores)
            detailed_evaluations.append({
                'question': q,
                'reference_answer': ref,
                'model_response': resp,
                'evaluation': eval_text,
                'scores': scores,
                'task_type': tt,
            })

    return all_scores, detailed_evaluations


def calculate_openended_metrics(all_scores):
    """Aggregate per-sample scores into summary metrics.

    Aligned with official MMAU-Pro: includes std and poor_response_rate.
    """
    if not all_scores:
        return {}
    metrics = {}
    for criterion in ['correctness', 'relevance', 'completeness', 'clarity', 'overall']:
        scores = [s.get(criterion, 3.0) for s in all_scores]
        metrics[f'avg_{criterion}'] = float(np.mean(scores))
        metrics[f'std_{criterion}'] = float(np.std(scores))
    metrics['good_response_rate'] = (
        sum(1 for s in all_scores if s.get('overall', 3.0) >= 4.0) / len(all_scores)
    )
    metrics['poor_response_rate'] = (
        sum(1 for s in all_scores if s.get('overall', 3.0) <= 2.0) / len(all_scores)
    )
    return metrics
