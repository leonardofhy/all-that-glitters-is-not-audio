#!/usr/bin/env python3
"""Unified LLM MCQ Judge — Claude Haiku answer extraction + correctness judgment.

Evaluates MCQ predictions across all 3 benchmarks (MMAU, MMAU-Pro, MMAR) using
a hybrid approach: regex extraction for obvious answers, Claude Haiku for ambiguous
cases. Outputs per-sample JSONL (for downstream analysis) and aggregate JSON.

Usage:
    # Single file
    python scripts/analysis/llm_mcq_judge.py results/mmau/qwen2_audio_7b_instruct/full.jsonl

    # Batch mode (full+none for all models on one benchmark)
    python scripts/analysis/llm_mcq_judge.py --batch --benchmark mmau --condition full,none

    # Batch mode (everything)
    python scripts/analysis/llm_mcq_judge.py --batch --benchmark all --model all

    # Dry run
    python scripts/analysis/llm_mcq_judge.py --batch --benchmark all --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.constants import (
    BENCHMARK_CONFIG,
    MODEL_REGISTRY,
    RESULTS_DIR,
    get_expected_conditions,
    get_model_dir,
    iter_model_benchmark_pairs,
)
from scripts.eval_utils_claude import (
    CLAUDE_MODEL,
    CONCURRENCY,
    _quick_regex_extract,
    remove_thinking_process,
)

# Import string_match for official comparison (MMAU/MMAR use identical logic)
from scripts.mmau.evaluate import string_match

# ── Auto-load .env ────────────────────────────────────────────────────────────
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ── Constants ─────────────────────────────────────────────────────────────────

# MMAU-Pro categories that are NOT closed-ended MCQ
MMAU_PRO_NON_MCQ = {"open", "instruction following"}

HYBRID_PROMPT = """\
Given a model's response to a multiple choice question, determine:
1. Which option letter the model selected
2. Whether the response is correct

Rules:
- If the model selects exactly one option via letter ({valid_letters}), \
audio index ("Audio 1" → A, "The second clip" → B), or content match, extract that letter.
- If the model cannot process audio, hedges, or selects multiple options, \
reply NONE for the letter and incorrect for judgment.
- Semantic equivalence counts (e.g., "furious" matches "angry").

Question: {question}

Options:
{options_text}

Ground truth: {gt_letter}) {gt_text}

Model's response:
{prediction}

Reply ONLY in this format:
Selected: <letter or NONE>
Judgment: <correct or incorrect>"""


# ── BenchmarkAdapter ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkAdapter:
    """Normalizes the three benchmark formats into a common schema."""
    benchmark: str
    prediction_key: str
    ground_truth_key: str
    choices_key: str = "choices"
    has_letter_prefix: bool = False  # MMAU choices have "(A) Man" format
    category_key: str = "category"

    @classmethod
    def detect(cls, sample: dict) -> BenchmarkAdapter:
        """Auto-detect benchmark format from field names."""
        if "model_prediction" in sample:
            return cls("mmar", "model_prediction", "answer",
                       category_key="modality")
        elif "task" in sample and "difficulty" in sample:
            return cls("mmau", "prediction", "ground_truth",
                       has_letter_prefix=True, category_key="task")
        else:
            return cls("mmau_pro", "prediction", "ground_truth",
                       category_key="category")

    @classmethod
    def from_name(cls, benchmark: str) -> BenchmarkAdapter:
        """Create adapter from benchmark name."""
        if benchmark == "mmar":
            return cls("mmar", "model_prediction", "answer",
                       category_key="modality")
        elif benchmark == "mmau":
            return cls("mmau", "prediction", "ground_truth",
                       has_letter_prefix=True, category_key="task")
        else:
            return cls("mmau_pro", "prediction", "ground_truth",
                       category_key="category")

    def is_mcq(self, sample: dict) -> bool:
        """Check if sample is an MCQ question."""
        choices = sample.get(self.choices_key)
        if not choices or not isinstance(choices, list) or len(choices) < 2:
            return False
        if self.benchmark == "mmau_pro":
            cat = sample.get(self.category_key, "")
            if cat in MMAU_PRO_NON_MCQ:
                return False
        return True


# ── Utility functions ─────────────────────────────────────────────────────────

def ground_truth_to_letter(gt: str, choices: list[str], has_letter_prefix: bool) -> str | None:
    """Map ground truth text to its letter position (A/B/C/D...).

    Handles: exact match, prefix differences, and token-overlap fallback.
    """
    if has_letter_prefix:
        # MMAU format: "(A) Man" → extract "A" from prefix
        m = re.match(r'\(?([A-Z])\)?', gt.strip())
        if m:
            return m.group(1)

    # Strip letter prefixes for comparison
    def strip_prefix(text: str) -> str:
        return re.sub(r'^\(?[A-Za-z]\)?\s*', '', text.strip()).strip()

    gt_clean = strip_prefix(gt).lower()

    # Pass 1: exact match (after prefix stripping)
    for i, choice in enumerate(choices):
        if strip_prefix(choice).lower() == gt_clean:
            return chr(65 + i)

    # Pass 2: substring containment (handles "Because X" vs "X")
    for i, choice in enumerate(choices):
        c = strip_prefix(choice).lower()
        if gt_clean in c or c in gt_clean:
            return chr(65 + i)

    # Pass 3: token overlap (fuzzy match)
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r'\b\w+\b', text.lower()))

    gt_tokens = tokenize(gt_clean)
    if gt_tokens:
        best_idx, best_overlap = -1, 0
        for i, choice in enumerate(choices):
            c_tokens = tokenize(strip_prefix(choice))
            overlap = len(gt_tokens & c_tokens) / max(len(gt_tokens | c_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_overlap >= 0.6:  # At least 60% Jaccard overlap
            return chr(65 + best_idx)

    return None


def _content_match_extract(prediction: str, choices: list[str]) -> str | None:
    """Try to match prediction text directly to a choice (for non-letter predictions).

    Handles numeric choices like ['0', '1', '3', '2'] where prediction is '2'.
    """
    pred_clean = prediction.strip().lower()
    if not pred_clean:
        return None

    # Exact content match
    for i, choice in enumerate(choices):
        if choice.strip().lower() == pred_clean:
            return chr(65 + i)

    # Prefix match (prediction starts with or is contained in choice)
    for i, choice in enumerate(choices):
        c = choice.strip().lower()
        if pred_clean.startswith(c) or c.startswith(pred_clean):
            if len(pred_clean) >= 1 and len(c) >= 1:
                return chr(65 + i)

    return None


def _build_hybrid_prompt(
    question: str, choices: list[str], prediction: str,
    gt_letter: str, gt_text: str,
) -> str:
    """Build the hybrid extraction+judgment prompt."""
    letters = [chr(65 + i) for i in range(len(choices))]
    options_text = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    valid_letters = ", ".join(letters)
    return HYBRID_PROMPT.format(
        question=question,
        options_text=options_text,
        prediction=prediction[:2000],
        valid_letters=valid_letters,
        gt_letter=gt_letter,
        gt_text=gt_text,
    )


def _parse_hybrid_response(
    response_text: str, choices: list[str], gt_letter: str,
) -> tuple[str | None, bool]:
    """Parse Claude's hybrid response into (selected_letter, is_correct).

    Returns:
        (letter_or_None, correctness_bool)
    """
    letters = set(chr(65 + i) for i in range(len(choices)))
    text = response_text.strip()

    selected = None
    correct = False

    # Parse "Selected: X"
    m = re.search(r'Selected:\s*([A-Z]|NONE)', text, re.IGNORECASE)
    if m:
        val = m.group(1).upper()
        if val in letters:
            selected = val
        # NONE → selected stays None

    # Parse "Judgment: correct/incorrect"
    m = re.search(r'Judg[e]?ment:\s*(correct|incorrect)', text, re.IGNORECASE)
    if m:
        correct = m.group(1).lower() == "correct"

    # Fallback: if no structured response, try to extract a letter
    if selected is None and not re.search(r'Selected:', text, re.IGNORECASE):
        for char in text.upper():
            if char in letters:
                selected = char
                break

    return selected, correct


def _compute_official_correct(
    prediction: str, ground_truth: str, choices: list[str],
    benchmark: str,
) -> bool | None:
    """Run official string_match for MMAU/MMAR. Returns None for MMAU-Pro."""
    if benchmark == "mmau_pro":
        return None
    return string_match(ground_truth, prediction, choices)


# ── Core async processing ────────────────────────────────────────────────────

async def process_samples(
    samples: list[dict],
    adapter: BenchmarkAdapter,
    use_regex_prefilter: bool = True,
) -> tuple[list[dict], dict]:
    """Process MCQ samples: regex prefilter + Claude API for ambiguous cases.

    Returns:
        (per_sample_results, stats_dict)
    """
    import anthropic

    async_client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    results = [None] * len(samples)
    stats = {
        "total": len(samples),
        "skipped_non_mcq": 0,
        "skipped_empty": 0,
        "regex_resolved": 0,
        "api_calls": 0,
        "api_errors": 0,
        "none_answers": 0,
    }

    async def process_one(idx: int, sample: dict) -> None:
        # Skip non-MCQ
        if not adapter.is_mcq(sample):
            stats["skipped_non_mcq"] += 1
            return

        prediction_raw = sample.get(adapter.prediction_key, "")
        prediction = remove_thinking_process(prediction_raw)
        # Fall back to raw text for LLM extraction when think-stripping
        # empties the output (e.g., unclosed <think> blocks from max_tokens)
        if not prediction.strip() and prediction_raw.strip():
            prediction = prediction_raw
        ground_truth = sample.get(adapter.ground_truth_key, "")
        choices = sample.get(adapter.choices_key, [])
        sample_id = sample.get("id", str(idx))

        # Map GT to letter
        gt_letter = ground_truth_to_letter(
            ground_truth, choices, adapter.has_letter_prefix
        )

        # Official eval (uses think-stripped prediction for string-match)
        official_correct = _compute_official_correct(
            remove_thinking_process(prediction_raw), ground_truth, choices, adapter.benchmark
        )

        if not prediction.strip():
            stats["skipped_empty"] += 1
            results[idx] = {
                "id": sample_id,
                "gt_letter": gt_letter,
                "predicted_letter": None,
                "llm_correct": False,
                "extraction_method": "empty",
                "official_correct": official_correct,
                "agreement": (False == official_correct) if official_correct is not None else None,
                "prediction_snippet": "",
            }
            return

        # Regex prefilter (letter patterns like "A", "B.", "(C)")
        if use_regex_prefilter:
            regex_letter = _quick_regex_extract(prediction, choices)
            if regex_letter is None:
                # Content-match fallback (handles numeric choices, exact text)
                regex_letter = _content_match_extract(prediction, choices)
            if regex_letter is not None:
                stats["regex_resolved"] += 1
                llm_correct = (regex_letter == gt_letter) if gt_letter else False
                results[idx] = {
                    "id": sample_id,
                    "gt_letter": gt_letter,
                    "predicted_letter": regex_letter,
                    "llm_correct": llm_correct,
                    "extraction_method": "regex",
                    "official_correct": official_correct,
                    "agreement": (llm_correct == official_correct) if official_correct is not None else None,
                    "prediction_snippet": prediction[:200],
                }
                return

        # Claude API call with hybrid prompt
        gt_text = ground_truth
        if adapter.has_letter_prefix:
            gt_text = re.sub(r'^\([A-Z]\)\s*', '', ground_truth)

        prompt = _build_hybrid_prompt(
            question=sample.get("question", ""),
            choices=choices,
            prediction=prediction,
            gt_letter=gt_letter or "?",
            gt_text=gt_text,
        )

        async with semaphore:
            try:
                response = await async_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=20,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                stats["api_calls"] += 1
                selected, claude_judgment = _parse_hybrid_response(
                    response.content[0].text, choices, gt_letter or ""
                )
                # Deterministic correctness: if we extracted a letter, compare directly
                if selected is not None and gt_letter is not None:
                    llm_correct = (selected == gt_letter)
                else:
                    # Fall back to Claude's judgment when letter extraction failed
                    llm_correct = claude_judgment
                if selected is None:
                    stats["none_answers"] += 1

                results[idx] = {
                    "id": sample_id,
                    "gt_letter": gt_letter,
                    "predicted_letter": selected,
                    "llm_correct": llm_correct,
                    "extraction_method": "claude",
                    "official_correct": official_correct,
                    "agreement": (llm_correct == official_correct) if official_correct is not None else None,
                    "prediction_snippet": prediction[:200],
                }
            except Exception as e:
                stats["api_errors"] += 1
                if stats["api_errors"] <= 5:
                    print(f"  Warning: API error on sample {idx}: {e}")
                elif stats["api_errors"] == 6:
                    print("  (suppressing further API error messages)")
                # Write error entry so resume can skip it
                results[idx] = {
                    "id": sample_id,
                    "gt_letter": gt_letter,
                    "predicted_letter": None,
                    "llm_correct": False,
                    "extraction_method": "error",
                    "official_correct": official_correct,
                    "agreement": None,
                    "prediction_snippet": prediction[:200],
                }

    t0 = time.time()
    tasks = [process_one(i, s) for i, s in enumerate(samples)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    stats["elapsed_seconds"] = round(elapsed, 1)
    resolved = stats["regex_resolved"] + stats["api_calls"] - stats["api_errors"]
    print(f"  Done: {resolved}/{stats['total']} resolved "
          f"({stats['regex_resolved']} regex, {stats['api_calls']} API, "
          f"{stats['api_errors']} err) in {elapsed:.1f}s")

    return [r for r in results if r is not None], stats


# ── File I/O ──────────────────────────────────────────────────────────────────

def load_inference_file(path: Path) -> list[dict]:
    """Load JSONL or JSON array inference file."""
    content = path.read_text().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_existing_judge(path: Path) -> set[str]:
    """Load already-judged sample IDs from an existing judge JSONL."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def compute_aggregate(
    per_sample: list[dict],
    samples: list[dict],
    adapter: BenchmarkAdapter,
    stats: dict,
) -> dict:
    """Compute aggregate results from per-sample data."""
    total = 0
    correct = 0
    no_pred = 0
    category_metrics = defaultdict(lambda: [0, 0])  # [correct, total]

    # Official comparison accumulators
    official_total = 0
    official_correct_count = 0
    agree_count = 0
    both_correct = 0
    llm_only = 0
    official_only = 0
    both_wrong = 0

    for r in per_sample:
        total += 1
        cat = "unknown"
        # Find the original sample to get category
        for s in samples:
            if s.get("id") == r["id"]:
                cat = s.get(adapter.category_key, "unknown")
                break

        if r["predicted_letter"] is None:
            no_pred += 1
            category_metrics[cat][1] += 1
            continue

        if r["llm_correct"]:
            correct += 1
            category_metrics[cat][0] += 1
        category_metrics[cat][1] += 1

        # Official comparison (MMAU/MMAR only)
        if r["official_correct"] is not None:
            official_total += 1
            if r["official_correct"]:
                official_correct_count += 1
            if r["agreement"] is not None:
                if r["agreement"]:
                    agree_count += 1
            if r["llm_correct"] and r["official_correct"]:
                both_correct += 1
            elif r["llm_correct"] and not r["official_correct"]:
                llm_only += 1
            elif not r["llm_correct"] and r["official_correct"]:
                official_only += 1
            else:
                both_wrong += 1

    output = {
        "overall_accuracy": correct / total if total > 0 else 0,
        "overall_correct": correct,
        "overall_total": total,
        "no_prediction_count": no_pred,
        "evaluation_method": "llm-mcq-judge-hybrid",
        "extraction_stats": stats,
        "category_accuracy": {
            cat: {
                "accuracy": vals[0] / vals[1] if vals[1] > 0 else 0,
                "correct": vals[0],
                "total": vals[1],
            }
            for cat, vals in sorted(category_metrics.items())
        },
    }

    if official_total > 0:
        output["official_comparison"] = {
            "official_accuracy": official_correct_count / official_total if official_total > 0 else 0,
            "llm_accuracy": correct / total if total > 0 else 0,
            "delta_mcq": (correct / total - official_correct_count / official_total) if total > 0 and official_total > 0 else 0,
            "agreement_rate": agree_count / official_total if official_total > 0 else 0,
            "both_correct": both_correct,
            "llm_only_correct": llm_only,
            "official_only_correct": official_only,
            "both_wrong": both_wrong,
        }

    return output


# ── Main process_file ─────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    benchmark: str | None = None,
    force: bool = False,
) -> dict | None:
    """Process one inference file. Returns aggregate stats or None if skipped."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"  SKIP (not found): {input_path}")
        return None

    # Output paths
    stem = input_path.stem
    judge_jsonl = input_path.parent / f"{stem}_llm_judge.jsonl"
    judge_results = input_path.parent / f"{stem}_llm_judge_results.json"

    # Load data
    samples = load_inference_file(input_path)
    if not samples:
        print(f"  SKIP (empty): {input_path}")
        return None

    # Detect or use provided benchmark
    if benchmark:
        adapter = BenchmarkAdapter.from_name(benchmark)
    else:
        adapter = BenchmarkAdapter.detect(samples[0])

    # Filter to MCQ only
    mcq_samples = [s for s in samples if adapter.is_mcq(s)]
    if not mcq_samples:
        print(f"  SKIP (no MCQ): {input_path}")
        return None

    # Resume: check existing output (force mode truncates existing file)
    if force and judge_jsonl.exists():
        judge_jsonl.unlink()
    existing_ids = set() if force else load_existing_judge(judge_jsonl)
    remaining = [s for s in mcq_samples if s.get("id", "") not in existing_ids]

    if not remaining and not force:
        print(f"  SKIP (complete, {len(existing_ids)} samples): {input_path.name}")
        # Reload existing results and recompute aggregate
        all_results = []
        for line in judge_jsonl.read_text().splitlines():
            if line.strip():
                all_results.append(json.loads(line))
        agg = compute_aggregate(all_results, mcq_samples, adapter, {"resumed": True})
        # Save aggregate (may have been missing or outdated)
        judge_results.write_text(json.dumps(agg, indent=2))
        return agg

    print(f"  Processing {input_path.name}: {len(remaining)} new + {len(existing_ids)} existing "
          f"({len(mcq_samples)} MCQ total, benchmark={adapter.benchmark})")

    # Process remaining samples
    per_sample, stats = asyncio.run(process_samples(remaining, adapter))

    # Append new results to JSONL
    with open(judge_jsonl, "a") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Reload all results for aggregate
    all_results = []
    for line in judge_jsonl.read_text().splitlines():
        if line.strip():
            all_results.append(json.loads(line))

    # Compute and save aggregate
    agg = compute_aggregate(all_results, mcq_samples, adapter, stats)
    judge_results.write_text(json.dumps(agg, indent=2))

    # Print summary
    print(f"  LLM accuracy: {agg['overall_accuracy']:.2%} ({agg['overall_correct']}/{agg['overall_total']})")
    if "official_comparison" in agg:
        oc = agg["official_comparison"]
        print(f"  Official accuracy: {oc['official_accuracy']:.2%}, "
              f"ΔMCQ: {oc['delta_mcq']:+.2%}, "
              f"Agreement: {oc['agreement_rate']:.2%}")

    return agg


# ── Batch runner ──────────────────────────────────────────────────────────────

def iter_files(
    benchmark_filter: str = "all",
    model_filter: str = "all",
    condition_filter: str = "all",
) -> list[tuple[str, str, str, Path]]:
    """Enumerate (model_short, benchmark, condition, file_path) tuples."""
    files = []
    for model_short, bench in iter_model_benchmark_pairs(benchmark_filter, model_filter):
        model_dir = get_model_dir(bench, model_short)
        if not model_dir.exists():
            continue

        # Determine which conditions to process
        if condition_filter == "all":
            conditions = get_expected_conditions(bench)
        else:
            conditions = [c.strip() for c in condition_filter.split(",")]

        cfg = BENCHMARK_CONFIG[bench]
        ext = cfg["ext"]

        for cond in conditions:
            fname = f"{cond}{ext}"
            fpath = model_dir / fname
            if fpath.exists():
                files.append((model_short, bench, cond, fpath))

    return files


def run_batch(
    benchmark_filter: str = "all",
    model_filter: str = "all",
    condition_filter: str = "all",
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Run LLM judge on all matching files."""
    files = iter_files(benchmark_filter, model_filter, condition_filter)

    print(f"\n{'=' * 60}")
    print(f"LLM MCQ Judge — Batch Run")
    print(f"Benchmarks: {benchmark_filter}, Models: {model_filter}, "
          f"Conditions: {condition_filter}")
    print(f"Files found: {len(files)}")
    print(f"{'=' * 60}\n")

    if dry_run:
        for model_short, bench, cond, fpath in files:
            judge_path = fpath.parent / f"{fpath.stem}_llm_judge.jsonl"
            status = "EXISTS" if judge_path.exists() else "NEW"
            print(f"  [{status}] {bench}/{MODEL_REGISTRY[model_short]['dir']}/{cond}")
        new = sum(1 for _, _, _, fp in files
                  if not (fp.parent / f"{fp.stem}_llm_judge.jsonl").exists())
        print(f"\n{new} new files to process, {len(files) - new} already done.")
        return

    results_summary = []
    for i, (model_short, bench, cond, fpath) in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {bench} / {MODEL_REGISTRY[model_short]['dir']} / {cond}")
        agg = process_file(fpath, benchmark=bench, force=force)
        if agg:
            results_summary.append({
                "model": model_short,
                "benchmark": bench,
                "condition": cond,
                "llm_accuracy": agg["overall_accuracy"],
                "official_accuracy": agg.get("official_comparison", {}).get("official_accuracy"),
                "delta": agg.get("official_comparison", {}).get("delta_mcq"),
                "agreement": agg.get("official_comparison", {}).get("agreement_rate"),
            })

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Bench':<10} {'Cond':<8} {'LLM%':>7} {'Off%':>7} {'Δ':>7} {'Agree':>7}")
    print("-" * 66)
    for r in results_summary:
        llm_pct = f"{r['llm_accuracy']:.1%}"
        off_pct = f"{r['official_accuracy']:.1%}" if r['official_accuracy'] is not None else "n/a"
        delta = f"{r['delta']:+.1%}" if r['delta'] is not None else "n/a"
        agree = f"{r['agreement']:.1%}" if r['agreement'] is not None else "n/a"
        print(f"{r['model']:<20} {r['benchmark']:<10} {r['condition']:<8} "
              f"{llm_pct:>7} {off_pct:>7} {delta:>7} {agree:>7}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM MCQ Judge — Claude Haiku answer extraction + correctness judgment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single file mode
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to a single inference JSONL/JSON file",
    )

    # Batch mode
    parser.add_argument("--batch", action="store_true", help="Batch mode: process multiple files")
    parser.add_argument("--benchmark", default="all", help="Benchmark filter (mmau, mmau_pro, mmar, all)")
    parser.add_argument("--model", default="all", help="Model filter (short name from constants.py, or all)")
    parser.add_argument("--condition", default="all", help="Condition filter (full, none, full,none, all)")

    # Options
    parser.add_argument("--force", action="store_true", help="Re-process even if judge output exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")

    args = parser.parse_args()

    if args.batch or args.dry_run:
        run_batch(
            benchmark_filter=args.benchmark,
            model_filter=args.model,
            condition_filter=args.condition,
            dry_run=args.dry_run,
            force=args.force,
        )
    elif args.input_file:
        process_file(Path(args.input_file), force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
