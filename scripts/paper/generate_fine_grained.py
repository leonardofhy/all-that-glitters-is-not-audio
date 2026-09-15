#!/usr/bin/env python3
"""Generate fine-grained per-category tables for §5.3.

Outputs one table per benchmark (MMAU, MMAR, MMAU-Pro) with models as rows
and per-category accuracy columns.  Supports two chunked aggregation methods:
  AVG — average accuracy across chunks (default)
  OR  — oracle: item correct if ANY chunk gets it right (MCQ only)

Supports two MCQ scoring methods via --scorer:
  llm_judge  — Claude Haiku hybrid extraction (*_llm_judge_results.json)
  official   — MMAU/MMAR string-match (*_results.json),
               MMAU-Pro NVEmbed (*_comprehensive_results.json, closed-ended only)
Note: OR computation always uses llm_judge per-item files regardless of --scorer.

Usage:
    python scripts/paper/generate_fine_grained.py                          # all benchmarks, AVG
    python scripts/paper/generate_fine_grained.py --scorer official        # official scorer
    python scripts/paper/generate_fine_grained.py --benchmark mmau         # one benchmark
    python scripts/paper/generate_fine_grained.py --method both            # AVG + OR
    python scripts/paper/generate_fine_grained.py --conditions full,none   # subset
    python scripts/paper/generate_fine_grained.py --latex                  # LaTeX output
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

# ============================================================
# Configuration — shared with generate_tables.py
# ============================================================

MODELS = OrderedDict([
    ("Qwen3-Omni (I)", dict(
        dir="qwen3_omni_30b_a3b_instruct", size="30B")),
    ("Qwen3-Omni (T)", dict(
        dir="qwen3_omni_30b_a3b_thinking", size="30B")),
    ("Audio-Flamingo-3", dict(
        dir="audio_flamingo_3", size="7B")),
    ("Qwen2.5-Omni", dict(
        dir="qwen2.5_omni_7b", size="7B")),
    # Kimi-Audio removed from paper (unreliable None condition + official scorer anomaly)
    ("DeSTA-2.5", dict(
        dir="desta2.5", size="8B")),
    ("Qwen2-Audio", dict(
        dir="qwen2_audio_7b_instruct", size="7B")),
    ("Phi-4-MM", dict(
        dir="phi4_multimodal", size="14B")),
    ("Voxtral-Mini", dict(
        dir="voxtral_mini_3b", size="3B")),
])

# Per-benchmark category definitions.
# Keys: display name → (json_key, source).
# source: "llm_judge" reads from *_llm_judge_results.json category_accuracy
#          "comprehensive" reads from *_comprehensive_results.json category_results
BENCHMARK_CATEGORIES = {
    "mmau": OrderedDict([
        ("Sound",  ("sound",  "llm_judge")),
        ("Music",  ("music",  "llm_judge")),
        ("Speech", ("speech", "llm_judge")),
    ]),
    "mmar": OrderedDict([
        ("Snd",    ("sound",                "llm_judge")),
        ("Mus",    ("music",                "llm_judge")),
        ("Spe",    ("speech",               "llm_judge")),
        ("S-M",    ("mix-sound-music",      "llm_judge")),
        ("S-Sp",   ("mix-sound-speech",     "llm_judge")),
        ("M-Sp",   ("mix-music-speech",     "llm_judge")),
        ("S-M-Sp", ("mix-sound-music-speech", "llm_judge")),
    ]),
    "mmau_pro": OrderedDict([
        ("Snd",  ("sound",              "llm_judge")),
        ("Mus",  ("music",              "llm_judge")),
        ("Spe",  ("speech",             "llm_judge")),
        ("S-M",  ("sound_music",        "llm_judge")),
        ("Sp-M", ("music_speech",       "llm_judge")),
        ("Sp-S", ("sound_speech",       "llm_judge")),
        ("SMS",  ("sound_music_speech", "llm_judge")),
        ("Spa",  ("spatial_audio",      "llm_judge")),
        ("Voi",  ("voice_chat",         "llm_judge")),
        ("Mul",  ("multi",              "llm_judge")),
        ("Opn",  ("open",               "comprehensive")),
        ("IF",   ("instruction following", "comprehensive")),
    ]),
}

# Inference file category field name per benchmark.
CATEGORY_FIELD = {"mmau": "task", "mmar": "modality", "mmau_pro": "category"}

BM_DISPLAY = {"mmau": "MMAU", "mmar": "MMAR", "mmau_pro": "MMAU-Pro"}

ALL_CONDITIONS = ["full", "n2", "n3", "n4", "n5", "none"]
DEFAULT_CONDITIONS = ["full", "n2", "n5", "none"]
CHUNK_NS = {"n2": 2, "n3": 3, "n4": 4, "n5": 5}


# ============================================================
# File I/O helpers
# ============================================================

def read_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def read_jsonl(filepath):
    if not os.path.exists(filepath):
        return None
    items = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_source(bm, base_source, scorer):
    """Map (bm, base_source, scorer) → actual file source type.

    Returns one of: "llm_judge", "comprehensive", "official".
    """
    if base_source == "comprehensive":
        return "comprehensive"  # non-MCQ always from comprehensive
    # base_source == "llm_judge" (MCQ category)
    if scorer == "llm_judge":
        return "llm_judge"
    # scorer == "official"
    if bm in ("mmau", "mmar"):
        return "official"       # *_results.json
    return "comprehensive"      # mmau_pro MCQ → comprehensive closed-ended


def _result_filepath(results_dir, bm, model_dir, condition, source):
    """Build result file path for a given source type."""
    if source == "llm_judge":
        fname = f"{condition}_llm_judge_results.json"
    elif source == "comprehensive":
        fname = f"{condition}_comprehensive_results.json"
    elif source == "official":
        fname = f"{condition}_results.json"
    else:
        raise ValueError(f"Unknown source: {source}")
    return os.path.join(results_dir, bm, model_dir, fname)


def extract_category_acc(data, json_key, source, bm=None):
    """Extract accuracy for one category from a result file's data."""
    if source == "llm_judge":
        cats = data.get("category_accuracy", {})
        if json_key in cats:
            val = cats[json_key]
            return val["accuracy"] if isinstance(val, dict) else val
        return None
    elif source == "comprehensive":
        cats = data.get("category_results", {})
        if json_key in cats:
            return cats[json_key].get("performance_score")
        return None
    elif source == "official":
        # MMAU → task_accuracy, MMAR → modality_accuracy
        if bm == "mmau":
            cats = data.get("task_accuracy", {})
        elif bm == "mmar":
            cats = data.get("modality_accuracy", {})
        else:
            return None
        if json_key in cats:
            val = cats[json_key]
            return val["accuracy"] if isinstance(val, dict) else val
        return None
    return None


# ============================================================
# AVG computation for chunked conditions
# ============================================================

def compute_avg_chunked(results_dir, bm, model_dir, n, categories,
                        scorer="llm_judge"):
    """Compute AVG per-category accuracy across N chunks.

    For each category, averages the per-chunk accuracy from the appropriate
    result file based on the scorer.
    """
    cat_sums = {disp: 0.0 for disp in categories}
    cat_counts = {disp: 0 for disp in categories}

    for k in range(n):
        for disp, (json_key, base_source) in categories.items():
            source = resolve_source(bm, base_source, scorer)
            fp = _result_filepath(results_dir, bm, model_dir,
                                  f"n{n}_chunk{k}", source)
            data = read_json(fp)
            if data is None:
                continue
            acc = extract_category_acc(data, json_key, source, bm)
            if acc is not None:
                cat_sums[disp] += acc
                cat_counts[disp] += 1

    result = {}
    for disp in categories:
        if cat_counts[disp] == n:
            result[disp] = cat_sums[disp] / n
        else:
            result[disp] = None
    return result


# ============================================================
# OR computation for chunked conditions
# ============================================================

def _load_inference_items(results_dir, bm, model_dir, filename):
    """Load items from an inference file (JSONL or JSON array)."""
    fp = os.path.join(results_dir, bm, model_dir, filename)
    if bm == "mmar":
        data = read_json(fp)
        return data if data else None
    else:
        return read_jsonl(fp)


def load_item_categories(results_dir, bm, model_dir):
    """Load category mapping from inference file (id → category).

    Tries multiple candidate files and picks the one with the most items
    to guard against broken/incomplete inference files.
    """
    cat_field = CATEGORY_FIELD[bm]
    ext = ".json" if bm == "mmar" else ".jsonl"

    # Candidate files in priority order
    candidates = [f"full{ext}"]
    for n in (2, 3, 4, 5):
        candidates.append(f"n{n}_chunk0{ext}")

    best_mapping = None
    best_count = 0

    for fname in candidates:
        items = _load_inference_items(results_dir, bm, model_dir, fname)
        if items and len(items) > best_count:
            best_mapping = {item["id"]: item[cat_field] for item in items}
            best_count = len(items)

    return best_mapping


def compute_or_chunked(results_dir, bm, model_dir, n, categories,
                       item_categories, scorer="llm_judge"):
    """Compute OR per-category accuracy across N chunks.

    For MCQ items: correct if ANY chunk gets the item right.
    Always uses *_llm_judge.jsonl per-item files (official eval has no
    per-item output).
    Non-MCQ categories (comprehensive) fall back to AVG.
    """
    if item_categories is None:
        return {disp: None for disp in categories}

    # Separate MCQ vs non-MCQ categories
    mcq_cats = {d: (k, s) for d, (k, s) in categories.items()
                if s == "llm_judge"}
    non_mcq_cats = {d: (k, s) for d, (k, s) in categories.items()
                    if s != "llm_judge"}

    # --- MCQ: OR logic ---
    # Build per-item OR correctness
    item_correct = {}  # id → bool (OR across chunks)
    all_chunks_loaded = True

    for k in range(n):
        fp = os.path.join(results_dir, bm, model_dir,
                          f"n{n}_chunk{k}_llm_judge.jsonl")
        chunk_items = read_jsonl(fp)
        if not chunk_items:  # None (missing) or [] (empty)
            all_chunks_loaded = False
            break
        for item in chunk_items:
            iid = item["id"]
            if iid not in item_correct:
                item_correct[iid] = False
            if item.get("llm_correct", False):
                item_correct[iid] = True

    result = {}
    if all_chunks_loaded and item_correct:
        # Warn if many items lack category mapping (broken inference file)
        unmapped = sum(1 for iid in item_correct
                       if item_categories.get(iid) is None)
        if unmapped > 0:
            pct = unmapped / len(item_correct) * 100
            print(f"WARNING: {model_dir} N={n}: {unmapped}/{len(item_correct)}"
                  f" ({pct:.0f}%) items lack category mapping",
                  file=sys.stderr)

        # Group by category and compute accuracy
        for disp, (json_key, _) in mcq_cats.items():
            correct = 0
            total = 0
            for iid, is_correct in item_correct.items():
                cat = item_categories.get(iid)
                if cat == json_key:
                    total += 1
                    if is_correct:
                        correct += 1
            result[disp] = correct / total if total > 0 else None
    else:
        for disp in mcq_cats:
            result[disp] = None

    # --- Non-MCQ: fall back to AVG ---
    if non_mcq_cats:
        avg = compute_avg_chunked(results_dir, bm, model_dir, n, non_mcq_cats,
                                  scorer)
        result.update(avg)

    return result


# ============================================================
# Full / None computation
# ============================================================

def compute_full_or_none(results_dir, bm, model_dir, condition, categories,
                         scorer="llm_judge"):
    """Read per-category accuracy for full or none condition."""
    result = {}
    for disp, (json_key, base_source) in categories.items():
        source = resolve_source(bm, base_source, scorer)
        fp = _result_filepath(results_dir, bm, model_dir, condition, source)
        data = read_json(fp)
        if data is None:
            result[disp] = None
            continue
        result[disp] = extract_category_acc(data, json_key, source, bm)
    return result


# ============================================================
# Weighted average
# ============================================================

def compute_weighted_avg(cat_accs, results_dir, bm, model_dir, categories,
                         scorer="llm_judge"):
    """Compute sample-count weighted average across MCQ categories.

    Avg is always MCQ-only regardless of scorer (consistent scale).
    """
    # Get sample counts from the full result file matching the scorer
    source = resolve_source(bm, "llm_judge", scorer)  # MCQ source
    fp = _result_filepath(results_dir, bm, model_dir, "full", source)
    data = read_json(fp)
    if data is None:
        return None

    # Extract {json_key: total} mapping from the correct data structure
    if source == "llm_judge":
        cat_acc_data = data.get("category_accuracy", {})
    elif source == "comprehensive":
        # MMAU-Pro comprehensive: category_results.*.count (closed only)
        cat_acc_data = {}
        for key, val in data.get("category_results", {}).items():
            if val.get("type") == "closed":
                cat_acc_data[key] = {"total": val.get("count", 0)}
    elif source == "official":
        # MMAU → task_accuracy, MMAR → modality_accuracy
        cat_key = "task_accuracy" if bm == "mmau" else "modality_accuracy"
        cat_acc_data = data.get(cat_key, {})
    else:
        cat_acc_data = {}

    total_correct = 0
    total_items = 0

    for disp, (json_key, base_source) in categories.items():
        if base_source != "llm_judge":
            continue  # MCQ-only weighting
        acc = cat_accs.get(disp)
        if acc is None:
            continue
        if json_key in cat_acc_data:
            val = cat_acc_data[json_key]
            n_items = val["total"] if isinstance(val, dict) else None
            if n_items:
                total_correct += acc * n_items
                total_items += n_items

    return total_correct / total_items if total_items > 0 else None


# ============================================================
# Main computation
# ============================================================

def compute_benchmark_table(results_dir, bm, conditions, methods,
                            scorer="llm_judge"):
    """Compute full fine-grained table for one benchmark.

    Returns: {condition_label: {model_name: {category: accuracy, "Avg": avg}}}
    """
    categories = BENCHMARK_CATEGORIES[bm]
    tables = {}

    for cond in conditions:
        if cond in ("full", "none"):
            label = cond.capitalize()
            tables[label] = {}
            for name, info in MODELS.items():
                accs = compute_full_or_none(
                    results_dir, bm, info["dir"], cond, categories, scorer)
                accs["Avg"] = compute_weighted_avg(
                    accs, results_dir, bm, info["dir"], categories, scorer)
                tables[label][name] = accs

        elif cond in CHUNK_NS:
            n = CHUNK_NS[cond]
            if "avg" in methods:
                label = f"N={n} (AVG)"
                tables[label] = {}
                for name, info in MODELS.items():
                    accs = compute_avg_chunked(
                        results_dir, bm, info["dir"], n, categories, scorer)
                    accs["Avg"] = compute_weighted_avg(
                        accs, results_dir, bm, info["dir"], categories,
                        scorer)
                    tables[label][name] = accs

            if "or" in methods:
                label = f"N={n} (OR)"
                tables[label] = {}
                for name, info in MODELS.items():
                    item_cats = load_item_categories(
                        results_dir, bm, info["dir"])
                    accs = compute_or_chunked(
                        results_dir, bm, info["dir"], n, categories,
                        item_cats, scorer)
                    accs["Avg"] = compute_weighted_avg(
                        accs, results_dir, bm, info["dir"], categories,
                        scorer)
                    tables[label][name] = accs

    return tables


# ============================================================
# Output — Markdown
# ============================================================

def fmt(val, decimals=1):
    if val is None:
        return "--"
    return f"{val * 100:.{decimals}f}"


def print_table_md(bm, tables):
    categories = BENCHMARK_CATEGORIES[bm]
    col_names = list(categories.keys()) + ["Avg"]

    print(f"\n{'=' * 100}")
    print(f"{BM_DISPLAY[bm]} — Fine-Grained Per-Category Accuracy (%)")
    print(f"{'=' * 100}")

    for label, model_data in tables.items():
        print(f"\n--- {label} ---")
        # Header
        hdr = f"{'Model':<20} {'Size':>4}"
        for col in col_names:
            hdr += f" {col:>6}"
        print(hdr)
        print("-" * len(hdr))

        for name in MODELS:
            if name not in model_data:
                continue
            accs = model_data[name]
            line = f"{name:<20} {MODELS[name]['size']:>4}"
            for col in col_names:
                line += f" {fmt(accs.get(col)):>6}"
            print(line)


# ============================================================
# Output — LaTeX
# ============================================================

def print_table_latex(bm, tables):
    categories = BENCHMARK_CATEGORIES[bm]
    col_names = list(categories.keys()) + ["Avg"]

    print(f"\n% === {BM_DISPLAY[bm]} Fine-Grained LaTeX ===")

    for label, model_data in tables.items():
        print(f"% --- {label} ---")
        for name in MODELS:
            if name not in model_data:
                continue
            accs = model_data[name]
            vals = [fmt(accs.get(col)) for col in col_names]
            row = " & ".join(vals)
            print(f"{name:<20} & {MODELS[name]['size']:>3} & {row} \\\\")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate fine-grained per-category tables for §5.3")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory")
    parser.add_argument("--benchmark", default="all",
                        choices=["mmau", "mmar", "mmau_pro", "all"],
                        help="Which benchmark(s) to generate")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated conditions (default: full,n2,n5,none)")
    parser.add_argument("--method", default="avg",
                        choices=["avg", "or", "both"],
                        help="Chunked aggregation method (default: avg)")
    parser.add_argument("--scorer", default="llm_judge",
                        choices=["official", "llm_judge"],
                        help="MCQ scoring method (default: llm_judge)")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX formatted rows")
    args = parser.parse_args()

    benchmarks = ["mmau", "mmar", "mmau_pro"] if args.benchmark == "all" \
        else [args.benchmark]
    conditions = args.conditions.split(",") if args.conditions \
        else DEFAULT_CONDITIONS
    methods = {"avg", "or"} if args.method == "both" \
        else {args.method}
    scorer = args.scorer

    print(f"[scorer={scorer}]")

    if scorer != "llm_judge" and "or" in methods:
        print("NOTE: OR always uses llm_judge per-item files "
              "(official eval has no per-item output). "
              "OR values may not be directly comparable to AVG "
              f"values from the {scorer} scorer.",
              file=sys.stderr)

    for bm in benchmarks:
        tables = compute_benchmark_table(
            args.results_dir, bm, conditions, methods, scorer)
        print_table_md(bm, tables)
        if args.latex:
            print_table_latex(bm, tables)


if __name__ == "__main__":
    main()
