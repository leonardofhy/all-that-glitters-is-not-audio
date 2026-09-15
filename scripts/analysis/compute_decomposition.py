#!/usr/bin/env python3
"""Compute the 5-category score decomposition (TS, FS, XS, AH, UN) from evaluation results.

Categories (mutually exclusive and exhaustive):
  TS (Text-Solvable):      Full correct  AND None correct
  AN (Audio-Needed):        Full correct  AND None incorrect
    FS (Fragment-Sufficient): AN AND at least one chunk correct (any N in {2,3,4,5})
    XS (Cross-Segment):      AN AND ALL chunks incorrect (all N, all K)
  AH (Audio-Harmful):      Full incorrect AND None correct
  UN (Unsolvable):          Full incorrect AND None incorrect

TS + FS + XS + AH + UN = |Q|
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARKS = ["mmau", "mmar", "mmau_pro"]
BENCHMARK_DISPLAY = {"mmau": "MMAU", "mmar": "MMAR", "mmau_pro": "MMAU-Pro"}

MODELS = [
    "audio_flamingo_3",
    "desta2.5",
    "phi4_multimodal",
    "qwen2_audio_7b_instruct",
    "qwen2.5_omni_7b",
    "qwen3_omni_30b_a3b_instruct",
    "qwen3_omni_30b_a3b_thinking",
    "voxtral_mini_3b",
]

CATEGORIES = ["TS", "FS", "XS", "AH", "UN"]


def load_llm_judge(path: Path) -> dict[str, bool]:
    """Load a _llm_judge.jsonl file and return {id: llm_correct}."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result[item["id"]] = bool(item["llm_correct"])
    return result


def load_errors(path: Path) -> set[str]:
    """Load base jsonl and return a set of IDs that contain an ERROR prediction."""
    err_ids = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if str(item.get("prediction", "")).startswith("ERROR"):
                    err_ids.add(item["id"])
    except FileNotFoundError:
        pass
    return err_ids


def decompose_model(model_dir: Path) -> dict | None:
    """Compute decomposition for a single model directory.

    Returns dict with counts and percentages, or None if required files are missing.
    """
    full_path = model_dir / "full_llm_judge.jsonl"
    none_path = model_dir / "none_llm_judge.jsonl"

    if not full_path.exists() or not none_path.exists():
        return None

    full_correct = load_llm_judge(full_path)
    none_correct = load_llm_judge(none_path)

    # Load all chunk results: {id: any_chunk_correct}
    chunk_any_correct: dict[str, bool] = {}
    chunk_any_error: set[str] = set()
    chunk_files_found = 0

    for n in range(2, 6):
        for k in range(n):
            chunk_path = model_dir / f"n{n}_chunk{k}_llm_judge.jsonl"
            pred_path = model_dir / f"n{n}_chunk{k}.jsonl"
            if not chunk_path.exists():
                continue
            chunk_files_found += 1
            chunk_correct = load_llm_judge(chunk_path)
            for item_id, correct in chunk_correct.items():
                if correct:
                    chunk_any_correct[item_id] = True
                elif item_id not in chunk_any_correct:
                    chunk_any_correct[item_id] = False
                    
            if pred_path.exists():
                errs = load_errors(pred_path)
                chunk_any_error.update(errs)

    # Expected: 2+3+4+5 = 14 chunk files
    if chunk_files_found < 14:
        print(f"  WARNING: only {chunk_files_found}/14 chunk files in {model_dir.name}")

    # Use intersection of IDs present in both full and none
    common_ids = sorted(set(full_correct.keys()) & set(none_correct.keys()))
    if not common_ids:
        return None

    counts = {"TS": 0, "FS": 0, "XS": 0, "AH": 0, "UN": 0}
    total = len(common_ids)

    for item_id in common_ids:
        fc = full_correct[item_id]
        nc = none_correct[item_id]

        if fc and nc:
            counts["TS"] += 1
        elif fc and not nc:
            # Audio-Needed → check chunks
            if chunk_any_correct.get(item_id, False) or item_id in chunk_any_error:
                counts["FS"] += 1
            else:
                counts["XS"] += 1
        elif not fc and nc:
            counts["AH"] += 1
        else:
            counts["UN"] += 1

    pcts = {cat: counts[cat] / total * 100 for cat in CATEGORIES}
    an_count = counts["FS"] + counts["XS"]
    xs_an_ratio = (counts["XS"] / an_count * 100) if an_count > 0 else 0.0

    return {
        "total": total,
        "counts": counts,
        "pcts": pcts,
        "xs_an_ratio": xs_an_ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute 5-category score decomposition (TS/FS/XS/AH/UN)."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (default: results/)",
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    # Collect all per-model results
    all_results: dict[str, dict[str, dict]] = {}  # benchmark -> model -> decomposition

    for benchmark in BENCHMARKS:
        bench_dir = results_dir / benchmark
        if not bench_dir.exists():
            print(f"Skipping {benchmark}: directory not found")
            continue

        all_results[benchmark] = {}
        for model in MODELS:
            model_dir = bench_dir / model
            if not model_dir.exists():
                continue
            result = decompose_model(model_dir)
            if result is not None:
                all_results[benchmark][model] = result

    # --- Print per-model tables ---
    for benchmark in BENCHMARKS:
        if benchmark not in all_results or not all_results[benchmark]:
            continue

        display = BENCHMARK_DISPLAY[benchmark]
        print(f"\n{'=' * 80}")
        print(f"  {display} — Per-Model Decomposition")
        print(f"{'=' * 80}")
        header = f"{'Model':<40} {'|Q|':>5}  " + "  ".join(f"{c:>6}" for c in CATEGORIES) + "  XS/AN%"
        print(header)
        print("-" * len(header))

        for model in MODELS:
            if model not in all_results[benchmark]:
                continue
            r = all_results[benchmark][model]
            row = f"{model:<40} {r['total']:>5}  "
            row += "  ".join(f"{r['pcts'][c]:>5.1f}%" for c in CATEGORIES)
            row += f"  {r['xs_an_ratio']:>5.1f}%"
            print(row)

    # --- Compute model-averaged percentages per benchmark ---
    print(f"\n{'=' * 80}")
    print("  Model-Averaged Decomposition (Paper Table 4)")
    print(f"{'=' * 80}")

    summary_rows = []
    header = f"{'Benchmark':<12}  " + "  ".join(f"{c:>6}" for c in CATEGORIES)
    header += "   XS/AN%  XS/AN_min  XS/AN_max"
    print(header)
    print("-" * len(header))

    for benchmark in BENCHMARKS:
        if benchmark not in all_results or not all_results[benchmark]:
            continue

        display = BENCHMARK_DISPLAY[benchmark]
        models_data = all_results[benchmark]
        n_models = len(models_data)

        avg_pcts = {}
        for cat in CATEGORIES:
            avg_pcts[cat] = sum(m["pcts"][cat] for m in models_data.values()) / n_models

        xs_an_ratios = [m["xs_an_ratio"] for m in models_data.values()]
        avg_xs_an = sum(xs_an_ratios) / n_models
        min_xs_an = min(xs_an_ratios)
        max_xs_an = max(xs_an_ratios)

        row = f"{display:<12}  "
        row += "  ".join(f"{avg_pcts[c]:>5.1f}%" for c in CATEGORIES)
        row += f"   {avg_xs_an:>5.1f}%  {min_xs_an:>8.1f}%  {max_xs_an:>8.1f}%"
        print(row)

        summary_rows.append({
            "benchmark": display,
            "n_models": n_models,
            **{cat: round(avg_pcts[cat], 1) for cat in CATEGORIES},
            "XS_AN": round(avg_xs_an, 1),
            "XS_AN_LOW": round(min_xs_an, 1),
            "XS_AN_HIGH": round(max_xs_an, 1),
        })

    # --- Cross-validate against previous summary (regression check) ---
    prev_path = results_dir / "decomposition_summary.json"
    if prev_path.exists():
        with open(prev_path, "r", encoding="utf-8") as f:
            prev_rows = {r["benchmark"]: r for r in json.load(f)}

        print(f"\n{'=' * 80}")
        print("  Regression Check vs. Previous Summary (tolerance: +/-0.2%)")
        print(f"{'=' * 80}")

        all_pass = True
        for row in summary_rows:
            display = row["benchmark"]
            if display not in prev_rows:
                print(f"  {display}: SKIP (not in previous summary)")
                continue
            prev = prev_rows[display]
            mismatches = []
            for cat in CATEGORIES:
                computed = row[cat]
                expected = prev[cat]
                if abs(computed - expected) > 0.2:
                    mismatches.append(f"{cat}: computed={computed:.1f} previous={expected:.1f}")

            if mismatches:
                print(f"  {display}: CHANGED")
                for m in mismatches:
                    print(f"    {m}")
                all_pass = False
            else:
                print(f"  {display}: PASS")

        if all_pass:
            print("\n  All benchmarks match previous summary.")
        else:
            print("\n  Some benchmarks differ from previous summary (review before overwriting).")
    else:
        print(f"\n  No previous summary at {prev_path}; skipping regression check.")
        all_pass = True

    # --- Save summary JSON ---
    output_path = results_dir / "decomposition_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {output_path}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
