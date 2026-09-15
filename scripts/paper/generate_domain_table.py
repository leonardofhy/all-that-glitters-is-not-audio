#!/usr/bin/env python3
"""
Generate the model-averaged accuracy-by-category table (paper Table 5, tab:domain).

Each cell averages per-model values over the eight LALMs. Per-model values are first
rounded to one decimal place (as printed by generate_fine_grained.py), then averaged
with math.fsum and rounded again. F-N is the difference of the two unrounded means.
MCQ rows use the LLM-judge accuracy; the MMAU-Pro instruction-following (IF) and
open-ended (Open) rows use the official MMAU-Pro evaluation's performance score.

Usage:
    python scripts/paper/generate_domain_table.py [--results-dir results] [--latex]
"""

import argparse
import json
import math
from pathlib import Path

MODELS = [
    "audio_flamingo_3", "desta2.5", "phi4_multimodal", "qwen2_audio_7b_instruct",
    "qwen2.5_omni_7b", "qwen3_omni_30b_a3b_instruct", "qwen3_omni_30b_a3b_thinking",
    "voxtral_mini_3b",
]

# (category label, benchmark label, benchmark dir, result key, source)
# source "judge": *_llm_judge_results.json category_accuracy
# source "official": *_comprehensive_results.json category_results (MMAU-Pro evaluation)
ROWS = [
    ("IF", "Pro", "mmau_pro", "instruction following", "official"),
    ("Speech", "MMAU", "mmau", "speech", "judge"),
    ("Speech", "MMAR", "mmar", "speech", "judge"),
    ("Sound", "MMAU", "mmau", "sound", "judge"),
    ("Speech", "Pro", "mmau_pro", "speech", "judge"),
    ("Music", "MMAU", "mmau", "music", "judge"),
    ("Sound", "Pro", "mmau_pro", "sound", "judge"),
    ("Open", "Pro", "mmau_pro", "open", "official"),
]


def load_cell(results_dir: Path, benchmark: str, model: str, condition: str, key: str, source: str):
    """Return (score in [0, 1], item count) for one model/condition/category."""
    if source == "judge":
        path = results_dir / benchmark / model / f"{condition}_llm_judge_results.json"
        entry = json.loads(path.read_text())["category_accuracy"][key]
        return entry["accuracy"], entry["total"]
    path = results_dir / benchmark / model / f"{condition}_comprehensive_results.json"
    entry = json.loads(path.read_text())["category_results"][key]
    return entry["performance_score"], entry["count"]


def model_mean(values):
    rounded = [round(v * 100, 1) for v in values]
    return math.fsum(rounded) / len(rounded)


def compute_rows(results_dir: Path):
    rows = []
    for category, bench_label, benchmark, key, source in ROWS:
        full, none, n2 = [], [], []
        counts = set()
        for model in MODELS:
            score, count = load_cell(results_dir, benchmark, model, "full", key, source)
            full.append(score)
            counts.add(count)
            none.append(load_cell(results_dir, benchmark, model, "none", key, source)[0])
            chunks = [load_cell(results_dir, benchmark, model, f"n2_chunk{k}", key, source)[0] for k in (0, 1)]
            n2.append(sum(chunks) / 2)
        if len(counts) != 1:
            raise ValueError(f"{category}/{bench_label}: item counts differ across models: {sorted(counts)}")
        full_mean, none_mean = model_mean(full), model_mean(none)
        rows.append({
            "category": category,
            "benchmark": bench_label,
            "items": counts.pop(),
            "full": round(full_mean, 1),
            "n2": round(model_mean(n2), 1),
            "none": round(none_mean, 1),
            "full_minus_none": round(full_mean - none_mean, 1),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate paper Table 5 (accuracy by audio category)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table rows")
    args = parser.parse_args()

    rows = compute_rows(args.results_dir)
    if args.latex:
        for r in rows:
            items = f"{r['items']:,}".replace(",", "{,}")
            diff = f"$-${abs(r['full_minus_none']):.1f}" if r["full_minus_none"] < 0 else f"{r['full_minus_none']:.1f}"
            print(f"{r['category']:<8} & {r['benchmark']:<4} & {items:>7} & {r['full']:.1f} & "
                  f"{r['n2']:.1f} & {r['none']:.1f} & {diff} \\\\")
        return
    print(f"{'Category':<9}{'BM':<6}{'#Items':>7}{'Full':>7}{'N=2':>7}{'None':>7}{'F-N':>7}")
    for r in rows:
        print(f"{r['category']:<9}{r['benchmark']:<6}{r['items']:>7}{r['full']:>7.1f}{r['n2']:>7.1f}"
              f"{r['none']:>7.1f}{r['full_minus_none']:>7.1f}")


if __name__ == "__main__":
    main()
