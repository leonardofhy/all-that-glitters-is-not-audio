#!/usr/bin/env python3
"""Generate paper tables (Table 1, 2, 3) from result files.

Supports two MCQ scoring methods via --scorer:
  llm_judge  — Claude Haiku hybrid extraction (*_llm_judge_results.json)
  official   — MMAU/MMAR string-match (*_results.json),
               MMAU-Pro NVEmbed (*_comprehensive_results.json, closed-ended only)

Usage:
    python scripts/paper/generate_tables.py                    # markdown
    python scripts/paper/generate_tables.py --scorer official  # official scorer
    python scripts/paper/generate_tables.py --latex            # LaTeX rows
    python scripts/paper/generate_tables.py --prose            # §5.1/5.2 stats
    python scripts/paper/generate_tables.py --appendix         # per-model retention
    python scripts/paper/generate_tables.py --check            # data quality check
    python scripts/paper/generate_tables.py --all              # everything
"""

import argparse
import json
import os
from collections import OrderedDict

# ============================================================
# Configuration
# ============================================================

BENCHMARKS = ["mmau", "mmar", "mmau_pro"]
BM_DISPLAY = {"mmau": "MMAU", "mmar": "MMAR", "mmau_pro": "MMAU-Pro"}
CHUNKS = [2, 3, 4, 5]

# Model registry — alphabetically sorted by model name for stable ordering.
# tb_dir: directory containing the text backbone's None results.
# tb_note: "dagger" → TB was run without thinking mode.
# exclude_none: set of benchmarks where None is broken (display as --).
MODELS = OrderedDict([
    ("Audio-Flamingo-3", dict(
        dir="audio_flamingo_3",
        tb_dir="tb_qwen2.5_7b_instruct",
        tb_note=None,
        exclude_none=set(),
    )),
    ("DeSTA-2.5", dict(
        dir="desta2.5",
        tb_dir="tb_llama_3.1_8b_instruct",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Phi-4-MM", dict(
        dir="phi4_multimodal",
        tb_dir="tb_phi4_mini_instruct",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Qwen2-Audio", dict(
        dir="qwen2_audio_7b_instruct",
        tb_dir="tb_qwen_7b_chat",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Qwen2.5-Omni", dict(
        dir="qwen2.5_omni_7b",
        tb_dir="tb_qwen2.5_7b_instruct",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Qwen3-Omni (I)", dict(
        dir="qwen3_omni_30b_a3b_instruct",
        tb_dir="tb_qwen3_30b_a3b_instruct",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Qwen3-Omni (T)", dict(
        dir="qwen3_omni_30b_a3b_thinking",
        tb_dir="tb_qwen3_30b_a3b_thinking",
        tb_note=None,
        exclude_none=set(),
    )),
    ("Voxtral-Mini", dict(
        dir="voxtral_mini_3b",
        tb_dir="tb_ministral_3b",
        tb_note=None,
        exclude_none=set(),
    )),
    # Kimi-Audio removed from paper (unreliable None condition + official scorer anomaly)
])

# Domain keys per benchmark (for Table 3).
DOMAIN_KEYS = {
    "mmau": {"Sound": "sound", "Music": "music", "Speech": "speech"},
    "mmar": {"Sound": "sound", "Music": "music", "Speech": "speech"},
    "mmau_pro": {"Sound": "sound", "Music": "music", "Speech": "speech"},
}


# ============================================================
# File I/O
# ============================================================

def result_path(results_dir: str, bm: str, model_dir: str,
                condition: str, scorer: str = "llm_judge") -> str:
    """Build result file path based on benchmark and scorer."""
    if scorer == "llm_judge":
        suffix = "_llm_judge_results.json"
    elif bm == "mmau_pro":
        suffix = "_comprehensive_results.json"
    else:  # mmau, mmar official → *_results.json
        suffix = "_results.json"
    return os.path.join(results_dir, bm, model_dir, f"{condition}{suffix}")


def read_result(filepath: str, bm: str = None,
                scorer: str = "llm_judge") -> dict | None:
    """Read a result file and normalise to a common dict.

    Returns: {accuracy, correct, total, no_pred, category_accuracy} or None.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        data = json.load(f)

    if scorer == "llm_judge":
        return {
            "accuracy": data.get("overall_accuracy"),
            "correct": data.get("overall_correct"),
            "total": data.get("overall_total"),
            "no_pred": data.get("no_prediction_count", 0),
            "category_accuracy": data.get("category_accuracy", {}),
        }

    # --- Official scorer ---
    if bm == "mmau_pro":
        # comprehensive_results.json — extract closed-ended (MCQ) only
        cat_results = data.get("category_results", {})
        cat_acc = {}
        mcq_correct = 0
        mcq_total = 0
        for key, val in cat_results.items():
            if val.get("type") != "closed":
                continue
            count = val.get("count", 0)
            perf = val.get("performance_score")
            if perf is not None:
                correct = round(perf * count)
                cat_acc[key] = {
                    "accuracy": perf,
                    "correct": correct,
                    "total": count,
                }
                mcq_correct += correct
                mcq_total += count
        return {
            "accuracy": mcq_correct / mcq_total if mcq_total > 0 else None,
            "correct": mcq_correct,
            "total": mcq_total,
            "no_pred": 0,
            "category_accuracy": cat_acc,
        }

    # MMAU / MMAR official — *_results.json
    cat_key = "task_accuracy" if bm == "mmau" else "modality_accuracy"
    return {
        "accuracy": data.get("overall_accuracy"),
        "correct": data.get("overall_correct"),
        "total": data.get("overall_total"),
        "no_pred": data.get("no_prediction_count", 0),
        "category_accuracy": data.get(cat_key, {}),
    }


def pct(val: float | None, decimals: int = 1) -> str:
    """Format a 0-1 accuracy float as percentage string, or '--'."""
    if val is None:
        return "--"
    return f"{val * 100:.{decimals}f}"


def is_none_excluded(name: str, bm: str) -> bool:
    """Check whether this model's None condition should be displayed as --."""
    return bm in MODELS[name]["exclude_none"]


# ============================================================
# Table computations
# ============================================================

def compute_table1(results_dir: str, scorer: str = "llm_judge") -> dict:
    """Table 1: Full / None / TB accuracy per model × benchmark."""
    table = {}
    for bm in BENCHMARKS:
        table[bm] = {}
        for name, info in MODELS.items():
            row = {}
            for cond in ("full", "none"):
                path = result_path(results_dir, bm, info["dir"], cond,
                                   scorer)
                result = read_result(path, bm, scorer)
                acc = result["accuracy"] if result else None
                # Mark excluded None as display-only
                if cond == "none" and is_none_excluded(name, bm):
                    row["none"] = None
                    row["none_raw"] = acc  # keep raw for diagnostics
                else:
                    row[cond] = acc
                row[f"{cond}_detail"] = result

            # Text backbone
            tb_path = result_path(results_dir, bm, info["tb_dir"], "none",
                                  scorer)
            tb_result = read_result(tb_path, bm, scorer)
            row["tb"] = tb_result["accuracy"] if tb_result else None
            row["tb_note"] = info["tb_note"]
            table[bm][name] = row
    return table


def compute_table2(results_dir: str,
                    scorer: str = "llm_judge") -> tuple[dict, dict]:
    """Table 2: average retention rates + per-model detail."""
    averages = {}
    details = {}

    for bm in BENCHMARKS:
        chunk_vals = {n: [] for n in CHUNKS}
        none_vals = []
        model_ret = {}

        for name, info in MODELS.items():
            full = read_result(
                result_path(results_dir, bm, info["dir"], "full", scorer),
                bm, scorer)
            if not full or not full["accuracy"]:
                continue
            full_acc = full["accuracy"]

            none = read_result(
                result_path(results_dir, bm, info["dir"], "none", scorer),
                bm, scorer)
            none_acc = none["accuracy"] if none else None

            ret = {}
            all_chunks_complete = True
            for n in CHUNKS:
                accs = []
                for k in range(n):
                    path = result_path(results_dir, bm, info["dir"],
                                       f"n{n}_chunk{k}", scorer)
                    r = read_result(path, bm, scorer)
                    if r and r["accuracy"] is not None:
                        accs.append(r["accuracy"])
                if len(accs) == n:
                    ret[n] = (sum(accs) / n / full_acc) * 100
                else:
                    ret[n] = None
                    all_chunks_complete = False

            # None retention — skip excluded models
            if is_none_excluded(name, bm) or none_acc is None:
                ret["none"] = None
            else:
                ret["none"] = (none_acc / full_acc) * 100

            model_ret[name] = ret
            if all_chunks_complete:
                for n in CHUNKS:
                    chunk_vals[n].append(ret[n])
            if ret["none"] is not None:
                none_vals.append(ret["none"])

        n_chunk = len(chunk_vals[CHUNKS[0]])
        n_none = len(none_vals)
        avg = {
            n: sum(chunk_vals[n]) / n_chunk if n_chunk else None
            for n in CHUNKS
        }
        avg["none"] = sum(none_vals) / n_none if n_none else None
        avg["n_chunk"] = n_chunk
        avg["n_none"] = n_none

        averages[bm] = avg
        details[bm] = model_ret

    return averages, details


def compute_table3(results_dir: str,
                    scorer: str = "llm_judge") -> dict:
    """Table 3: model-averaged domain accuracy (Full & None)."""
    table = {}
    for bm in BENCHMARKS:
        domain_full = {d: [] for d in DOMAIN_KEYS[bm]}
        domain_none = {d: [] for d in DOMAIN_KEYS[bm]}

        for name, info in MODELS.items():
            if is_none_excluded(name, bm):
                continue
            full = read_result(
                result_path(results_dir, bm, info["dir"], "full", scorer),
                bm, scorer)
            none = read_result(
                result_path(results_dir, bm, info["dir"], "none", scorer),
                bm, scorer)
            if not full or not none or none["accuracy"] is None:
                continue

            full_cats = full.get("category_accuracy", {})
            none_cats = none.get("category_accuracy", {})
            for display, key in DOMAIN_KEYS[bm].items():
                for src, dst in [(full_cats, domain_full),
                                 (none_cats, domain_none)]:
                    if key in src:
                        val = src[key]
                        acc = val["accuracy"] if isinstance(val, dict) else val
                        dst[display].append(acc)

        table[bm] = {}
        for d in DOMAIN_KEYS[bm]:
            fv, nv = domain_full[d], domain_none[d]
            table[bm][d] = {
                "full": sum(fv) / len(fv) * 100 if fv else None,
                "none": sum(nv) / len(nv) * 100 if nv else None,
                "n_models": len(fv),
            }
    return table


# ============================================================
# Output — Markdown
# ============================================================

def print_table1_md(table1: dict):
    print("\n" + "=" * 90)
    print("TABLE 1: Full / None / TB Accuracy (%) and Text-Prior Rate")
    print("=" * 90)
    hdr = f"{'Model':<20}"
    for bm in BENCHMARKS:
        hdr += f" | {BM_DISPLAY[bm]+' Full':>9} {'None':>5} {'TB':>6} {'RTP':>5}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        line = f"{name:<20}"
        for bm in BENCHMARKS:
            row = table1[bm][name]
            f, n, t = pct(row["full"]), pct(row["none"]), pct(row["tb"])
            rtp = pct(row["none"] / row["full"]) if row["full"] and row["none"] else "--"
            if row["tb_note"] == "dagger":
                t += "†"
            line += f" | {f:>9} {n:>5} {t:>6} {rtp:>5}"
        print(line)

    # OVERALL row: average across all models
    print("-" * len(hdr))
    line = f"{'OVERALL':<20}"
    for bm in BENCHMARKS:
        fulls = [table1[bm][m]["full"] for m in MODELS if table1[bm][m]["full"]]
        nones = [table1[bm][m]["none"] for m in MODELS if table1[bm][m]["none"]]
        tbs = [table1[bm][m]["tb"] for m in MODELS if table1[bm][m]["tb"]]

        avg_full = sum(fulls) / len(fulls) * 100 if fulls else None
        avg_none = sum(nones) / len(nones) * 100 if nones else None
        avg_tb = sum(tbs) / len(tbs) * 100 if tbs else None
        avg_rtp = (sum(nones) / sum(fulls) * 100) if fulls and nones else None

        f = f"{avg_full:.1f}" if avg_full else "--"
        n = f"{avg_none:.1f}" if avg_none else "--"
        t = f"{avg_tb:.1f}" if avg_tb else "--"
        rtp = f"{avg_rtp:.1f}" if avg_rtp else "--"
        line += f" | {f:>9} {n:>5} {t:>6} {rtp:>5}"
    print(line)


def print_table2_md(table2: dict):
    print("\n" + "=" * 70)
    print("TABLE 2: Average Retention Rate (%)")
    print("=" * 70)
    print(f"{'Benchmark':<12} | {'N=2':>6} {'N=3':>6} {'N=4':>6} "
          f"{'N=5':>6} {'None':>6} | #chunk #none")
    print("-" * 70)
    for bm in BENCHMARKS:
        a = table2[bm]
        vs = [f"{a[n]:.1f}" if a[n] is not None else "--" for n in CHUNKS]
        nv = f"{a['none']:.1f}" if a["none"] is not None else "--"
        print(f"{BM_DISPLAY[bm]:<12} | {vs[0]:>6} {vs[1]:>6} "
              f"{vs[2]:>6} {vs[3]:>6} {nv:>6} "
              f"| {a['n_chunk']:>4}  {a['n_none']:>4}")


def print_table3_md(table3: dict):
    print("\n" + "=" * 80)
    print("TABLE 3: Model-Averaged Accuracy (%) by Domain")
    print("=" * 80)
    hdr = f"{'Domain':<8}"
    for bm in BENCHMARKS:
        hdr += f" | {BM_DISPLAY[bm]+' Full':>12} {BM_DISPLAY[bm]+' None':>12}"
    print(hdr)
    print("-" * 80)
    for d in ("Sound", "Music", "Speech"):
        line = f"{d:<8}"
        for bm in BENCHMARKS:
            entry = table3[bm].get(d, {"full": None, "none": None})
            fv = f"{entry['full']:.1f}" if entry["full"] is not None else "--"
            nv = f"{entry['none']:.1f}" if entry["none"] is not None else "--"
            line += f" | {fv:>12} {nv:>12}"
        print(line)
    for bm in BENCHMARKS:
        d0 = next(iter(table3[bm]))
        print(f"  {BM_DISPLAY[bm]}: {table3[bm][d0]['n_models']} models")


# ============================================================
# Output — LaTeX
# ============================================================

def print_table1_latex(table1: dict):
    print("\n% === TABLE 1 LaTeX rows ===")
    for name in MODELS:
        parts = []
        for bm in BENCHMARKS:
            row = table1[bm][name]
            f, n, t = pct(row["full"]), pct(row["none"]), pct(row["tb"])
            rtp = pct(row["none"] / row["full"]) if row["full"] and row["none"] else "--"
            if row["tb_note"] == "dagger":
                t += r"\rlap{$^\dagger$}"
            parts.extend([f, n, t, rtp])
        vals = " & ".join(parts)
        print(f"{name:<20} & {vals} \\\\")

    # OVERALL row
    print("\\midrule")
    parts = []
    for bm in BENCHMARKS:
        fulls = [table1[bm][m]["full"] for m in MODELS if table1[bm][m]["full"]]
        nones = [table1[bm][m]["none"] for m in MODELS if table1[bm][m]["none"]]
        tbs = [table1[bm][m]["tb"] for m in MODELS if table1[bm][m]["tb"]]

        avg_full = sum(fulls) / len(fulls) * 100 if fulls else None
        avg_none = sum(nones) / len(nones) * 100 if nones else None
        avg_tb = sum(tbs) / len(tbs) * 100 if tbs else None
        avg_rtp = (sum(nones) / sum(fulls) * 100) if fulls and nones else None

        f = f"{avg_full:.1f}" if avg_full else "--"
        n = f"{avg_none:.1f}" if avg_none else "--"
        t = f"{avg_tb:.1f}" if avg_tb else "--"
        rtp = f"{avg_rtp:.1f}" if avg_rtp else "--"
        parts.extend([rf"\textbf{{{f}}}", rf"\textbf{{{n}}}",
                      rf"\textbf{{{t}}}", rf"\textbf{{{rtp}}}"])
    vals = " & ".join(parts)
    print(rf"\textbf{{OVERALL}}            & {vals} \\\\")


def print_table2_latex(table2: dict):
    print("\n% === TABLE 2 LaTeX rows ===")
    for bm in BENCHMARKS:
        a = table2[bm]
        vs = [f"{a[n]:.1f}" if a[n] is not None else "--" for n in CHUNKS]
        nv = f"{a['none']:.1f}" if a["none"] is not None else "--"
        row = " & ".join(vs + [nv])
        print(f"{BM_DISPLAY[bm]:<10} & {row} \\\\")


def print_table3_latex(table3: dict):
    print("\n% === TABLE 3 LaTeX rows ===")
    for d in ("Sound", "Music", "Speech"):
        parts = []
        for bm in BENCHMARKS:
            entry = table3[bm].get(d, {"full": None, "none": None})
            fv = f"{entry['full']:.1f}" if entry["full"] is not None else "--"
            nv = f"{entry['none']:.1f}" if entry["none"] is not None else "--"
            parts.extend([fv, nv])
        print(f"{d:<8} & {' & '.join(parts)} \\\\")


# ============================================================
# Output — Appendix per-model retention
# ============================================================

def print_appendix(details: dict):
    print("\n" + "=" * 70)
    print("APPENDIX: Per-Model Retention Rates (%)")
    print("=" * 70)
    for bm in BENCHMARKS:
        print(f"\n--- {BM_DISPLAY[bm]} ---")
        print(f"{'Model':<20} | {'N=2':>6} {'N=3':>6} {'N=4':>6} "
              f"{'N=5':>6} {'None':>6}")
        print("-" * 60)
        for name in MODELS:
            if name not in details[bm]:
                continue
            ret = details[bm][name]
            vs = [f"{ret[n]:.1f}" if ret.get(n) is not None else "--"
                  for n in CHUNKS]
            nv = f"{ret['none']:.1f}" if ret.get("none") is not None else "--"
            print(f"{name:<20} | {vs[0]:>6} {vs[1]:>6} "
                  f"{vs[2]:>6} {vs[3]:>6} {nv:>6}")


# ============================================================
# Output — Prose statistics for §5.1 / §5.2
# ============================================================

def print_prose(table1: dict, table2: dict):
    print("\n" + "=" * 70)
    print("PROSE STATISTICS (for §5.1 and §5.2)")
    print("=" * 70)

    # §5.1
    print("\n§5.1:")
    tb_vals = [table1["mmau"][m]["tb"] for m in MODELS
               if table1["mmau"][m]["tb"] is not None]
    print(f"  TB range on MMAU: {min(tb_vals)*100:.1f}--"
          f"{max(tb_vals)*100:.1f}%")

    print("  Wrapper gain (None - TB) on MMAU:")
    for name in MODELS:
        r = table1["mmau"][name]
        if r["none"] is not None and r["tb"] is not None:
            print(f"    {name:<20}: {(r['none']-r['tb'])*100:+.1f} pp")

    for bm in ("mmau", "mmar"):
        gaps = [(table1[bm][m]["full"] - table1[bm][m]["none"]) * 100
                for m in MODELS
                if table1[bm][m]["full"] is not None
                and table1[bm][m]["none"] is not None]
        print(f"  Full-None gap on {BM_DISPLAY[bm]}: "
              f"{min(gaps):.1f}--{max(gaps):.1f} pp")

    # §5.2
    print("\n§5.2:")
    for label, n in [("N=2", 2), ("N=5", 5)]:
        vs = [table2[bm][n] for bm in BENCHMARKS
              if table2[bm][n] is not None]
        print(f"  {label} range: {min(vs):.1f}--{max(vs):.1f}%")

    drops = [table2[bm][2] - table2[bm][5] for bm in BENCHMARKS
             if table2[bm][2] and table2[bm][5]]
    print(f"  N=2→N=5 drop: {min(drops):.1f}--{max(drops):.1f} pp")

    gaps = [table2[bm][5] - table2[bm]["none"] for bm in BENCHMARKS
            if table2[bm][5] and table2[bm]["none"]]
    print(f"  N=5→None gap: {min(gaps):.1f}--{max(gaps):.1f} pp")

    # MMAU recovery
    mn, m2 = table2["mmau"]["none"], table2["mmau"][2]
    contrib = 100 - mn
    recov = m2 - mn
    print(f"  MMAU: None={mn:.1f}%, loss={contrib:.1f} pp, "
          f"N=2 recovers {recov:.1f} pp ({recov/contrib*100:.1f}%)")


# ============================================================
# Data quality check
# ============================================================

def print_check(results_dir: str, table1: dict, details: dict,
                scorer: str = "llm_judge"):
    print("\n" + "=" * 70)
    print(f"DATA QUALITY CHECK (scorer={scorer})")
    print("=" * 70)

    # MMAU-Pro MCQ total is 4592 for both scorers (official extracts
    # closed-ended only).
    expected = {"mmau": 1000, "mmar": 1000, "mmau_pro": 4592}

    warnings = []
    for bm in BENCHMARKS:
        for name, info in MODELS.items():
            for cond in ("full", "none"):
                path = result_path(results_dir, bm, info["dir"], cond,
                                   scorer)
                r = read_result(path, bm, scorer)
                if r is None:
                    warnings.append(f"MISSING: {bm}/{info['dir']}/{cond}")
                    continue
                total = r["total"]
                no_pred = r["no_pred"]
                if total != expected.get(bm):
                    warnings.append(
                        f"WRONG TOTAL: {bm}/{info['dir']}/{cond}: "
                        f"{total} (expected {expected[bm]})")
                # High no-prediction rate (>30%)
                if total and no_pred / total > 0.3:
                    warnings.append(
                        f"HIGH NO-PRED: {bm}/{info['dir']}/{cond}: "
                        f"{no_pred}/{total} ({no_pred/total*100:.0f}%)")

            # TB check
            tb_path = result_path(results_dir, bm, info["tb_dir"], "none",
                                  scorer)
            r = read_result(tb_path, bm, scorer)
            if r is None:
                warnings.append(f"MISSING TB: {bm}/{info['tb_dir']}/none")

        # Retention >100% check
        for name in MODELS:
            if name not in details[bm]:
                continue
            for key in list(CHUNKS) + ["none"]:
                v = details[bm][name].get(key)
                if v is not None and v > 100:
                    label = f"N={key}" if isinstance(key, int) else "None"
                    warnings.append(
                        f"RETENTION>100: {bm}/{name} {label}: {v:.1f}%")

    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  All checks passed.")
    print(f"  Total warnings: {len(warnings)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables from result files")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory (default: results/)")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX formatted rows")
    parser.add_argument("--prose", action="store_true",
                        help="Output prose statistics for §5.1/§5.2")
    parser.add_argument("--appendix", action="store_true",
                        help="Output per-model retention details")
    parser.add_argument("--check", action="store_true",
                        help="Run data quality checks")
    parser.add_argument("--scorer", default="llm_judge",
                        choices=["official", "llm_judge"],
                        help="MCQ scoring method (default: llm_judge)")
    parser.add_argument("--all", action="store_true",
                        help="Output everything")
    args = parser.parse_args()

    results_dir = args.results_dir
    scorer = args.scorer
    show_all = args.all

    print(f"[scorer={scorer}]")

    # Compute
    table1 = compute_table1(results_dir, scorer)
    table2, ret_details = compute_table2(results_dir, scorer)
    table3 = compute_table3(results_dir, scorer)

    # Always print markdown tables
    print_table1_md(table1)
    print_table2_md(table2)
    print_table3_md(table3)

    if args.latex or show_all:
        print_table1_latex(table1)
        print_table2_latex(table2)
        print_table3_latex(table3)

    if args.prose or show_all:
        print_prose(table1, table2)

    if args.appendix or show_all:
        print_appendix(ret_details)

    if args.check or show_all:
        print_check(results_dir, table1, ret_details, scorer)


if __name__ == "__main__":
    main()
