#!/usr/bin/env python3
"""Generate retention rate figures with Okabe-Ito color scheme.

Outputs:
  - retention_rate_trends_v3.png: Model trends across 3 benchmarks (Okabe-Ito colors)
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json
import os
from collections import OrderedDict

# Set consistent font with Figure 2 (stacked decomposition)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ============================================================
# Configuration (shared with generate_tables.py)
# ============================================================

BENCHMARKS = ["mmau", "mmar", "mmau_pro"]
BM_DISPLAY = {"mmau": "MMAU", "mmar": "MMAR", "mmau_pro": "MMAU-Pro"}
CHUNKS = [2, 3, 4, 5]

MODELS = OrderedDict([
    ("Audio-Flamingo-3", dict(
        dir="audio_flamingo_3",
        tb_dir="tb_qwen2.5_7b_instruct",
        exclude_none=set(),
    )),
    ("DeSTA-2.5", dict(
        dir="desta2.5",
        tb_dir="tb_llama_3.1_8b_instruct",
        exclude_none=set(),
    )),
    ("Phi-4-MM", dict(
        dir="phi4_multimodal",
        tb_dir="tb_phi4_mini_instruct",
        exclude_none=set(),
    )),
    ("Qwen2-Audio", dict(
        dir="qwen2_audio_7b_instruct",
        tb_dir="tb_qwen_7b_chat",
        exclude_none=set(),
    )),
    ("Qwen2.5-Omni", dict(
        dir="qwen2.5_omni_7b",
        tb_dir="tb_qwen2.5_7b_instruct",
        exclude_none=set(),
    )),
    ("Qwen3-Omni (I)", dict(
        dir="qwen3_omni_30b_a3b_instruct",
        tb_dir="tb_qwen3_30b_a3b_instruct",
        exclude_none=set(),
    )),
    ("Qwen3-Omni (T)", dict(
        dir="qwen3_omni_30b_a3b_thinking",
        tb_dir="tb_qwen3_30b_a3b_thinking",
        exclude_none=set(),
    )),
    ("Voxtral-Mini", dict(
        dir="voxtral_mini_3b",
        tb_dir="tb_ministral_3b",
        exclude_none=set(),
    )),
])

# Qualitative color set for model tracing (user-provided + safe fallbacks).
COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#fc8d62",
    "#1f78b4",
    "#e6ab02",
]

# Line styles for visual differentiation
LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot',
              'solid', 'dashed', 'dotted', 'dashdot']

# ============================================================
# File I/O
# ============================================================

def result_path(results_dir: str, bm: str, model_dir: str,
                condition: str, scorer: str = "llm_judge") -> str:
    if scorer == "llm_judge":
        suffix = "_llm_judge_results.json"
    elif bm == "mmau_pro":
        suffix = "_comprehensive_results.json"
    else:
        suffix = "_results.json"
    return os.path.join(results_dir, bm, model_dir, f"{condition}{suffix}")


def read_result(filepath: str, bm: str = None,
                scorer: str = "llm_judge") -> dict | None:
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

    if bm == "mmau_pro":
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
            "accuracy": mcq_correct / mcq_total if mcq_total else None,
            "correct": mcq_correct,
            "total": mcq_total,
            "no_pred": 0,
            "category_accuracy": cat_acc,
        }

    if bm in ("mmau", "mmar"):
        cat_key = "task_accuracy" if bm == "mmau" else "modality_accuracy"
        cat_acc = data.get(cat_key, {})
        total = data.get("total", 0)
        correct = data.get("correct", 0)
        return {
            "accuracy": correct / total if total else None,
            "correct": correct,
            "total": total,
            "no_pred": 0,
            "category_accuracy": cat_acc,
        }

    return None


def compute_retention(results_dir: str, scorer: str = "llm_judge") -> dict:
    """Compute retention rates for all models and benchmarks.

    Returns: {bm: {model_name: {2: val, 3: val, 4: val, 5: val, none: val}}}
    """
    details = {}

    for bm in BENCHMARKS:
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

            if none_acc is not None:
                ret["none"] = (none_acc / full_acc) * 100
            else:
                ret["none"] = None

            model_ret[name] = ret

        details[bm] = model_ret

    return details


# ============================================================
# Visualization: Trends (3 subplots, 8 models, unified legend)
# ============================================================

def plot_retention_trends(details, output_path="docs/paper/figures/retention_rate_trends.png"):
    """Plot retention rate trends across benchmarks (2×2 grid layout)."""

    # Layout: 2×2 grid (3 plots + legend panel) - compact spacing
    fig = plt.figure(figsize=(3.6, 3.6))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15,
                         height_ratios=[1, 1], width_ratios=[1, 1])
    ax_mmau = fig.add_subplot(gs[0, 0])    # Top left (MMAU)
    ax_mmar = fig.add_subplot(gs[0, 1])    # Top right (MMAR)
    ax_pro = fig.add_subplot(gs[1, 0])     # Bottom left (MMAU-Pro)
    ax_legend = fig.add_subplot(gs[1, 1])  # Bottom right (legend)
    ax_legend.axis('off')

    axes_plots = [ax_mmau, ax_mmar, ax_pro]

    # X positions for N=2,3,4,5,None
    x_labels = ['2', '3', '4', '5', 'None']
    x_pos = [0, 1, 2, 3, 4]

    # Collect all line objects for legend
    legend_lines = []
    legend_labels = []

    for bm_idx, bm in enumerate(BENCHMARKS):
        ax = axes_plots[bm_idx]

        # Get data for this benchmark
        model_data = details[bm]

        for model_idx, (model_name, info) in enumerate(MODELS.items()):
            if model_name not in model_data:
                continue

            ret = model_data[model_name]
            y_values = [ret.get(n) for n in CHUNKS] + [ret.get("none")]

            # Skip if all values are None
            if all(v is None for v in y_values):
                continue

            # Plot line
            color = COLORS[model_idx]
            linestyle = LINESTYLES[model_idx]

            ax.plot(x_pos, y_values, marker='o', label=model_name,
                   color=color, linestyle=linestyle, linewidth=1.4, markersize=4.0,
                   zorder=2)

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylim([40, 105])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3, zorder=0)
        ax.tick_params(labelsize=7)

        # Benchmark name (top-left corner, small)
        ax.text(0.02, 0.98, BM_DISPLAY[bm], transform=ax.transAxes,
                ha='left', va='top', fontsize=8, fontweight='bold')

        # Y-axis label only on left column
        if bm_idx in [0, 2]:
            ax.set_ylabel('Retention (%)', fontsize=7.5)
        else:
            ax.set_yticklabels([])

        # X-axis label on bottom row only
        if bm_idx == 2:
            ax.set_xlabel('Number of Fragments', fontsize=7.5)

    # Collect legend entries (only once)
    for model_idx, (model_name, info) in enumerate(MODELS.items()):
        model_data = details[BENCHMARKS[0]]  # Get from first benchmark
        if model_name in model_data and model_data[model_name]:
            color = COLORS[model_idx]
            linestyle = LINESTYLES[model_idx]
            line = mlines.Line2D([], [], color=color, linestyle=linestyle,
                                linewidth=1.4, marker='o', markersize=4.0,
                                label=model_name)
            legend_lines.append(line)
            legend_labels.append(model_name)

    # Create legend in bottom-right panel (1 column × 8 rows)
    # Add padding to prevent overlap with bottom-left plot
    ax_legend.legend(legend_lines, legend_labels, ncol=1, loc='center',
                    fontsize=8.5, frameon=True, fancybox=True, framealpha=0.8,
                    edgecolor='gray', facecolor='#f5f5f5', handlelength=1.2,
                    columnspacing=0.6, labelspacing=0.22, borderpad=0.65,
                    markerscale=1.1)

    # Ensure no overlap between subplots and legend
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved {output_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    results_dir = "results"

    print("Computing retention rates...")
    details = compute_retention(results_dir)

    print("Generating trends figure...")
    plot_retention_trends(details)


if __name__ == "__main__":
    main()
