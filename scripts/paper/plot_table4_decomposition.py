#!/usr/bin/env python3
"""Create a publication-style bar chart from decomposition_summary.json.

Reads model-averaged decomposition data computed by
scripts/analysis/compute_decomposition.py (default: results/decomposition_summary.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


DEFAULT_SUMMARY_PATH = Path("results/decomposition_summary.json")

CATEGORIES = ["TS", "FS", "XS", "AH", "UN"]
PALETTE = {
    "TS": "#8dd3c7",
    "FS": "#ffffb3",
    "XS": "#bebada",
    "AH": "#fb8072",
    "UN": "#80b1d3",
}
LEGEND_LABELS = {
    "TS": "|TS|/|Q|",
    "FS": "|FS|/|Q|",
    "XS": "|XS|/|Q|",
    "AH": "|AH|/|Q|",
    "UN": "|UN|/|Q|",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a bar chart for Table 4 per-item decomposition."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=(
            "JSON file with rows containing benchmark, TS, FS, XS, AH, UN, "
            "XS_AN, XS_AN_LOW, XS_AN_HIGH "
            f"(default: {DEFAULT_SUMMARY_PATH})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/paper/figures"),
        help="Output directory (default: docs/paper/figures)",
    )
    parser.add_argument(
        "--filename",
        default="audio_reliance_breakdown",
        help="Base filename without extension (default: audio_reliance_breakdown)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display preview window after saving files.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run compute_decomposition.py first:\n"
            f"  python scripts/analysis/compute_decomposition.py"
        )

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    required_keys = {"benchmark", "TS", "FS", "XS", "AH", "UN", "XS_AN"}
    normalized = []
    for row in rows:
        if not required_keys.issubset(row):
            missing = ", ".join(sorted(required_keys - row.keys()))
            raise ValueError(f"Row for {row.get('benchmark', '<unknown>')} missing: {missing}")
        row["XS_AN_LOW"] = row.get("XS_AN_LOW", row["XS_AN"])
        row["XS_AN_HIGH"] = row.get("XS_AN_HIGH", row["XS_AN"])
        normalized.append(
            {
                "benchmark": row["benchmark"],
                "TS": float(row["TS"]),
                "FS": float(row["FS"]),
                "XS": float(row["XS"]),
                "AH": float(row["AH"]),
                "UN": float(row["UN"]),
                "XS_AN": float(row["XS_AN"]),
                "XS_AN_LOW": float(row["XS_AN_LOW"]),
                "XS_AN_HIGH": float(row["XS_AN_HIGH"]),
            }
        )
    return normalized


def dark_text_color(color_hex: str) -> str:
    rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#ffffff" if luminance < 140 else "#222222"


def plot_table4(rows: list[dict], output_dir: Path, filename: str, show: bool) -> None:
    benchmarks = [row["benchmark"] for row in rows]
    y = np.arange(len(benchmarks))
    bar_height = 0.5

    matrix = np.array([[row[cat] for cat in CATEGORIES] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavusans",
            "grid.alpha": 0.25,
            "axes.grid": True,
            "grid.linestyle": "--",
            "figure.dpi": 400,
            "savefig.dpi": 1200,
        }
    )

    fig, ax_dist = plt.subplots(figsize=(8.5, 3.0))

    left = np.zeros(len(rows))
    for idx, cat in enumerate(CATEGORIES):
        vals = matrix[:, idx]
        bars = ax_dist.barh(
            y,
            vals,
            height=bar_height,
            left=left,
            color=PALETTE[cat],
            edgecolor="white",
            linewidth=0.8,
            label=LEGEND_LABELS[cat],
        )

        for bar, val in zip(bars, vals):
            if val >= 3.0:
                x = bar.get_x() + bar.get_width() / 2
                ax_dist.text(
                    x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=dark_text_color(PALETTE[cat]),
                    fontweight="bold",
                )
        left += vals

    ax_dist.set_yticks(y)
    ax_dist.set_yticklabels(benchmarks, ha="right")
    ax_dist.tick_params(axis="y", pad=6)
    ax_dist.set_xlabel("Model-averaged share of items (%)", labelpad=12)
    ax_dist.set_ylabel("Benchmark")
    ax_dist.set_xlim(0, 105)
    ax_dist.axvline(0, color="#444444", linewidth=0.6, zorder=2)
    ax_dist.set_axisbelow(True)

    # Place legend in figure space to keep it below the plot and visible.
    handles, labels = ax_dist.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=11,
        frameon=False,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.32)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path_png = output_dir / f"{filename}.png"
    fig.savefig(fig_path_png, dpi=1200)
    print(f"Saved: {fig_path_png}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_json)
    plot_table4(rows, args.output_dir, args.filename, show=args.show)


if __name__ == "__main__":
    main()
