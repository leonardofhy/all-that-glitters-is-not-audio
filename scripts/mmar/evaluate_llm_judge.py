#!/usr/bin/env python3
"""MMAR LLM MCQ Judge — Claude Haiku answer extraction + correctness judgment.

Thin wrapper around scripts/analysis/llm_mcq_judge.py for the MMAR benchmark.
Produces per-sample JSONL and aggregate JSON alongside inference files.

Usage:
    # Single file
    python scripts/mmar/evaluate_llm_judge.py results/mmar/voxtral_mini_3b/full.json

    # Batch (all models, full+none)
    python scripts/mmar/evaluate_llm_judge.py --batch
    python scripts/mmar/evaluate_llm_judge.py --batch --condition full,none

    # Dry run
    python scripts/mmar/evaluate_llm_judge.py --batch --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analysis.llm_mcq_judge import process_file, run_batch

BENCHMARK = "mmar"


def main():
    parser = argparse.ArgumentParser(
        description=f"LLM MCQ Judge — {BENCHMARK.upper()}",
    )
    parser.add_argument("input_file", nargs="?", default=None,
                        help="Path to inference JSON file")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: process all models")
    parser.add_argument("--model", default="all",
                        help="Model filter (short name, or all)")
    parser.add_argument("--condition", default="full,none",
                        help="Condition filter (full, none, full,none, all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if judge output exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed")
    args = parser.parse_args()

    if args.batch or args.dry_run:
        run_batch(
            benchmark_filter=BENCHMARK,
            model_filter=args.model,
            condition_filter=args.condition,
            dry_run=args.dry_run,
            force=args.force,
        )
    elif args.input_file:
        process_file(Path(args.input_file), benchmark=BENCHMARK, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
