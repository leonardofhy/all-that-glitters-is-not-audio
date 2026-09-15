#!/usr/bin/env python3
"""
Build the released per-item prediction archive from a full results tree.

Keeps only pipeline-produced fields in raw prediction files and copies per-item
LLM-judge files unchanged. Records whose schema contains a field that is in neither
the pipeline nor the benchmark list abort the run, so nothing is dropped silently.

Usage:
    python scripts/release/strip_benchmark_fields.py --src results --dst release_archive/results
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.release.benchmark_fields import (  # noqa: E402
    BENCHMARKS, BENCHMARK_FIELDS, PIPELINE_FIELDS, RESULT_FILE, is_judge_file,
)

# Superseded runs that are not part of the published results.
EXCLUDED_MODEL_DIRS = {"voxtral_mini_3b_legacy_hf"}
# Duplicate key written by one runner version (Qwen3-Omni-Thinking on MMAU); equal to sub_category.
KNOWN_ALIASES = {"mmau": {"sub-category"}}


def read_records(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_records(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_record(record: dict, benchmark: str, path: Path) -> dict:
    known = set(PIPELINE_FIELDS[benchmark]) | set(BENCHMARK_FIELDS[benchmark])
    unknown = set(record) - known - KNOWN_ALIASES.get(benchmark, set())
    if unknown:
        raise ValueError(f"{path}: unclassified fields {sorted(unknown)}")
    return {k: record[k] for k in PIPELINE_FIELDS[benchmark] if k in record}


def iter_result_files(src: Path, benchmark: str):
    for model_dir in sorted(p for p in (src / benchmark).iterdir() if p.is_dir()):
        if model_dir.name in EXCLUDED_MODEL_DIRS or model_dir.name.startswith("."):
            continue
        names = {p.name for p in model_dir.iterdir() if p.is_file()}
        for name in sorted(names):
            match = RESULT_FILE.match(name)
            if not match:
                continue
            stem, judge, ext = match.groups()
            # MMAR results are canonical as .json; some runs also left a .jsonl copy.
            if benchmark == "mmar" and not judge and ext == "jsonl" and f"{stem}.json" in names:
                continue
            yield model_dir / name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Full results root (contains mmau/, mmar/, mmau_pro/)")
    parser.add_argument("--dst", type=Path, required=True, help="Output results root for the archive")
    args = parser.parse_args()

    n_files = n_records = 0
    for benchmark in BENCHMARKS:
        for path in iter_result_files(args.src, benchmark):
            out = args.dst / path.relative_to(args.src)
            if is_judge_file(path.name):
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out)
            else:
                records = read_records(path)
                write_records(out, [strip_record(r, benchmark, path) for r in records])
                n_records += len(records)
            n_files += 1
    print(f"Wrote {n_files} files ({n_records} stripped prediction records) to {args.dst}")


if __name__ == "__main__":
    main()
