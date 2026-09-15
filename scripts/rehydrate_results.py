#!/usr/bin/env python3
"""
Restore benchmark fields into the released per-item prediction files.

The released archive omits each benchmark's question text, choices and answers
(CC BY-NC 4.0). This script re-joins them by ``id`` from the official Hugging Face
datasets, reproducing the fields exactly as the inference scripts wrote them, so the
output can be fed to ``scripts/*/evaluate_llm_judge.py`` and ``scripts/mmau_pro/evaluate.py``.

Only metadata is downloaded (MMAR-meta.json, MMAU-Pro test.parquet, and the text
columns of MMAU test-mini via parquet column projection); no audio is fetched.

Requirements: huggingface_hub, pyarrow, fsspec

Usage:
    python scripts/rehydrate_results.py --src release_archive/results --dst results_full
"""

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.release.benchmark_fields import BENCHMARKS, RESULT_FILE, is_judge_file  # noqa: E402


# ============================================================
# Benchmark metadata loaders (mirror the inference scripts)
# ============================================================

def load_mmau_mini():
    """Mirror scripts/mmau/utils.py build_mmau_result over gamma-lab-umd/MMAU-test-mini."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    columns = ["instruction", "choices", "answer", "other_attributes"]
    with HfFileSystem().open("datasets/gamma-lab-umd/MMAU-test-mini/test_mini.parquet", "rb") as f:
        rows = pq.ParquetFile(f).read(columns=columns).to_pylist()

    items = {}
    for index, row in enumerate(rows):
        attrs = _parse_other_attributes(row.get("other_attributes"))
        item_id = attrs.get("id", row.get("id", str(index)))
        items[item_id] = {
            "task": attrs.get("task", ""),
            "category": attrs.get("category", ""),
            "sub_category": attrs.get("sub-category", ""),
            "difficulty": attrs.get("difficulty", ""),
            "dataset": attrs.get("dataset", ""),
            "question": row.get("instruction", ""),
            "choices": row.get("choices", []),
            "ground_truth": row.get("answer", ""),
        }
    return items


def _parse_other_attributes(attrs):
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (json.JSONDecodeError, TypeError):
            try:
                attrs = ast.literal_eval(attrs)
            except (ValueError, SyntaxError):
                attrs = {}
    return attrs if isinstance(attrs, dict) else {}


def load_mmar():
    """Mirror scripts/mmar/run_inference_vllm.py record construction over BoJack/MMAR."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("BoJack/MMAR", "MMAR-meta.json", repo_type="dataset")
    items = {}
    for sample in json.loads(Path(path).read_text()):
        items[sample["id"]] = {
            "question": _first_text(sample),
            "choices": _normalize_choices(sample.get("choices") or sample.get("options")),
            "answer": sample.get("answer", ""),
            "modality": sample.get("modality", "unknown"),
            "category": sample.get("category", "unknown"),
            "sub-category": sample.get("sub-category") or sample.get("sub_category"),
        }
    return items


def _first_text(sample):
    for key in ["question", "query", "prompt", "instruction", "text"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


def _normalize_choices(choices):
    if choices is None:
        return []
    if isinstance(choices, dict):
        items = list(choices.items())
        try:
            items = sorted(items, key=lambda x: x[0])
        except Exception:
            pass
        return [str(v) for _, v in items]
    if isinstance(choices, (list, tuple)):
        return [str(c) for c in choices]
    return [str(choices)]


def load_mmau_pro():
    """Mirror scripts/mmau_pro/utils.py build_result_dict over gamma-lab-umd/MMAU-Pro."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("gamma-lab-umd/MMAU-Pro", "test.parquet", repo_type="dataset")
    items = {}
    for sample in pq.read_table(path).to_pylist():
        items[sample["id"]] = {
            "category": sample.get("category", ""),
            "question": sample.get("question", ""),
            "choices": sample.get("choices", []),
            "ground_truth": sample.get("answer", ""),
            "task_identifier": sample.get("task_identifier"),
            "kwargs": sample.get("kwargs"),
            # The inference scripts read this key, which the dataset does not define.
            "prompt_transcription": sample.get("prompt_transcription"),
        }
    return items


LOADERS = {"mmau": load_mmau_mini, "mmar": load_mmar, "mmau_pro": load_mmau_pro}


# ============================================================
# Rehydration
# ============================================================

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


def rehydrate_record(record: dict, items: dict, path: Path) -> dict:
    item = items.get(record["id"])
    if item is None:
        raise KeyError(f"{path}: id {record['id']!r} not found in the benchmark metadata")
    return {"id": record["id"], **item, **{k: v for k, v in record.items() if k != "id"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Released archive results root")
    parser.add_argument("--dst", type=Path, required=True, help="Output results root")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="Comma-separated subset")
    args = parser.parse_args()

    for benchmark in args.benchmarks.split(","):
        bench_dir = args.src / benchmark
        if not bench_dir.is_dir():
            print(f"[skip] {bench_dir} not found")
            continue
        items = LOADERS[benchmark]()
        print(f"[{benchmark}] loaded {len(items)} benchmark items")
        n_files = 0
        for path in sorted(bench_dir.rglob("*")):
            if not path.is_file() or not RESULT_FILE.match(path.name):
                continue
            out = args.dst / path.relative_to(args.src)
            if is_judge_file(path.name):
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out)
            else:
                write_records(out, [rehydrate_record(r, items, path) for r in read_records(path)])
            n_files += 1
        print(f"[{benchmark}] wrote {n_files} files")


if __name__ == "__main__":
    main()
