"""Field layout of the per-item result files, shared by the strip and rehydrate scripts.

Raw prediction files embed each benchmark's own question text, choices and answers,
which are licensed CC BY-NC 4.0 by their authors. The released archive keeps only the
fields produced by our pipeline; the benchmark fields are re-joined by ``id`` from the
official Hugging Face datasets with ``scripts/rehydrate_results.py``.
"""

import re

BENCHMARKS = ("mmau", "mmar", "mmau_pro")

# Fields written by our inference pipeline (kept in the released archive).
PIPELINE_FIELDS = {
    "mmau": ("id", "prediction"),
    "mmar": ("id", "model_prediction", "model_id", "audio_condition", "audio_source", "error"),
    "mmau_pro": ("id", "prediction"),
}

# Fields copied from the benchmark dataset (removed from the released archive).
BENCHMARK_FIELDS = {
    "mmau": ("question", "choices", "ground_truth", "task", "category", "sub_category",
             "difficulty", "dataset"),
    "mmar": ("question", "choices", "answer", "modality", "category", "sub-category"),
    "mmau_pro": ("category", "question", "choices", "ground_truth", "task_identifier",
                 "kwargs", "prompt_transcription"),
}

# Per-item LLM-judge files contain no benchmark text and are released unchanged.
RESULT_FILE = re.compile(r"^(full|none|n\d+_chunk\d+)(_llm_judge)?\.(jsonl|json)$")


def is_judge_file(name: str) -> bool:
    match = RESULT_FILE.match(name)
    return bool(match and match.group(2))
