"""Shared constants for analysis scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_REGISTRY = {
    "qwen2_audio": {
        "model_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "dir": "qwen2_audio_7b_instruct",
        "params": "7B",  # Qwen2-7B LLM + Whisper-large-v2 encoder
    },
    "qwen2.5_omni": {
        "model_id": "Qwen/Qwen2.5-Omni-7B",
        "dir": "qwen2.5_omni_7b",
        "params": "7B",
    },
    "qwen3_instruct": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "dir": "qwen3_omni_30b_a3b_instruct",
        "params": "30B (A3B)",  # MoE: 30B total, 3B active
    },
    "qwen3_thinking": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "dir": "qwen3_omni_30b_a3b_thinking",
        "params": "30B (A3B)",  # MoE: 30B total, 3B active
    },
    "flamingo": {
        "model_id": "nvidia/audio-flamingo-3-hf",
        "dir": "audio_flamingo_3",
        "params": "8.4B",  # Qwen2.5-7B backbone + AF-Whisper encoder + adapter
    },
    "desta2.5": {
        "model_id": "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        "dir": "desta2.5",
        "params": "8.8B",  # Llama-3.1-8B + Whisper-large-v3 + Q-Former (131M trainable)
    },
    "phi4": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "dir": "phi4_multimodal",
        "params": "5.6B",  # HF model card: Phi-4-Mini backbone + vision/speech encoders
    },
    "voxtral": {
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "dir": "voxtral_mini_3b",
        "params": "4.7B",  # Ministral-3B decoder + Whisper-large-v3 encoder + adapter
    },
}

BENCHMARK_CONFIG = {
    "mmau": {
        "expected_samples": 1000,
        "ext": ".jsonl",
        "eval_suffix": "_results.json",
        "relaxed_eval_suffix": "_relaxed_results.json",
        "chunk_ns": [2, 3, 4, 5],
    },
    "mmau_pro": {
        "expected_samples": 5305,
        "ext": ".jsonl",
        "eval_suffix": "_comprehensive_results.json",
        "relaxed_eval_suffix": "_relaxed_results.json",
        "chunk_ns": [2, 3, 4, 5],
    },
    "mmar": {
        "expected_samples": 1000,
        "ext": ".json",
        "eval_suffix": "_results.json",
        "relaxed_eval_suffix": "_relaxed_results.json",
        "chunk_ns": [2, 3, 4, 5],
    },
}

# Model×benchmark combinations that don't exist
# Note: all 8 models have results on all 3 benchmarks
EXCLUDED = set()

# Anomaly thresholds
RANDOM_BASELINE = 0.25  # 4-choice MCQ
NO_PREDICTION_THRESHOLD = 0.05  # 5% of total samples
CHUNK_POSITION_VARIANCE_THRESHOLD = 0.05  # 5pp between chunks at same N
CROSS_BENCHMARK_RANK_THRESHOLD = 3  # rank difference > 3 positions
EMPTY_PREDICTION_THRESHOLD = 0.01  # 1% empty predictions
TRUNCATION_THRESHOLD = 0.10  # 10% truncated predictions


def get_model_dir(benchmark: str, model_short: str) -> Path:
    """Get the results directory for a model×benchmark combination."""
    return RESULTS_DIR / benchmark / MODEL_REGISTRY[model_short]["dir"]


def iter_model_benchmark_pairs(benchmark_filter: str = "all", model_filter: str = "all"):
    """Iterate over valid (model_short, benchmark) pairs, respecting filters and exclusions."""
    benchmarks = list(BENCHMARK_CONFIG.keys()) if benchmark_filter == "all" else [benchmark_filter]
    models = list(MODEL_REGISTRY.keys()) if model_filter == "all" else [model_filter]
    for model in models:
        for bench in benchmarks:
            if (model, bench) not in EXCLUDED:
                yield model, bench


def get_expected_conditions(benchmark: str):
    """Return list of expected condition names for a benchmark."""
    cfg = BENCHMARK_CONFIG[benchmark]
    conditions = ["full", "none"]
    for n in cfg["chunk_ns"]:
        for k in range(n):
            conditions.append(f"n{n}_chunk{k}")
    return conditions


def get_chunk_conditions(benchmark: str):
    """Return only chunk condition names (excluding full/none)."""
    return get_expected_conditions(benchmark)[2:]


# ── Display name mappings (unified across all benchmarks) ───────────────────
# All benchmarks use bold human-readable names for consistency.

_UNIFIED_DISPLAY = {
    "qwen2_audio": "**Qwen2-Audio-7B**",
    "qwen2.5_omni": "**Qwen2.5-Omni-7B**",
    "qwen3_instruct": "**Qwen3-Instruct**",
    "qwen3_thinking": "**Qwen3-Thinking**",
    "flamingo": "**Audio-Flamingo-3**",
    "desta2.5": "**DeSTA-2.5**",
    "phi4": "**Phi-4-Multimodal**",
    "voxtral": "**Voxtral-Mini-3B**",
}

DISPLAY_NAMES = {
    "mmau": _UNIFIED_DISPLAY,
    "mmar": _UNIFIED_DISPLAY,
    "mmau_pro": _UNIFIED_DISPLAY,
}

# Model ordering per benchmark doc (as they appear in tables)
MODEL_ORDER = {
    "mmau": [  # alphabetical by dir name
        "flamingo", "desta2.5", "phi4", "qwen2.5_omni",
        "qwen2_audio", "qwen3_instruct", "qwen3_thinking", "voxtral",
    ],
    "mmar": [  # ranked by Full accuracy descending (fixed order)
        "qwen3_instruct", "qwen3_thinking", "qwen2.5_omni",
        "flamingo", "voxtral", "phi4", "desta2.5", "qwen2_audio",
    ],
    "mmau_pro": [  # ranked by Full Overall descending (fixed order)
        "phi4", "qwen3_thinking", "voxtral", "qwen3_instruct",
        "desta2.5", "qwen2_audio", "flamingo", "qwen2.5_omni",
    ],
}

# MMAU sub-metric keys (matching task_accuracy keys in _results.json)
MMAU_TASKS = ["sound", "music", "speech"]

# MMAR sub-metric keys (matching category_accuracy keys in _results.json)
MMAR_LAYERS = ["Signal Layer", "Perception Layer", "Semantic Layer", "Cultural Layer"]

# Model colors for charts (matplotlib tab10, colorblind-friendly)
MODEL_COLORS = {
    "qwen3_instruct": "#1f77b4",   # Blue
    "qwen3_thinking": "#ff7f0e",   # Orange
    "qwen2.5_omni":   "#2ca02c",   # Green
    "qwen2_audio":    "#d62728",   # Red
    "flamingo":       "#9467bd",   # Purple
    "desta2.5":          "#8c564b",   # Brown
    "phi4":           "#e377c2",   # Pink
    "voxtral":        "#7f7f7f",   # Gray
}

# Doc file paths
DOC_PATHS = {
    "mmau": PROJECT_ROOT / "docs" / "mmau_benchmark_overview.md",
    "mmar": PROJECT_ROOT / "docs" / "mmar_benchmark_overview.md",
    "mmau_pro": PROJECT_ROOT / "docs" / "mmau_pro_benchmark_overview.md",
}
