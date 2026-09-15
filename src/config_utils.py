import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "generation_params.yaml"

def load_generation_config(benchmark: str, model_id: str) -> Dict[str, Any]:
    """
    Load generation parameters based on benchmark and model ID.
    Priority: Model Type > Benchmark > Default
    """
    if not CONFIG_PATH.exists():
        # Fallback if config missing
        return {"max_tokens": 128, "temperature": 0.0}

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Start with defaults
    params = config.get('default', {}).copy()

    # 2. Apply Benchmark overrides
    if benchmark in config.get('benchmarks', {}):
        params.update(config['benchmarks'][benchmark])

    # 3. Apply Model Type overrides (only 'thinking' models currently)
    model_types = config.get('model_types', {})
    model_id_lower = model_id.lower()

    if "thinking" in model_id_lower or "deepseek-r1" in model_id_lower:
        params.update(model_types.get('thinking', {}))

    return params
