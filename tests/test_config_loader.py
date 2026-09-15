import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_utils import load_generation_config

def test_loader():
    print("Testing Config Loader...")

    # Case 1: Standard Model in MMAR
    config_std = load_generation_config("mmar", "Qwen/Qwen2-Audio-7B-Instruct")
    print(f"\n[Standard] MMAR + Qwen2.5:")
    print(f"  max_tokens: {config_std['max_tokens']} (Expected: 512)")
    print(f"  temperature: {config_std['temperature']} (Expected: 0.0)")
    assert config_std['max_tokens'] == 512

    # Case 2: Thinking Model
    config_think = load_generation_config("mmar", "Qwen/Qwen3-Omni-30B-A3B-Thinking")
    print(f"\n[Thinking] MMAR + Qwen3-Thinking:")
    print(f"  max_tokens: {config_think['max_tokens']} (Expected: 4096)")
    print(f"  temperature: {config_think['temperature']} (Expected: 0.6)")
    assert config_think['max_tokens'] == 4096
    
    # Case 3: MMAU Benchmark Check
    config_mmau = load_generation_config("mmau", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    print(f"\n[MMAU] MMAU + Qwen3-Instruct:")
    print(f"  max_tokens: {config_mmau['max_tokens']} (Expected: 128)")
    assert config_mmau['max_tokens'] == 128

    # Case 4: Gemma-3n (should inherit benchmark max_tokens)
    config_gemma_mmar = load_generation_config("mmar", "google/gemma-3n-E4B-it")
    print(f"\n[Gemma3n] MMAR + Gemma-3n:")
    print(f"  max_tokens: {config_gemma_mmar['max_tokens']} (Expected: 512)")
    print(f"  temperature: {config_gemma_mmar['temperature']} (Expected: 0.0)")
    assert config_gemma_mmar['max_tokens'] == 512
    assert config_gemma_mmar['temperature'] == 0.0

    config_gemma_mmau = load_generation_config("mmau", "google/gemma-3n-E4B-it")
    print(f"\n[Gemma3n] MMAU + Gemma-3n:")
    print(f"  max_tokens: {config_gemma_mmau['max_tokens']} (Expected: 128)")
    assert config_gemma_mmau['max_tokens'] == 128

    print("\n✅ All config tests passed!")

if __name__ == "__main__":
    test_loader()
