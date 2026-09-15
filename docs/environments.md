# Environments

The models need mutually incompatible library versions, so inference was run in one environment per model
family. Library versions were not recorded alongside the results; the notes below list what each runner
imports and the constraints we know of.

## Analysis, scoring and rehydration

`requirements.txt` covers stages 1–3 of [reproduction.md](reproduction.md) except the MMAU-Pro official
evaluation:

| Purpose | Packages |
|---|---|
| Tables and figures | `numpy`, `matplotlib` |
| Rehydrating predictions | `huggingface_hub`, `pyarrow` |
| MCQ judge | `anthropic` |

The MMAU-Pro official evaluation (`scripts/mmau_pro/evaluate.py`) additionally needs `torch`,
`transformers`, `pandas`, `scikit-learn`, `nltk` and `tqdm`, and loads `nvidia/NV-Embed-v2` and
`Qwen/Qwen2.5-7B-Instruct`. It patches NV-Embed-v2 to run with `transformers>=4.50`.

## Inference

All benchmark runners import `datasets`, `numpy`, `tqdm` and `pyyaml`, and read audio with `librosa`
and/or `soundfile`.

| Model family | Runners | Additional packages |
|---|---|---|
| Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni | `run_inference.py`, `run_inference_vllm.py`, `run_qwen.py` | `vllm`, `transformers` |
| Audio-Flamingo-3 | `run_flamingo.py`, `run_flamingo_mmar.py` | `vllm` (built from source), `transformers>=5.0.0rc1` |
| DeSTA-2.5 | `run_desta.py`, `run_desta_mmar.py` | `desta`, `torch` |
| Phi-4-Multimodal | `run_phi4_hf.py`, `run_phi4_hf_mmar.py` | `transformers`, `torch` |
| Voxtral-Mini-3B | `run_voxtral.py`, `run_voxtral_inference.py` | `vllm`, `mistral_common` |
| Text backbones | `scripts/run_text_backbone.py`, `scripts/retry_truncated.py` | `vllm`, `transformers` |

### DeSTA-2.5

The `desta` package comes from [kehanlu/DeSTA2.5-Audio](https://github.com/kehanlu/DeSTA2.5-Audio) and is
not included here. Its `desta/utils/audio.py` uses `np.sctypes`, which NumPy 2.0 removed. We ran it with
lines 227 and 230 changed to:

```python
if np.issubdtype(samples.dtype, np.integer):
    ...
elif np.issubdtype(samples.dtype, np.floating):
```

and with its `transformers>=4.49.0` requirement relaxed to `transformers>=4.38.0`.
