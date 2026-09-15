# Third-party resources

## Benchmarks

All three benchmarks are distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
license. The released per-item predictions exclude their question text, choices and answers;
`scripts/rehydrate_results.py` retrieves those fields from the datasets below. Per-item judge files
(`*_llm_judge.jsonl`) record the answer option letter for each item.

| Benchmark | Dataset | License |
|---|---|---|
| MMAU | [`gamma-lab-umd/MMAU-test-mini`](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | CC BY-NC 4.0 |
| MMAR | [`BoJack/MMAR`](https://huggingface.co/datasets/BoJack/MMAR) | CC BY-NC 4.0 |
| MMAU-Pro | [`gamma-lab-umd/MMAU-Pro`](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro) | CC BY-NC 4.0 |

Use of results derived from these benchmarks is subject to their license terms. Please cite MMAU, MMAR and
MMAU-Pro when using them.

## Evaluated models

Model outputs in the released predictions are subject to each model's own license; see the linked model
cards. Identifiers are taken from the `model_id` field of the MMAR result files; the MMAU and MMAU-Pro
result files do not record one.

| Model | Result directory | Hugging Face model |
|---|---|---|
| Audio-Flamingo-3 | `audio_flamingo_3` | [`nvidia/audio-flamingo-3-hf`](https://huggingface.co/nvidia/audio-flamingo-3-hf) |
| DeSTA-2.5 | `desta2.5` | [`DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B`](https://huggingface.co/DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B) |
| Phi-4-Multimodal | `phi4_multimodal` | [`microsoft/Phi-4-multimodal-instruct`](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |
| Qwen2-Audio-7B | `qwen2_audio_7b_instruct` | [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| Qwen2.5-Omni-7B | `qwen2.5_omni_7b` | [`Qwen/Qwen2.5-Omni-7B`](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |
| Qwen3-Omni (Instruct) | `qwen3_omni_30b_a3b_instruct` | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Qwen3-Omni (Thinking) | `qwen3_omni_30b_a3b_thinking` | [`Qwen/Qwen3-Omni-30B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) |
| Voxtral-Mini-3B | `voxtral_mini_3b` | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) |

| Text backbone run | Result directory | Hugging Face model |
|---|---|---|
| Qwen2.5-7B-Instruct | `tb_qwen2.5_7b_instruct` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama-3.1-8B-Instruct | `tb_llama_3.1_8b_instruct` | [`DeSTA-ntu/Llama-3.1-8B-Instruct`](https://huggingface.co/DeSTA-ntu/Llama-3.1-8B-Instruct) |
| Phi-4-Mini-Instruct | `tb_phi4_mini_instruct` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Qwen-7B-Chat | `tb_qwen_7b_chat` | [`Qwen/Qwen-7B-Chat`](https://huggingface.co/Qwen/Qwen-7B-Chat) |
| Qwen3-30B-A3B-Instruct | `tb_qwen3_30b_a3b_instruct` | [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| Qwen3-30B-A3B-Thinking | `tb_qwen3_30b_a3b_thinking` | [`Qwen/Qwen3-30B-A3B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) |
| Ministral-3B | `tb_ministral_3b` | [`ministral/Ministral-3b-instruct`](https://huggingface.co/ministral/Ministral-3b-instruct) |

## Evaluation models

| Use | Model |
|---|---|
| MCQ answer extraction (LLM judge) | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) via the Anthropic API |
| MMAU-Pro closed-ended matching | [`nvidia/NV-Embed-v2`](https://huggingface.co/nvidia/NV-Embed-v2) |
| MMAU-Pro open-ended judge | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

## Software

[DeSTA2.5-Audio](https://github.com/kehanlu/DeSTA2.5-Audio) is required to run DeSTA-2.5 and is not
redistributed here; see [docs/environments.md](docs/environments.md).
