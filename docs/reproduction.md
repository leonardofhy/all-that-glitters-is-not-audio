# Reproduction

The pipeline has four stages. Each stage's outputs are released, so you can start from whichever stage
you need.

| Stage | Starts from | Produces | Requires |
|---|---|---|---|
| 1. Tables and figures | `results/` in this repository | paper tables and figures | CPU, `numpy`, `matplotlib` |
| 2. Decomposition | released predictions archive | `results/decomposition_summary.json` | CPU |
| 3. Scoring | rehydrated predictions | `*_llm_judge*.json(l)`, `*_comprehensive_results.json` | Anthropic API key; GPU for MMAU-Pro official evaluation |
| 4. Inference | benchmark audio | raw predictions | GPUs and per-model environments ([environments.md](environments.md)) |

All commands are run from the repository root.

## 1. Tables and figures

```bash
python scripts/paper/generate_tables.py --latex          # tab:text_prior
python scripts/paper/generate_domain_table.py --latex    # tab:domain
python scripts/paper/generate_retention_figures.py       # fig:retention_trends
python scripts/paper/plot_table4_decomposition.py        # fig:audio_reliance_breakdown
```

`tab:audio_needed` is the Audio-Needed column (FS + XS), the XS/AN mean, and the XS/AN range across models in
`results/decomposition_summary.json`; `compute_decomposition.py` prints it as "Model-Averaged Decomposition".

## 2. Decomposition from per-item predictions

Download the archive, check it, and extract it into a separate directory:

```bash
sha256sum -c SHA256SUMS                 # predictions.tar.gz: OK
mkdir -p release_archive && tar xzf predictions.tar.gz -C release_archive
python scripts/analysis/compute_decomposition.py --results_dir release_archive/results
cmp release_archive/results/decomposition_summary.json results/decomposition_summary.json
```

The decomposition uses only the per-item judge decisions (`*_llm_judge.jsonl`) and the raw predictions, so it
runs on the archive as distributed.

## 3. Scoring

### Restore benchmark fields

The archive omits each benchmark's question text, choices and answers. Restore them before scoring
(requires `huggingface_hub` and `pyarrow`; only text metadata is downloaded, no audio):

```bash
python scripts/rehydrate_results.py --src release_archive/results --dst results
```

Writing to `results/` places the per-item files next to the aggregate scores, where the scoring scripts
expect them. They are ignored by git.

### MCQ judge

The MCQ judge calls the Anthropic API (`claude-haiku-4-5-20251001`). Set `ANTHROPIC_API_KEY` in the
environment or in a `.env` file at the repository root. Existing judge files are skipped unless `--force`
is given.

```bash
python scripts/mmau/evaluate_llm_judge.py --batch --dry-run                         # list files
python scripts/mmau/evaluate_llm_judge.py results/mmau/voxtral_mini_3b/full.jsonl --force
python scripts/mmar/evaluate_llm_judge.py --batch --condition full,none --force
python scripts/mmau_pro/evaluate_llm_judge.py --batch --force
```

Re-running the judge sends every prediction that regular-expression extraction cannot resolve to the API,
so decisions for those items can differ from the released ones.

### MMAU-Pro official evaluation

Open-ended and instruction-following items are scored with the benchmark's evaluation, which loads
NV-Embed and a Qwen2.5-7B-Instruct judge on a GPU:

```bash
python scripts/mmau_pro/evaluate.py results/mmau_pro/voxtral_mini_3b/full.jsonl
python scripts/mmau_pro/evaluate_batch.py results/mmau_pro/*/n*.jsonl   # same output, batched
```

## 4. Inference

### Benchmark data

| Benchmark | Hugging Face dataset | Split used | Audio |
|---|---|---|---|
| MMAU | `gamma-lab-umd/MMAU-test-mini` | 1,000 items | embedded in the parquet file (1.2 GB) |
| MMAR | `BoJack/MMAR` | 1,000 items | `mmar-audio.tar.gz` (2.8 GB) |
| MMAU-Pro | `gamma-lab-umd/MMAU-Pro` | 5,305 items | `data.zip` (45 GB) |

The runners download and cache the data on first use.

### Runners

| Model | MMAU | MMAR | MMAU-Pro |
|---|---|---|---|
| Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni | `scripts/mmau/run_inference.py` | `scripts/mmar/run_inference_vllm.py` | `scripts.mmau_pro.run_qwen` |
| Audio-Flamingo-3 | `scripts/mmau/run_flamingo.py` | `scripts/mmar/run_flamingo_mmar.py` | `scripts.mmau_pro.run_flamingo` |
| DeSTA-2.5 | `scripts/mmau/run_desta.py` | `scripts/mmar/run_desta_mmar.py` | `scripts.mmau_pro.run_desta` |
| Phi-4-Multimodal | `scripts/mmau/run_phi4_hf.py` | `scripts/mmar/run_phi4_hf_mmar.py` | `scripts.mmau_pro.run_phi4_hf` |
| Voxtral-Mini-3B | `scripts/mmau/run_voxtral.py` | `scripts/mmar/run_voxtral_inference.py` | `scripts.mmau_pro.run_voxtral` |

Outputs follow `results/<benchmark>/<model>/<condition>.jsonl` (`.json` for MMAR), where the condition is
`full`, `none`, or `n<N>_chunk<K>`.

Full and None conditions:

```bash
python scripts/mmau/run_inference.py --model_id Qwen/Qwen2.5-Omni-7B --use_mini \
    --audio_condition full --output_path results/mmau/qwen2.5_omni_7b/full.jsonl
python scripts/mmau/run_inference.py --model_id Qwen/Qwen2.5-Omni-7B --use_mini \
    --audio_condition none --output_path results/mmau/qwen2.5_omni_7b/none.jsonl
```

`scripts/mmau/run_inference.py` evaluates the 9,000-item MMAU test split unless `--use_mini` is given.

Fragment conditions (N = 2–5, all K) run through a wrapper that uses two GPUs:

```bash
bash scripts/mmau/run_chunked.sh Qwen/Qwen2.5-Omni-7B results/mmau/qwen2.5_omni_7b
INFERENCE_SCRIPT=scripts/mmar/run_voxtral_inference.py \
    bash scripts/mmar/run_chunked.sh mistralai/Voxtral-Mini-3B-2507 results/mmar/voxtral_mini_3b
INFERENCE_SCRIPT=scripts.mmau_pro.run_voxtral \
    bash scripts/mmau_pro/run_chunked.sh mistralai/Voxtral-Mini-3B-2507 results/mmau_pro/voxtral_mini_3b
```

Text backbones:

```bash
python scripts/run_text_backbone.py --model_id Qwen/Qwen2.5-7B-Instruct --benchmark mmau \
    --output_path results/mmau/tb_qwen2.5_7b_instruct/none.jsonl
```

Thinking models can exhaust the token budget before closing their reasoning. `scripts/retry_truncated.py`
re-runs only those items with a larger budget and patches them into the results file:

```bash
python scripts/retry_truncated.py --results_file results/mmar/qwen3_omni_30b_a3b_thinking/none.json \
    --model_id Qwen/Qwen3-Omni-30B-A3B-Thinking --benchmark mmar --max_tokens 8192 --max_model_len 16384
```

Decoding parameters come from `configs/generation_params.yaml`. Inference library versions were not
recorded with the results, so re-running inference is not expected to reproduce the released predictions
exactly.
