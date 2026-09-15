# Results

Aggregate scores for every model, benchmark and condition used in the paper.

```
results/<benchmark>/<model>/<condition>_llm_judge_results.json    MCQ scores from the LLM judge (used in the paper)
results/<benchmark>/<model>/<condition>_results.json              official string-match scores (MMAU, MMAR)
results/mmau_pro/<model>/<condition>_comprehensive_results.json   MMAU-Pro official evaluation (open-ended, instruction following)
results/decomposition_summary.json                                model-averaged item decomposition
results/human_calibration.json                                    LLM judge vs. human annotation summary
```

`<benchmark>` is `mmau`, `mmar` or `mmau_pro`. `<condition>` is `full`, `none`, or `n<N>_chunk<K>` for the
K-th of N fragments. Text-backbone directories (`tb_*`) contain only the `none` condition.

Per-item predictions and judge decisions are distributed separately; see [../docs/reproduction.md](../docs/reproduction.md).
In `*_comprehensive_results.json`, `evaluation_summary.parquet_file` was rewritten from an absolute path to a
repository-relative path in four files.
