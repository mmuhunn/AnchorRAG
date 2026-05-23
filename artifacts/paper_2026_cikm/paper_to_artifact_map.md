# Paper To Artifact Map

This map uses paths relative to the public repository root.

| Paper claim / object | Paper location | Public artifact | Notes |
|---|---|---|---|
| Human-curated 210-query set over seven AllSides/Qbias topics | Setup | `data/queries/*_30.json`, `artifacts/paper_2026_cikm/source_results/exp4_eval_data_v5_7methods_210q_20260508.jsonl` | Six topics start from seed/candidate expansion and manual review; Climate Change uses the curated 30-query set. |
| Community-structure diagnostics | Table 2 | `artifacts/paper_2026_cikm/paper_values.yaml` | Values are frozen from the current manuscript; graph regeneration will be wired in a later live-reproduction phase. |
| Retrieval-side validation | Table 3 | `artifacts/paper_2026_cikm/paper_values.yaml` | Cached table values are frozen from the current manuscript. |
| Main trust-sensitive downstream results | Table 4 | `artifacts/paper_2026_cikm/source_results/exp4_eval_summary_v5_gpt4o_20260508.csv`, `artifacts/paper_2026_cikm/source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl` | GPT-4o is the primary judge. |
| External MBFC citation/backbone validation | Table 5 | `artifacts/paper_2026_cikm/supporting_experiments/mbfc_external_validation/mbfc_external_eval_summary_20260513.csv`, `artifacts/paper_2026_cikm/supporting_experiments/mbfc_external_validation/mbfc_external_eval_per_query_20260513.csv` | External source-factuality proxy; includes borderline MBFC HT Cite vs vanilla. |
| Paired significance marks | Tables 4 and 5 | `artifacts/paper_2026_cikm/supporting_experiments/statistical_tests/stat_tests_20260513.csv` | Paired bootstrap and Wilcoxon outputs. |
| Cross-judge answer-quality diagnostics | Table 6 | `artifacts/paper_2026_cikm/source_results/exp4_eval_summary_v5_gpt4o_20260508.csv`, `artifacts/paper_2026_cikm/source_results/exp4_eval_summary_v5_claude_20260508.csv` | GPT-4o / Claude values are both stored in the summaries and paper baseline. |
| Top-10 trust-advantaged query slice | Table 7 | `artifacts/paper_2026_cikm/source_results/table5_v5_trust_advantaged_slice_gpt4o_20260508.csv`, `artifacts/paper_2026_cikm/source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl` | File name preserves the earlier internal name `table5`; in the current manuscript this is Table 7. |
| Ablation summary | Table 8 | `artifacts/paper_2026_cikm/supporting_experiments/ablation_210q/source_summaries/`, `artifacts/paper_2026_cikm/supporting_experiments/ablation_210q/source_judgments/`, `artifacts/paper_2026_cikm/tables/table8_ablation.csv` | `scripts/make_supporting.py` aggregates the seven final 30q topic snapshots and checks values against `paper_values.yaml`. |
| Source prior validation against MBFC top-50 | Figure 2 / Results | `artifacts/paper_2026_cikm/supporting_experiments/source_prior_validation/mbfc_validation_top50.csv`, `data/source_validation/mbfc_validation_top50.csv` | Spearman = 0.662; weighted Spearman = 0.728. |
| Diagnostic failure case | Figure 3 | `artifacts/paper_2026_cikm/paper_values.yaml`, `artifacts/paper_2026_cikm/source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl` | Query `imm-01`. |
| Current PDF snapshot | Whole paper | `artifacts/paper_2026_cikm/paper_pdf/_CIKM2026__AnchorRAG (13).pdf` | Current PDF snapshot used for Phase 0 value baseline. |

## Caution

The public repo intentionally starts from the v5 7-method 210-query paper baseline. Older v2/v3/v6, six-topic, 60-query, smoke, debug, and dry-run outputs are not copied into the public tree unless a later release explicitly adds them as non-main provenance.
