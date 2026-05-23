# Reproducibility

AnchorRAG public reproduction is split into two modes.

## Cached Mode

Cached mode uses included artifacts under `artifacts/paper_2026_cikm/`.

Current commands:

```bash
make audit
make reproduce-cached
```

`make reproduce-cached` regenerates:

- `artifacts/paper_2026_cikm/tables/table4_trust_sensitive_evidence_use.{csv,tex}`
- `artifacts/paper_2026_cikm/tables/table5_mbfc_external_rescoring.{csv,tex}`
- `artifacts/paper_2026_cikm/tables/table6_answer_quality_diagnostics.{csv,tex}`
- `artifacts/paper_2026_cikm/tables/table7_top10_trust_advantaged_slice.{csv,tex}`
- `artifacts/paper_2026_cikm/tables/table8_ablation.{csv,tex}`
- `artifacts/paper_2026_cikm/figures/figure2_source_prior_mbfc.{png,pdf}`
- `artifacts/paper_2026_cikm/figures/figure2_source_prior_mbfc_data.csv`
- `artifacts/paper_2026_cikm/supporting_experiments/ablation_210q/multi_anchor_note.csv`
- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/compact_sensitivity_values.csv`

The table, figure, ablation, and sensitivity scripts check regenerated values
against `paper_values.yaml`.

Cached mode should not require API keys.

## Live Mode

Live mode reruns generation and judging with model APIs.

First validate the config, graph inputs, query files, and output convention without API calls:

```bash
make live-dry-run
```

Generation:

```bash
python3.8 scripts/run_main_210q.py --stage generate --run-id main_210q_live
```

Primary GPT-4o judging:

```bash
python3.8 scripts/run_main_210q.py --stage evaluate --judge primary --run-id main_210q_live
```

Claude robustness judging:

```bash
python3.8 scripts/run_main_210q.py --stage evaluate --judge robustness --run-id main_210q_live
```

All live stages:

```bash
make reproduce-main-live
```

Live mode requires `OPENAI_API_KEY` for generation and primary judging. It requires
`ANTHROPIC_API_KEY` for Claude robustness judging.

Default output layout:

- `outputs/main_210q_live/<run_id>/generation/exp4_eval_data_live_7methods_210q.jsonl`
- `outputs/main_210q_live/<run_id>/cache/embeddings/`
- `outputs/main_210q_live/<run_id>/judging/primary/`
- `outputs/main_210q_live/<run_id>/judging/robustness/`
- `outputs/main_210q_live/<run_id>/run_manifest.json`

Resume behavior:

- Generation resumes by reading completed `(topic, query_id)` rows from the existing generation JSONL and appending missing rows.
- Evaluation resumes by reading existing judged JSONL files and appending missing method/query judgments.

The live rerun is not expected to be byte-identical to cached paper artifacts because
model APIs can change behavior over time. Cached mode remains the exact paper-value
audit path.
