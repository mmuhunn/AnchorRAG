# Sensitivity Snapshot

This folder separates compact manuscript values from appendix/supporting
sensitivity artifacts.

- `compact_sensitivity_values.csv`: generated from `paper_values.yaml` by
  `scripts/make_supporting.py`.
- `phase2_retrieval_grid/`: retrieval-side alpha/beta and community-penalty
  summaries.
- `phase3_anchor_card_proxy/`: anchor/card proxy summaries.
- `phase4_tau_eval/`: tau-threshold rescoring and rank-stability summaries.
- `phase5_downstream/`: downstream generation/evaluation sensitivity summaries.
- `appendix_artifacts/`: appendix tables, suggested text, and alpha/beta figure.

Run:

```bash
python3.8 scripts/make_supporting.py --target sensitivity
```
