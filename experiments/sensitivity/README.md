# Sensitivity

Cached inputs:

- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/phase2_retrieval_grid/`
- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/phase3_anchor_card_proxy/`
- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/phase4_tau_eval/`
- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/phase5_downstream/`
- `artifacts/paper_2026_cikm/supporting_experiments/sensitivity/appendix_artifacts/`

Regenerate compact manuscript values:

```bash
python3.8 scripts/make_supporting.py --target sensitivity
```

The compact CSV is generated from `paper_values.yaml`; the copied phase folders
preserve the appendix/supporting artifacts behind the manuscript text.
