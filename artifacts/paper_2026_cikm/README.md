# CIKM 2026 Paper Artifacts

This directory is the public artifact package for the AnchorRAG CIKM 2026 submission.

Important files:

- `MANIFEST.csv`: public manifest with file sizes, row counts, SHA-256 digests, and paper usage.
- `paper_values.yaml`: machine-readable baseline of values printed in the current paper PDF.
- `paper_to_artifact_map.md`: mapping from paper claims/tables/figures to public artifacts.
- `source_results/`: v5 7-method, 210-query outputs and summaries.
- `supporting_experiments/`: MBFC validation, statistical tests, source-prior validation, ablation, sensitivity, and diagnostic artifacts.
- `paper_pdf/`: current PDF snapshot used for the Phase 0 value freeze.
- `tables/`: cached regenerated CSV/TeX tables for Tables 4-8.
- `figures/`: cached regenerated Figure 2 data and plots.

The current main baseline is v5. Older v2/v3/v6 and 60-query pilot outputs are intentionally not included in this public tree.

Supporting-experiment regeneration currently includes Table 8 ablations and the
compact sensitivity values:

```bash
python3.8 scripts/make_supporting.py
```
