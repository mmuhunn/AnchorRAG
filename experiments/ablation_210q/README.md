# Ablation 210Q

Cached inputs:

- `artifacts/paper_2026_cikm/supporting_experiments/ablation_210q/source_summaries/`
- `artifacts/paper_2026_cikm/supporting_experiments/ablation_210q/source_judgments/`

Regenerate Table 8:

```bash
python3.8 scripts/make_supporting.py --target table8
```

The public aggregator uses the seven final 30-query topic snapshots. Most Table
8 columns come from the per-topic summary CSVs; the `claims` column is the mean
`evaluable_sentences` value from the judged JSONL files.
