# AnchorRAG

AnchorRAG is a trust-aware retrieval and anchor-guided synthesis framework for controversial question answering. This public repository is being prepared for the CIKM 2026 paper artifact release.

The repository is organized to support two reproduction modes:

- `cached`: regenerate and audit paper tables/figures from included JSONL/CSV artifacts without API keys.
- `live`: rerun generation and judging with model APIs using the config-driven 210-query runner.

## Current Status

This tree currently contains the Phase 3 public facade:

- `artifacts/paper_2026_cikm/`: frozen paper artifacts, manifest, current PDF snapshot, and paper value baseline.
- `data/queries/`: canonical seven-topic 210-query setup, stored as seven `*_30.json` files.
- `data/graphs/`: derived scored topic graphs used by the live rerun path.
- `data/source_validation/`: MBFC source-prior validation inputs.
- `configs/`, `experiments/`, `scripts/`, `docs/`: cached and live reproduction entrypoints.

## Quick Start

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make audit
```

`make audit` verifies that the manifest files exist, checks SHA-256 digests, validates row counts for CSV/JSONL artifacts, and parses the paper value baseline.

To regenerate the current cached paper tables and Figure 2:

```bash
make reproduce-cached
```

To validate the live rerun configuration without API calls:

```bash
make live-dry-run
```

To rerun generation and judging with APIs:

```bash
make reproduce-main-live
```

Live mode writes resume-safe outputs under `outputs/main_210q_live/<run_id>/`.

## Main Paper Artifacts

The current source-of-truth artifact package is:

```text
artifacts/paper_2026_cikm/
  MANIFEST.csv
  paper_values.yaml
  paper_to_artifact_map.md
  paper_pdf/
  source_results/
  supporting_experiments/
```

The main downstream paper run is the v5 7-method, 210-query evaluation dated `20260508`.
The live runner is `scripts/run_main_210q.py`, backed by
`configs/runs/main_210q_live.yaml`.

## Data Notice

The included query sets, derived graphs, and paper artifacts are intended for paper reproduction. The derived topic graphs in `data/graphs/` ship with the raw article body (`text`) removed, because the underlying AllSides/Qbias news corpus is third-party content that cannot be redistributed in full. The graph structure, derived trust/polarity scores, source names, stance labels, and article titles are retained.

Cached reproduction and the artifact audit run without the raw body. For live reproduction, obtain the AllSides/Qbias subset CSVs and re-hydrate the graphs:

```bash
python3.8 scripts/hydrate_graphs.py --subset-dir /path/to/AllSides_Qbias/data/subsets
```

See `data/graphs/README.md` for details.

## License

Code in this repository is released under the Apache License 2.0 (see `LICENSE`). The Apache license covers the source code and scripts. Third-party data (the AllSides/Qbias corpus and any raw article text) is not relicensed and remains subject to its original terms; see the Data Notice above.
