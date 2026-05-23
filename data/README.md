# Data

This directory contains public-facing inputs needed to understand and reproduce the paper artifact package.

- `queries/`: canonical seven-topic query files. The CIKM paper uses 30 queries per topic, 210 total.
- `graphs/`: derived scored topic graphs used by `scripts/run_main_210q.py` for live reruns.
- `source_validation/`: MBFC mapping files used to validate the source-level reliability prior.

Raw third-party corpora are not included. The graph files contain derived snippets and scoring metadata; confirm redistribution policy before a final public release.
