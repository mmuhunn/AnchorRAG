# Exp4 Pipeline Snapshot

This package contains the runnable generation and judging pipeline used by the
main 210-query experiment.

Public users should normally call:

```bash
python3.8 scripts/run_main_210q.py --dry-run
python3.8 scripts/run_main_210q.py --stage generate
python3.8 scripts/run_main_210q.py --stage evaluate --judge primary
```

The scripts in this package are kept close to the submission-time implementation.
`scripts/run_main_210q.py` is the public config adapter that injects repository
relative paths, model pins, and resume-safe output locations.
