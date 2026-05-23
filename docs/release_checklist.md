# Release Checklist

- Choose final license.
- Fill final author metadata in `CITATION.cff`.
- Run `make smoke` and `make reproduce-cached` in a clean clone.
- Confirm no `.env`, API keys, nohup logs, or local absolute secret paths are present.
- Confirm derived graph snippet redistribution terms.
- Confirm live dry-run succeeds with `make live-dry-run`.
- Document live reproduction cost and runtime.
- Tag the release after cached reproduction passes.
