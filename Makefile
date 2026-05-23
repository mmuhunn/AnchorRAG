RUN_ID ?= main_210q_live
LIVE_ARGS ?=

.PHONY: audit reproduce-cached reproduce-main-live live-dry-run smoke tables figures supporting

audit:
	python3.8 scripts/audit_release.py

tables:
	python3.8 scripts/make_tables.py

figures:
	python3.8 scripts/make_figures.py

supporting:
	python3.8 scripts/make_supporting.py

reproduce-cached: tables figures supporting audit
	@echo "Cached reproduction completed."

live-dry-run:
	python3.8 scripts/run_main_210q.py --dry-run $(LIVE_ARGS)

reproduce-main-live:
	python3.8 scripts/run_main_210q.py --stage all --judge both --run-id $(RUN_ID) $(LIVE_ARGS)

smoke: audit live-dry-run
	@echo "Smoke test completed."
