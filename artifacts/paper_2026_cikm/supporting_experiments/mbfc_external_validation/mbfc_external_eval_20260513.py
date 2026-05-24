"""MBFC-based external evaluation of citation trust (Phase 2.0, plan v3).

Re-computes High-Trust Citation Rate and Backbone Trust using MBFC factual
reporting labels as an external source-factuality proxy, instead of the internal S_trust
proxy. Addresses the circularity concern raised in the revision plan.

Inputs (resolved relative to artifacts/paper_2026_cikm/):
  - Raw 210q output (gives card_id -> source mapping per method):
      source_results/exp4_eval_data_v5_7methods_210q_20260508.jsonl
  - Judged eval output (gives support_checks per method/query):
      source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl
  - MBFC mapping (source_name -> mbfc_factual_numeric in 1..5):
      supporting_experiments/source_prior_validation/mbfc_validation_top50.csv

Outputs:
  - Per (method, query): mbfc_external_eval_per_query_20260513.csv
  - Per method aggregated: mbfc_external_eval_summary_20260513.csv
  - Per (method, topic): mbfc_external_eval_by_topic_20260513.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent

RAW_JSONL = PROJECT_ROOT / "source_results/exp4_eval_data_v5_7methods_210q_20260508.jsonl"
EVAL_JSONL = PROJECT_ROOT / "source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl"
MBFC_CSV = PROJECT_ROOT / "supporting_experiments/source_prior_validation/mbfc_validation_top50.csv"
MBFC_EXT_CSV = PROJECT_ROOT / "supporting_experiments/source_prior_validation/mbfc_extension_20260513.csv"

OUT_PER_QUERY = PROJECT_ROOT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_per_query_20260513.csv"
OUT_SUMMARY = PROJECT_ROOT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_summary_20260513.csv"
OUT_BY_TOPIC = PROJECT_ROOT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_by_topic_20260513.csv"
OUT_UNMAPPED = PROJECT_ROOT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_unmapped_sources_20260513.csv"

BACKBONE_K_DEFAULT = 3
HT_THRESHOLD_DEFAULT = 4  # 4 = "High", 5 = "Very High" in MBFC

CARD_RE = re.compile(r"\[\s*Card\s+(\d+)\s*\]", re.IGNORECASE)


def load_mbfc_mapping(csv_paths: list[Path]) -> dict[str, int]:
    """source_name -> mbfc_factual_numeric (1..5). Merges multiple CSVs."""
    mapping: dict[str, int] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("source_name", "").strip()
                val = row.get("mbfc_factual_numeric", "").strip()
                if not name or not val:
                    continue
                try:
                    mapping[name] = int(val)
                except ValueError:
                    continue
    return mapping


def load_card_source(jsonl_path: Path) -> dict[tuple[str, str, int], str]:
    """(query_id, method, card_id) -> source string."""
    out: dict[tuple[str, str, int], str] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            for method, retrievals in (rec.get("retrievals") or {}).items():
                for r in retrievals or []:
                    cid = r.get("card_id")
                    src = (r.get("source") or "").strip()
                    if cid is None:
                        continue
                    out[(qid, method, int(cid))] = src
    return out


def extract_card_refs(sentence: str) -> list[int]:
    return [int(m) for m in CARD_RE.findall(sentence or "")]


def compute_per_query_records(
    eval_jsonl: Path,
    card_source: dict[tuple[str, str, int], str],
    mbfc_map: dict[str, int],
    backbone_k: int,
    ht_threshold: int,
) -> tuple[list[dict], dict[str, int]]:
    """For each (method, query) compute MBFC-derived metrics.

    Returns (records, unmapped_source_counter).
    """
    records: list[dict] = []
    unmapped_counter: dict[str, int] = defaultdict(int)

    def mbfc_block(cites: list[str]) -> dict:
        n_total = len(cites)
        scores = []
        unmapped_local = 0
        for s in cites:
            if s in mbfc_map:
                scores.append(mbfc_map[s])
            else:
                unmapped_local += 1
                unmapped_counter[s] += 1
        n_mapped = len(scores)
        ht_count = sum(1 for v in scores if v >= ht_threshold)
        ht_rate = ht_count / n_mapped if n_mapped > 0 else None
        avg_score = mean(scores) if scores else None
        coverage = n_mapped / n_total if n_total > 0 else None
        return {
            "n_cite": n_total,
            "n_mapped": n_mapped,
            "n_unmapped": unmapped_local,
            "n_ht": ht_count,
            "ht_rate": ht_rate,
            "avg_factual": avg_score,
            "coverage": coverage,
        }

    with open(eval_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            method = rec["method"]
            topic = rec.get("topic", "")
            support = rec.get("support_checks") or []

            all_cites: list[str] = []
            backbone_cites: list[str] = []
            for i, sc in enumerate(support):
                refs = extract_card_refs(sc.get("sentence", ""))
                for cid in refs:
                    src = card_source.get((qid, method, cid))
                    if src is None or src == "":
                        continue
                    all_cites.append(src)
                    if i < backbone_k:
                        backbone_cites.append(src)

            all_m = mbfc_block(all_cites)
            bb_m = mbfc_block(backbone_cites)
            records.append(
                {
                    "query_id": qid,
                    "topic": topic,
                    "method": method,
                    # all-citation block
                    "n_cite_all": all_m["n_cite"],
                    "n_cite_all_mapped": all_m["n_mapped"],
                    "n_cite_all_unmapped": all_m["n_unmapped"],
                    "mbfc_ht_cite_rate": all_m["ht_rate"],
                    "mbfc_avg_factual_all": all_m["avg_factual"],
                    "mbfc_coverage_all": all_m["coverage"],
                    # backbone block
                    "n_cite_backbone": bb_m["n_cite"],
                    "n_cite_backbone_mapped": bb_m["n_mapped"],
                    "n_cite_backbone_unmapped": bb_m["n_unmapped"],
                    "mbfc_ht_backbone_rate": bb_m["ht_rate"],
                    "mbfc_avg_factual_backbone": bb_m["avg_factual"],
                    "mbfc_coverage_backbone": bb_m["coverage"],
                }
            )
    return records, dict(unmapped_counter)


def aggregate(group: list[dict]) -> dict:
    """Both per-query mean (None-skipping) and corpus-level (token-weighted)."""
    out: dict[str, float | int | None] = {}
    metric_keys = [
        "mbfc_ht_cite_rate",
        "mbfc_avg_factual_all",
        "mbfc_coverage_all",
        "mbfc_ht_backbone_rate",
        "mbfc_avg_factual_backbone",
        "mbfc_coverage_backbone",
    ]
    for k in metric_keys:
        vals = [r[k] for r in group if r[k] is not None]
        out[f"{k}_mean_perq"] = mean(vals) if vals else None
        out[f"{k}_n_perq"] = len(vals)

    # Corpus-level token totals
    total_cites = sum(r["n_cite_all"] for r in group)
    total_mapped = sum(r["n_cite_all_mapped"] for r in group)
    total_ht_mapped = sum(
        (r["n_cite_all_mapped"] * r["mbfc_ht_cite_rate"])
        for r in group
        if r["mbfc_ht_cite_rate"] is not None
    )
    total_bb = sum(r["n_cite_backbone"] for r in group)
    total_bb_mapped = sum(r["n_cite_backbone_mapped"] for r in group)
    total_bb_ht_mapped = sum(
        (r["n_cite_backbone_mapped"] * r["mbfc_ht_backbone_rate"])
        for r in group
        if r["mbfc_ht_backbone_rate"] is not None
    )

    out["corpus_total_cites_all"] = total_cites
    out["corpus_total_mapped_all"] = total_mapped
    out["corpus_coverage_all"] = total_mapped / total_cites if total_cites > 0 else None
    out["corpus_mbfc_ht_cite_rate_all"] = (
        total_ht_mapped / total_mapped if total_mapped > 0 else None
    )
    out["corpus_total_cites_backbone"] = total_bb
    out["corpus_total_mapped_backbone"] = total_bb_mapped
    out["corpus_coverage_backbone"] = (
        total_bb_mapped / total_bb if total_bb > 0 else None
    )
    out["corpus_mbfc_ht_backbone_rate"] = (
        total_bb_ht_mapped / total_bb_mapped if total_bb_mapped > 0 else None
    )
    out["n_queries"] = len(group)
    return out


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v) -> str:
    if v is None:
        return "  N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-k", type=int, default=BACKBONE_K_DEFAULT)
    ap.add_argument("--ht-threshold", type=int, default=HT_THRESHOLD_DEFAULT)
    ap.add_argument("--raw", default=str(RAW_JSONL))
    ap.add_argument("--eval", default=str(EVAL_JSONL))
    ap.add_argument("--mbfc", action="append", default=None,
                    help="MBFC CSV path(s). Repeat flag to merge multiple. "
                         "Default: top-50 + 20260513 extension.")
    args = ap.parse_args()

    mbfc_paths = [Path(p) for p in args.mbfc] if args.mbfc else [MBFC_CSV, MBFC_EXT_CSV]
    print("[1/4] Loading MBFC mapping from:")
    for p in mbfc_paths:
        print(f"  - {p}")
    mbfc_map = load_mbfc_mapping(mbfc_paths)
    print(f"  {len(mbfc_map)} source_name -> mbfc_factual_numeric entries (merged)")

    print("[2/4] Loading raw 210q JSONL for (qid, method, card_id) -> source...")
    card_source = load_card_source(Path(args.raw))
    print(f"  {len(card_source)} card entries loaded")

    print("[3/4] Computing MBFC metrics per (method, query)...")
    records, unmapped = compute_per_query_records(
        Path(args.eval),
        card_source,
        mbfc_map,
        backbone_k=args.backbone_k,
        ht_threshold=args.ht_threshold,
    )
    print(f"  {len(records)} method x query records")

    print("[4/4] Aggregating and writing outputs...")
    fieldnames = list(records[0].keys())
    write_csv(OUT_PER_QUERY, records, fieldnames)
    print(f"  -> {OUT_PER_QUERY.relative_to(PROJECT_ROOT)}")

    by_method: dict[str, list[dict]] = defaultdict(list)
    by_mt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)
        by_mt[(r["method"], r["topic"])].append(r)

    method_aggs = {m: aggregate(rs) for m, rs in by_method.items()}
    sample_agg = next(iter(method_aggs.values()))
    agg_fields = ["method", *sorted(sample_agg.keys())]
    method_rows = []
    for m in sorted(method_aggs.keys()):
        row = {"method": m, **method_aggs[m]}
        method_rows.append(row)
    write_csv(OUT_SUMMARY, method_rows, agg_fields)
    print(f"  -> {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")

    mt_rows = []
    mt_fields = ["method", "topic", *sorted(sample_agg.keys())]
    for (m, t), rs in sorted(by_mt.items()):
        row = {"method": m, "topic": t, **aggregate(rs)}
        mt_rows.append(row)
    write_csv(OUT_BY_TOPIC, mt_rows, mt_fields)
    print(f"  -> {OUT_BY_TOPIC.relative_to(PROJECT_ROOT)}")

    # Unmapped source frequency
    unmapped_rows = [
        {"source": s, "n_unmapped_cites": c}
        for s, c in sorted(unmapped.items(), key=lambda x: -x[1])
    ]
    write_csv(OUT_UNMAPPED, unmapped_rows, ["source", "n_unmapped_cites"])
    print(f"  -> {OUT_UNMAPPED.relative_to(PROJECT_ROOT)} ({len(unmapped_rows)} distinct unmapped sources)")

    print("\n=== Per-method summary (per-query mean) ===")
    header_cols = [
        ("method", 20),
        ("HT-Cite", 9),
        ("AvgFact", 9),
        ("Cov(all)", 9),
        ("BB-HT", 9),
        ("BB-AvgF", 9),
        ("BB-Cov", 9),
        ("N", 5),
    ]
    print(" ".join(f"{name:<{w}}" for name, w in header_cols))
    for m in sorted(method_aggs.keys()):
        d = method_aggs[m]
        print(" ".join(
            [
                f"{m:<20}",
                f"{fmt(d['mbfc_ht_cite_rate_mean_perq']):<9}",
                f"{fmt(d['mbfc_avg_factual_all_mean_perq']):<9}",
                f"{fmt(d['mbfc_coverage_all_mean_perq']):<9}",
                f"{fmt(d['mbfc_ht_backbone_rate_mean_perq']):<9}",
                f"{fmt(d['mbfc_avg_factual_backbone_mean_perq']):<9}",
                f"{fmt(d['mbfc_coverage_backbone_mean_perq']):<9}",
                f"{d['n_queries']:<5}",
            ]
        ))

    print("\n=== Per-method summary (corpus-level token-weighted) ===")
    for m in sorted(method_aggs.keys()):
        d = method_aggs[m]
        print(
            f"{m:<20} "
            f"HT(all)={fmt(d['corpus_mbfc_ht_cite_rate_all']):<9} "
            f"cov(all)={fmt(d['corpus_coverage_all']):<9} "
            f"HT(bb)={fmt(d['corpus_mbfc_ht_backbone_rate']):<9} "
            f"cov(bb)={fmt(d['corpus_coverage_backbone']):<9} "
            f"cites_all={d['corpus_total_cites_all']} "
            f"cites_bb={d['corpus_total_cites_backbone']}"
        )


if __name__ == "__main__":
    main()
