"""Paired bootstrap CI + Wilcoxon + Cliff's delta for AnchorRAG vs baselines/ablations.

Phase 2.1 (plan v3). Computes per-method-pair, per-metric statistical tests on the
210-query paired score matrix. AnchorRAG (trustrag_anchor) is the reference method.

Inputs (resolved relative to artifacts/paper_2026_cikm/):
  - Per-query judged eval scores (1470 records = 7 methods x 210 q):
      source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl
  - Per-query MBFC external scores (1470 records):
      supporting_experiments/mbfc_external_validation/mbfc_external_eval_per_query_20260513.csv

Outputs:
  - Long-form CSV with one row per (reference, comparison, metric):
      mbfc_external_eval_stat_tests_20260513.csv
  - Compact LaTeX-ready summary printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent

EVAL_JSONL = PROJECT_ROOT / "source_results/eval_v5_7methods_210q_gpt4o_20260508.jsonl"
MBFC_PERQ_CSV = PROJECT_ROOT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_per_query_20260513.csv"

OUT_STATS_CSV = PROJECT_ROOT / "supporting_experiments/statistical_tests/stat_tests_20260513.csv"

REFERENCE_METHOD = "trustrag_anchor"

# Metric name -> source ("eval" or "mbfc") + field name in the source
METRICS = [
    # Judge-dependent answer-quality (Table 4 right side)
    ("Groundedness", "eval", "groundedness"),
    ("Support", "eval", "evidence_support_rate"),
    ("Utility", "eval", "response_utility"),
    # Trust-sensitive (Table 4 left side)
    ("HT-Cite", "eval", "high_trust_citation_rate"),
    ("Cite-Cov", "eval", "citation_coverage"),
    ("Anchor-Util", "eval", "anchor_utilization_rate"),
    ("Backbone-Trust", "eval", "backbone_trust_rate"),
    ("TW-Claim-Support", "eval", "trust_weighted_claim_support"),
    ("Avg-Strust@k", "eval", "avg_s_trust_at_k"),
    ("Top3-Avg-Strust", "eval", "top3_avg_s_trust"),
    # External MBFC (new in Phase 2.0)
    ("MBFC-HT-Cite", "mbfc", "mbfc_ht_cite_rate"),
    ("MBFC-AvgFactual", "mbfc", "mbfc_avg_factual_all"),
    ("MBFC-Coverage", "mbfc", "mbfc_coverage_all"),
    ("MBFC-BB-HT", "mbfc", "mbfc_ht_backbone_rate"),
    ("MBFC-BB-AvgFactual", "mbfc", "mbfc_avg_factual_backbone"),
]

# Methods to compare against the reference
COMPARISON_METHODS = [
    "vanilla_rag",
    "mmr_rag",
    "graph_retrieval",
    "r2ag",
    "reclaim",
    "grag",
]


def load_eval_scores(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """(query_id, method) -> {metric_field: value or None}."""
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["query_id"], rec["method"])
            out[key] = {
                "groundedness": _to_float(rec.get("groundedness")),
                "evidence_support_rate": _to_float(rec.get("evidence_support_rate")),
                "response_utility": _to_float(rec.get("response_utility")),
                "high_trust_citation_rate": _to_float(rec.get("high_trust_citation_rate")),
                "citation_coverage": _to_float(rec.get("citation_coverage")),
                "anchor_utilization_rate": _to_float(rec.get("anchor_utilization_rate")),
                "backbone_trust_rate": _to_float(rec.get("backbone_trust_rate")),
                "trust_weighted_claim_support": _to_float(rec.get("trust_weighted_claim_support")),
                "avg_s_trust_at_k": _to_float(rec.get("avg_s_trust_at_k")),
                "top3_avg_s_trust": _to_float(rec.get("top3_avg_s_trust")),
            }
    return out


def load_mbfc_scores(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """(query_id, method) -> mbfc metric dict."""
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["query_id"], row["method"])
            out[key] = {
                "mbfc_ht_cite_rate": _to_float(row.get("mbfc_ht_cite_rate")),
                "mbfc_avg_factual_all": _to_float(row.get("mbfc_avg_factual_all")),
                "mbfc_coverage_all": _to_float(row.get("mbfc_coverage_all")),
                "mbfc_ht_backbone_rate": _to_float(row.get("mbfc_ht_backbone_rate")),
                "mbfc_avg_factual_backbone": _to_float(row.get("mbfc_avg_factual_backbone")),
            }
    return out


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def paired_arrays(
    eval_scores: dict, mbfc_scores: dict, ref: str, cmp: str, metric_source: str, field: str
) -> tuple[list[float], list[float]]:
    """Return paired arrays (ref_vals, cmp_vals) for queries where BOTH have non-None values."""
    src = eval_scores if metric_source == "eval" else mbfc_scores
    # Collect all query_ids that have both methods scored
    qids = sorted({qid for (qid, m) in src.keys() if m in (ref, cmp)})
    ref_vals: list[float] = []
    cmp_vals: list[float] = []
    for qid in qids:
        v_ref = src.get((qid, ref), {}).get(field)
        v_cmp = src.get((qid, cmp), {}).get(field)
        if v_ref is None or v_cmp is None:
            continue
        ref_vals.append(float(v_ref))
        cmp_vals.append(float(v_cmp))
    return ref_vals, cmp_vals


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired differences. Returns (W, two-sided p).

    Uses normal approximation with tie correction. Discards zero diffs.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    abs_diffs = [abs(d) for d in nz]
    # Rank with average ties
    paired = sorted(enumerate(abs_diffs), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    tie_correction = 0.0
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        t = j - i + 1
        if t > 1:
            tie_correction += t**3 - t
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg_rank
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(w_plus, w_minus)
    # Normal approximation
    mu = n * (n + 1) / 4.0
    sigma_sq = n * (n + 1) * (2 * n + 1) / 24.0
    sigma_sq -= tie_correction / 48.0
    sigma = math.sqrt(sigma_sq) if sigma_sq > 0 else 1.0
    z = (w - mu) / sigma if sigma > 0 else 0.0
    # Two-sided p from standard normal
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w, p


def paired_bootstrap_ci(
    ref_vals: list[float], cmp_vals: list[float], n_boot: int, alpha: float, seed: int
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(ref - cmp). Returns (mean_diff, lo, hi)."""
    diffs = [r - c for r, c in zip(ref_vals, cmp_vals)]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return mean_diff, lo, hi


def cliffs_delta(ref_vals: list[float], cmp_vals: list[float]) -> float:
    """Cliff's delta on the unpaired distributions (sign of dominance).

    delta = (#{r>c} - #{r<c}) / (n_ref * n_cmp)
    Range: [-1, 1]. Positive means ref tends to dominate cmp.
    """
    gt = lt = 0
    for r in ref_vals:
        for c in cmp_vals:
            if r > c:
                gt += 1
            elif r < c:
                lt += 1
    total = len(ref_vals) * len(cmp_vals)
    if total == 0:
        return 0.0
    return (gt - lt) / total


def magnitude_label(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--eval-jsonl", default=str(EVAL_JSONL))
    ap.add_argument("--mbfc-csv", default=str(MBFC_PERQ_CSV))
    ap.add_argument("--out-csv", default=str(OUT_STATS_CSV))
    args = ap.parse_args()

    print(f"[1/3] Loading eval scores from {args.eval_jsonl}...")
    eval_scores = load_eval_scores(Path(args.eval_jsonl))
    print(f"  {len(eval_scores)} (qid, method) records")

    print(f"[2/3] Loading MBFC scores from {args.mbfc_csv}...")
    mbfc_scores = load_mbfc_scores(Path(args.mbfc_csv))
    print(f"  {len(mbfc_scores)} (qid, method) records")

    print(f"[3/3] Running paired tests: ref={REFERENCE_METHOD}, "
          f"n_boot={args.n_boot}, alpha={args.alpha}")

    rows: list[dict] = []
    for cmp in COMPARISON_METHODS:
        for label, msrc, field in METRICS:
            ref_vals, cmp_vals = paired_arrays(
                eval_scores, mbfc_scores, REFERENCE_METHOD, cmp, msrc, field
            )
            n = len(ref_vals)
            if n == 0:
                continue
            mean_ref = mean(ref_vals)
            mean_cmp = mean(cmp_vals)
            diffs = [r - c for r, c in zip(ref_vals, cmp_vals)]
            w_stat, p = wilcoxon_signed_rank(diffs)
            mean_diff, lo, hi = paired_bootstrap_ci(
                ref_vals, cmp_vals, args.n_boot, args.alpha, args.seed
            )
            delta = cliffs_delta(ref_vals, cmp_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            rows.append({
                "reference": REFERENCE_METHOD,
                "comparison": cmp,
                "metric": label,
                "metric_field": field,
                "n_paired": n,
                "mean_ref": round(mean_ref, 4),
                "mean_cmp": round(mean_cmp, 4),
                "mean_diff": round(mean_diff, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
                "wilcoxon_W": round(w_stat, 2),
                "wilcoxon_p_two_sided": round(p, 6),
                "sig": sig,
                "cliffs_delta": round(delta, 4),
                "magnitude": magnitude_label(delta),
            })

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> wrote {args.out_csv}")

    # Pretty summary printout
    print("\n=== AnchorRAG (trustrag_anchor) vs each method ===")
    header = f"{'cmp':<18}{'metric':<22}{'ref':<8}{'cmp':<8}{'Δ':<10}{'95%CI':<22}{'p':<10}{'sig':<5}{'δ':<10}{'mag':<10}"
    print(header)
    print("-" * len(header))
    for cmp in COMPARISON_METHODS:
        for label, _, _ in METRICS:
            r = next((x for x in rows if x["comparison"] == cmp and x["metric"] == label), None)
            if r is None:
                continue
            ci = f"[{r['ci_lo_95']:+.4f},{r['ci_hi_95']:+.4f}]"
            print(
                f"{cmp:<18}{label:<22}"
                f"{r['mean_ref']:<8.4f}{r['mean_cmp']:<8.4f}"
                f"{r['mean_diff']:<+10.4f}{ci:<22}"
                f"{r['wilcoxon_p_two_sided']:<10.4g}{r['sig']:<5}"
                f"{r['cliffs_delta']:<+10.4f}{r['magnitude']:<10}"
            )
        print("-" * len(header))


if __name__ == "__main__":
    main()
