#!/usr/bin/env python3.8
"""Regenerate cached paper figures from public artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/paper_2026_cikm"
FIGURES = ARTIFACT / "figures"
PAPER_VALUES = ARTIFACT / "paper_values.yaml"
SOURCE_PRIOR = ARTIFACT / "supporting_experiments/source_prior_validation/mbfc_validation_top50.csv"


LABELS = {
    1: "Low",
    2: "Mixed",
    3: "Mostly\nFactual",
    4: "High",
    5: "Very\nHigh",
}


def read_rows() -> list[dict[str, str]]:
    with SOURCE_PRIOR.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_figure_data(rows: list[dict[str, object]]) -> None:
    path = FIGURES / "figure2_source_prior_mbfc_data.csv"
    fields = [
        "source_name",
        "mapped_source_name",
        "total_count",
        "current_source_score_mean",
        "mbfc_factual_numeric",
        "mbfc_factual_reporting",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure2() -> None:
    values = yaml.safe_load(PAPER_VALUES.read_text())["figure_2_source_prior_mbfc_validation"]
    source_rows = read_rows()
    matched = []
    for row in source_rows:
        if not row.get("current_source_score_mean") or not row.get("mbfc_factual_numeric"):
            continue
        matched.append(
            {
                "source_name": row["source_name"],
                "mapped_source_name": row.get("mapped_source_name") or row["canonical_source_name"],
                "total_count": int(float(row["total_count"])),
                "current_source_score_mean": float(row["current_source_score_mean"]),
                "mbfc_factual_numeric": float(row["mbfc_factual_numeric"]),
                "mbfc_factual_reporting": row["mbfc_factual_reporting"],
            }
        )

    xs = [row["current_source_score_mean"] for row in matched]
    ys = [row["mbfc_factual_numeric"] for row in matched]
    expanded_x = []
    expanded_y = []
    for row in matched:
        expanded_x.extend([row["current_source_score_mean"]] * row["total_count"])
        expanded_y.extend([row["mbfc_factual_numeric"]] * row["total_count"])

    spearman = spearmanr(xs, ys).correlation
    weighted_spearman = spearmanr(expanded_x, expanded_y).correlation
    covered_documents = sum(row["total_count"] for row in matched)

    if len(matched) != values["matched_sources"]:
        raise AssertionError(f"Figure 2 matched sources mismatch: {len(matched)}")
    if covered_documents != values["covered_documents"]:
        raise AssertionError(f"Figure 2 covered document mismatch: {covered_documents}")
    if round(spearman, 3) != round(float(values["spearman"]), 3):
        raise AssertionError(f"Figure 2 Spearman mismatch: {spearman}")
    if round(weighted_spearman, 3) != round(float(values["weighted_spearman"]), 3):
        raise AssertionError(f"Figure 2 weighted Spearman mismatch: {weighted_spearman}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    write_figure_data(matched)

    plt.figure(figsize=(7.2, 4.6))
    sizes = [28 + row["total_count"] * 1.1 for row in matched]
    plt.scatter(
        ys,
        xs,
        s=sizes,
        alpha=0.72,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.7,
    )
    top_labels = sorted(matched, key=lambda row: row["total_count"], reverse=True)[:10]
    for row in top_labels:
        plt.annotate(
            row["mapped_source_name"],
            (row["mbfc_factual_numeric"], row["current_source_score_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f2933",
        )
    plt.xticks([1, 2, 3, 4, 5], [LABELS[i] for i in [1, 2, 3, 4, 5]])
    plt.ylim(0.35, 1.0)
    plt.xlabel("MBFC Factual Reporting Label")
    plt.ylabel("Recovered Source Score")
    plt.title(
        f"Top-50 matched sources: {len(matched)} | Corpus coverage: 89.7% | "
        f"Spearman: {spearman:.3f} | Weighted: {weighted_spearman:.3f}",
        fontsize=10,
    )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "figure2_source_prior_mbfc.png", dpi=220)
    plt.savefig(FIGURES / "figure2_source_prior_mbfc.pdf")
    plt.close()
    print(f"[make_figures] wrote Figure 2 artifacts to {FIGURES.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["all", "figure2"], default="all")
    args = parser.parse_args()
    if args.figure in {"all", "figure2"}:
        make_figure2()


if __name__ == "__main__":
    main()
