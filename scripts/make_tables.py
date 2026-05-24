#!/usr/bin/env python3.8
"""Regenerate cached paper tables from public artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/paper_2026_cikm"
SOURCE = ARTIFACT / "source_results"
TABLES = ARTIFACT / "tables"
PAPER_VALUES = ARTIFACT / "paper_values.yaml"

METHOD_ORDER_MAIN = [
    "vanilla_rag",
    "mmr_rag",
    "r2ag",
    "reclaim",
    "grag",
    "trustrag_anchor",
]

METHOD_ORDER_DIAG = [
    "vanilla_rag",
    "mmr_rag",
    "graph_retrieval",
    "r2ag",
    "reclaim",
    "grag",
    "trustrag_anchor",
]

DISPLAY = {
    "vanilla_rag": "vanilla rag",
    "mmr_rag": "mmr rag",
    "graph_retrieval": "GL-RAG",
    "r2ag": "R2AG",
    "reclaim": "ReClaim",
    "grag": "GRAG",
    "trustrag_anchor": "AnchorRAG",
}

PAPER_KEY = {
    "trustrag_anchor": "anchorrag",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt4(value: object) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def fmt3(value: object) -> str:
    formatted = str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return formatted[1:] if formatted.startswith("0") else formatted


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def load_values() -> dict:
    return yaml.safe_load(PAPER_VALUES.read_text())


def all_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(path) if row["topic"] == "all"}


def public_key(method: str) -> str:
    return PAPER_KEY.get(method, method)


def assert_close(name: str, got: object, expected: object, digits: int) -> None:
    if round(float(got), digits) != round(float(expected), digits):
        raise AssertionError(f"{name}: got {got}, expected {expected}")


def latex_table(path: Path, headers: list[str], rows: list[list[object]], align: str) -> None:
    with path.open("w") as handle:
        handle.write("\\begin{tabular}{" + align + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(headers) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(tex_escape(v) for v in row) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def make_table4(values: dict) -> None:
    summary = all_rows(SOURCE / "exp4_eval_summary_v5_gpt4o_20260508.csv")
    fields = [
        "method",
        "citation_coverage",
        "high_trust_citation_rate",
        "anchor_utilization_rate",
        "backbone_trust_rate",
        "avg_s_trust_at_k",
    ]
    rows: list[dict[str, object]] = []
    tex_rows: list[list[object]] = []
    baseline = values["table_4_trust_sensitive_evidence_use"]["rows"]
    for method in METHOD_ORDER_MAIN:
        src = summary[method]
        out = {
            "method": method,
            "citation_coverage": fmt4(src["citation_coverage"]),
            "high_trust_citation_rate": fmt4(src["high_trust_citation_rate"]),
            "anchor_utilization_rate": fmt4(src["anchor_utilization_rate"]),
            "backbone_trust_rate": fmt4(src["backbone_trust_rate"]),
            "avg_s_trust_at_k": fmt4(src["avg_s_trust_at_k"]),
        }
        rows.append(out)
        expected = baseline[public_key(method)]
        for field in fields[1:]:
            assert_close(f"table4.{method}.{field}", out[field], expected[field], 4)
        tex_rows.append([DISPLAY[method], *[out[field] for field in fields[1:]]])
    write_csv(TABLES / "table4_trust_sensitive_evidence_use.csv", rows, fields)
    latex_table(
        TABLES / "table4_trust_sensitive_evidence_use.tex",
        ["Method", "Cite Cov.", "HT Cite", "Anchor Util.", "Backbone Trust", "Avg $S_{trust}@k$"],
        tex_rows,
        "lccccc",
    )


def make_table5(values: dict) -> None:
    rows_by_method = {
        row["method"]: row
        for row in read_csv(ARTIFACT / "supporting_experiments/mbfc_external_validation/mbfc_external_eval_summary_20260513.csv")
    }
    field_map = [
        ("mbfc_high_trust_citation_rate", "mbfc_ht_cite_rate_mean_perq"),
        ("mbfc_avg_factual_all", "mbfc_avg_factual_all_mean_perq"),
        ("mbfc_backbone_high_trust_rate", "mbfc_ht_backbone_rate_mean_perq"),
        ("mbfc_backbone_avg_factual", "mbfc_avg_factual_backbone_mean_perq"),
        ("mbfc_map_coverage", "mbfc_coverage_all_mean_perq"),
    ]
    fields = ["method"] + [left for left, _ in field_map]
    out_rows: list[dict[str, object]] = []
    tex_rows: list[list[object]] = []
    baseline = values["table_5_mbfc_external_rescoring"]["rows"]
    for method in METHOD_ORDER_MAIN:
        src = rows_by_method[method]
        out = {"method": method}
        for public_field, source_field in field_map:
            out[public_field] = fmt4(src[source_field])
        out_rows.append(out)
        expected = baseline[public_key(method)]
        for field in fields[1:]:
            assert_close(f"table5.{method}.{field}", out[field], expected[field], 4)
        tex_rows.append([DISPLAY[method], *[out[field] for field in fields[1:]]])
    write_csv(TABLES / "table5_mbfc_external_rescoring.csv", out_rows, fields)
    latex_table(
        TABLES / "table5_mbfc_external_rescoring.tex",
        ["Method", "MBFC HT Cite", "MBFC Avg Factual", "MBFC Backbone HT", "MBFC Backbone Avg Factual", "MBFC Map Cov."],
        tex_rows,
        "lccccc",
    )


def make_table6(values: dict) -> None:
    gpt = all_rows(SOURCE / "exp4_eval_summary_v5_gpt4o_20260508.csv")
    claude = all_rows(SOURCE / "exp4_eval_summary_v5_claude_20260508.csv")
    fields = [
        "method",
        "groundedness_gpt4o",
        "groundedness_claude",
        "support_gpt4o",
        "support_claude",
        "utility_gpt4o",
        "utility_claude",
    ]
    metric_pairs = [
        ("groundedness_gpt4o", gpt, "groundedness"),
        ("groundedness_claude", claude, "groundedness"),
        ("support_gpt4o", gpt, "evidence_support_rate"),
        ("support_claude", claude, "evidence_support_rate"),
        ("utility_gpt4o", gpt, "response_utility"),
        ("utility_claude", claude, "response_utility"),
    ]
    out_rows: list[dict[str, object]] = []
    tex_rows: list[list[object]] = []
    baseline = values["table_6_answer_quality_diagnostics"]["rows"]
    for method in METHOD_ORDER_DIAG:
        out = {"method": method}
        for public_field, source_rows, source_field in metric_pairs:
            out[public_field] = fmt3(source_rows[method][source_field])
        out_rows.append(out)
        expected = baseline[public_key(method)]
        for field in fields[1:]:
            assert_close(f"table6.{method}.{field}", out[field], expected[field], 3)
        tex_rows.append(
            [
                DISPLAY[method],
                f"{out['groundedness_gpt4o']} / {out['groundedness_claude']}",
                f"{out['support_gpt4o']} / {out['support_claude']}",
                f"{out['utility_gpt4o']} / {out['utility_claude']}",
            ]
        )
    write_csv(TABLES / "table6_answer_quality_diagnostics.csv", out_rows, fields)
    latex_table(
        TABLES / "table6_answer_quality_diagnostics.tex",
        ["Method", "Grd. (G/C)", "Supp. (G/C)", "Util. (G/C)"],
        tex_rows,
        "lccc",
    )


def make_table7(values: dict) -> None:
    src = SOURCE / "table5_v5_trust_advantaged_slice_gpt4o_20260508.csv"
    dst = TABLES / "table7_top10_trust_advantaged_slice.csv"
    shutil.copyfile(src, dst)
    rows = read_csv(dst)
    baseline = values["table_7_top10_trust_advantaged_slice"]
    expected_rows = baseline["rows"]
    for got, expected in zip(rows[:10], expected_rows):
        if got["qid"] != expected["qid"]:
            raise AssertionError(f"table7 qid mismatch: got {got['qid']}, expected {expected['qid']}")
        assert_close(f"table7.{got['qid']}.delta_ht_cite", got["delta_ht_cite"], expected["delta_ht_cite"], 4)
        assert_close(f"table7.{got['qid']}.delta_cite_coverage", got["delta_cite_cov"], expected["delta_cite_coverage"], 4)
    tex_rows = [
        [
            row["topic"],
            row["qid"],
            row["query_summary"],
            row["delta_ht_cite"],
            row["delta_cite_cov"],
            row["support_graph"],
            row["support_anchorrag"],
        ]
        for row in rows
    ]
    latex_table(
        TABLES / "table7_top10_trust_advantaged_slice.tex",
        ["Topic", "QID", "Query Summary", "$\\Delta$ HT Cite", "$\\Delta$ Cite Cov.", "Support Graph", "Support AnchorRAG"],
        tex_rows,
        "llp{4.5cm}cccc",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["all", "table4", "table5", "table6", "table7"], default="all")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    values = load_values()
    makers = {
        "table4": make_table4,
        "table5": make_table5,
        "table6": make_table6,
        "table7": make_table7,
    }
    if args.table == "all":
        for maker in makers.values():
            maker(values)
    else:
        makers[args.table](values)
    print(f"[make_tables] wrote cached table artifacts to {TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
