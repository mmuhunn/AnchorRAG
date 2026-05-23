#!/usr/bin/env python3.8
"""Regenerate cached supporting-experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/paper_2026_cikm"
TABLES = ARTIFACT / "tables"
SUPPORT = ARTIFACT / "supporting_experiments"
ABLATION = SUPPORT / "ablation_210q"
SENSITIVITY = SUPPORT / "sensitivity"
PAPER_VALUES = ARTIFACT / "paper_values.yaml"

TOPIC_RUNS = {
    "abortion": "20260512",
    "climate_change": "20260512",
    "economy_jobs": "20260512",
    "free_speech": "20260512",
    "gun_control": "20260511",
    "immigration": "20260511",
    "voting_rights_and_voter_fraud": "20260512",
}

METHOD_ORDER = [
    "vanilla_rag",
    "wo_trust",
    "wo_anchor",
    "wo_diversity",
    "trustrag_anchor",
]
TOP2_METHOD = "trustrag_anchor_top2"

DISPLAY = {
    "vanilla_rag": "vanilla rag",
    "wo_trust": "w/o trust",
    "wo_anchor": "w/o anchor",
    "wo_diversity": "w/o diversity",
    "trustrag_anchor": "AnchorRAG",
}

PAPER_KEY = {
    "trustrag_anchor": "anchorrag",
}

TABLE8_FIELDS = [
    "method",
    "groundedness",
    "support",
    "high_trust_citation_rate",
    "utility",
    "backbone_trust",
    "trust_weighted_claim_support",
    "claims",
    "citation_coverage",
]

SUMMARY_FIELD_MAP = {
    "groundedness": "groundedness",
    "support": "evidence_support_rate",
    "high_trust_citation_rate": "high_trust_citation_rate",
    "utility": "response_utility",
    "backbone_trust": "backbone_trust_rate",
    "trust_weighted_claim_support": "trust_weighted_claim_support",
    "citation_coverage": "citation_coverage",
}


def load_values() -> dict:
    return yaml.safe_load(PAPER_VALUES.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, pattern: str) -> str:
    formatted = str(Decimal(str(value)).quantize(Decimal(pattern), rounding=ROUND_HALF_UP))
    if formatted.startswith("0") and pattern.startswith("0."):
        return formatted[1:]
    return formatted


def fmt3(value: object) -> str:
    return fmt(value, "0.001")


def fmt4(value: object) -> str:
    return fmt(value, "0.0001")


def fmt2(value: object) -> str:
    return fmt(value, "0.01")


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


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


def public_key(method: str) -> str:
    return PAPER_KEY.get(method, method)


def assert_close(name: str, got: object, expected: object, digits: int) -> None:
    if round(float(got), digits) != round(float(expected), digits):
        raise AssertionError(f"{name}: got {got}, expected {expected}")


def summary_path(topic: str, run_tag: str) -> Path:
    return ABLATION / "source_summaries" / f"ablation_summary_report_{topic}_30q_{run_tag}.csv"


def judgment_path(topic: str, run_tag: str) -> Path:
    return ABLATION / "source_judgments" / f"ablation_judged_results_{topic}_30q_{run_tag}.jsonl"


def load_topic_summary(topic: str, run_tag: str) -> dict[str, dict[str, str]]:
    path = summary_path(topic, run_tag)
    if not path.exists():
        raise FileNotFoundError(path)
    return {row["method"]: row for row in read_csv(path) if row["topic"] == "all"}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def aggregate_claim_counts() -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for topic, run_tag in TOPIC_RUNS.items():
        path = judgment_path(topic, run_tag)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            method = row.get("method")
            if method in METHOD_ORDER:
                values[method].append(float(row.get("evaluable_sentences", 0) or 0))
    return {method: sum(values[method]) / len(values[method]) for method in METHOD_ORDER}


def aggregate_table8() -> tuple[list[dict[str, object]], dict[str, float]]:
    weighted = {
        method: {field: 0.0 for field in SUMMARY_FIELD_MAP}
        for method in METHOD_ORDER + [TOP2_METHOD]
    }
    counts = {method: 0.0 for method in METHOD_ORDER + [TOP2_METHOD]}

    for topic, run_tag in TOPIC_RUNS.items():
        rows = load_topic_summary(topic, run_tag)
        for method in METHOD_ORDER + [TOP2_METHOD]:
            row = rows[method]
            n = float(row["n"])
            counts[method] += n
            for public_field, source_field in SUMMARY_FIELD_MAP.items():
                weighted[method][public_field] += float(row[source_field]) * n

    claims = aggregate_claim_counts()
    out_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        out: dict[str, object] = {"method": method}
        for field in SUMMARY_FIELD_MAP:
            out[field] = weighted[method][field] / counts[method]
        out["claims"] = claims[method]
        out_rows.append(out)

    top2 = {
        "support_single_anchor": weighted["trustrag_anchor"]["support"] / counts["trustrag_anchor"],
        "support_multi_anchor": weighted[TOP2_METHOD]["support"] / counts[TOP2_METHOD],
        "trust_weighted_claim_support_single_anchor": weighted["trustrag_anchor"]["trust_weighted_claim_support"]
        / counts["trustrag_anchor"],
        "trust_weighted_claim_support_multi_anchor": weighted[TOP2_METHOD]["trust_weighted_claim_support"]
        / counts[TOP2_METHOD],
    }
    return out_rows, top2


def make_table8(values: dict) -> None:
    rows, top2 = aggregate_table8()
    baseline = values["table_8_ablation"]["rows"]
    out_rows: list[dict[str, object]] = []
    tex_rows: list[list[object]] = []

    for row in rows:
        method = str(row["method"])
        out = {
            "method": method,
            "groundedness": fmt3(row["groundedness"]),
            "support": fmt3(row["support"]),
            "high_trust_citation_rate": fmt3(row["high_trust_citation_rate"]),
            "utility": fmt3(row["utility"]),
            "backbone_trust": fmt3(row["backbone_trust"]),
            "trust_weighted_claim_support": fmt3(row["trust_weighted_claim_support"]),
            "claims": fmt2(row["claims"]),
            "citation_coverage": fmt3(row["citation_coverage"]),
        }
        expected = baseline[public_key(method)]
        for field in TABLE8_FIELDS[1:]:
            digits = 2 if field == "claims" else 3
            assert_close(f"table8.{method}.{field}", out[field], expected[field], digits)
        out_rows.append(out)
        tex_rows.append([DISPLAY[method], *[out[field] for field in TABLE8_FIELDS[1:]]])

    write_csv(TABLES / "table8_ablation.csv", out_rows, TABLE8_FIELDS)
    latex_table(
        TABLES / "table8_ablation.tex",
        ["Method", "Ground.", "Supp.", "HT Cite", "Util.", "Backbone Trust", "TW Claim Supp.", "Claims", "Cite Cov."],
        tex_rows,
        "lcccccccc",
    )

    note = values["table_8_ablation"]["multi_anchor_note"]
    note_rows: list[dict[str, object]] = []
    for field in [
        "support_single_anchor",
        "support_multi_anchor",
        "trust_weighted_claim_support_single_anchor",
        "trust_weighted_claim_support_multi_anchor",
    ]:
        formatted = fmt4(top2[field])
        assert_close(f"table8.multi_anchor_note.{field}", formatted, note[field], 4)
        note_rows.append({"metric": field, "value": formatted})
    write_csv(ABLATION / "multi_anchor_note.csv", note_rows, ["metric", "value"])


def flatten_sensitivity(values: dict) -> list[dict[str, object]]:
    section = values["section_5_3_sensitivity_text_values"]
    rows: list[dict[str, object]] = []
    for group, metrics in section.items():
        for metric, value in metrics.items():
            if isinstance(value, list):
                rendered = ";".join(str(item) for item in value)
            else:
                rendered = value
            rows.append({"group": group, "metric": metric, "value": rendered})
    return rows


def make_sensitivity(values: dict) -> None:
    rows = flatten_sensitivity(values)
    write_csv(SENSITIVITY / "compact_sensitivity_values.csv", rows, ["group", "metric", "value"])
    expected_count = sum(len(metrics) for metrics in values["section_5_3_sensitivity_text_values"].values())
    if len(rows) != expected_count:
        raise AssertionError(f"sensitivity compact row count mismatch: got {len(rows)}, expected {expected_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["all", "table8", "sensitivity"], default="all")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    values = load_values()
    if args.target in ("all", "table8"):
        make_table8(values)
    if args.target in ("all", "sensitivity"):
        make_sensitivity(values)
    print("[make_supporting] wrote supporting artifacts under {}".format(SUPPORT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
