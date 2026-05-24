#!/usr/bin/env python3.8
"""Audit the public AnchorRAG artifact package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - handled for fresh environments.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "artifacts/paper_2026_cikm/MANIFEST.csv"
PAPER_VALUES = REPO_ROOT / "artifacts/paper_2026_cikm/paper_values.yaml"
TABLES = REPO_ROOT / "artifacts/paper_2026_cikm/tables"
FIGURES = REPO_ROOT / "artifacts/paper_2026_cikm/figures"
LIVE_CONFIG = REPO_ROOT / "configs/runs/main_210q_live.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("rb") as handle:
            return str(sum(1 for line in handle if line.strip()))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return str(max(sum(1 for _ in csv.reader(handle)) - 1, 0))
    return ""


def audit_manifest() -> list[str]:
    issues: list[str] = []
    if not MANIFEST.exists():
        return [f"missing manifest: {MANIFEST}"]

    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        rel = row["path"]
        path = REPO_ROOT / rel
        if not path.exists():
            issues.append(f"missing path: {rel}")
            continue

        expected_type = row.get("object_type", "")
        actual_type = "directory" if path.is_dir() else "file"
        if expected_type and expected_type != actual_type:
            issues.append(f"type mismatch for {rel}: manifest={expected_type}, actual={actual_type}")

        if path.is_file():
            expected_sha = row.get("sha256", "")
            if expected_sha and sha256(path) != expected_sha:
                issues.append(f"sha256 mismatch: {rel}")

            expected_rows = row.get("row_count", "")
            actual_rows = row_count(path)
            if expected_rows and actual_rows and expected_rows != actual_rows:
                issues.append(f"row_count mismatch for {rel}: manifest={expected_rows}, actual={actual_rows}")

        if row.get("inventory_status") != "ok":
            issues.append(f"inventory_status is not ok for {rel}: {row.get('inventory_status')}")

    print(f"[audit] manifest rows: {len(rows)}")
    return issues


def audit_paper_values() -> list[str]:
    if not PAPER_VALUES.exists():
        return [f"missing paper values: {PAPER_VALUES}"]
    if yaml is None:
        return ["pyyaml is not installed; run `pip install -r requirements.txt`"]

    data = yaml.safe_load(PAPER_VALUES.read_text())
    issues: list[str] = []
    current_pdf = data.get("paper", {}).get("current_pdf")
    if current_pdf and not (REPO_ROOT / current_pdf).exists():
        issues.append(f"paper_values current_pdf path does not exist: {current_pdf}")

    table4 = data.get("table_4_trust_sensitive_evidence_use", {})
    if table4.get("n_queries") != 210:
        issues.append("paper_values table_4 n_queries is not 210")

    print("[audit] paper_values.yaml parsed")
    return issues


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def paper_method(method: str) -> str:
    return "anchorrag" if method == "trustrag_anchor" else method


def check_float(issues: list[str], name: str, got: str, expected, digits: int) -> None:
    if round(float(got), digits) != round(float(expected), digits):
        issues.append(f"{name}: got {got}, expected {expected}")


def audit_generated_tables(data: dict) -> list[str]:
    issues: list[str] = []

    table4 = TABLES / "table4_trust_sensitive_evidence_use.csv"
    if table4.exists():
        baseline = data["table_4_trust_sensitive_evidence_use"]["rows"]
        for row in load_csv(table4):
            expected = baseline[paper_method(row["method"])]
            for field in [
                "citation_coverage",
                "high_trust_citation_rate",
                "anchor_utilization_rate",
                "backbone_trust_rate",
                "avg_s_trust_at_k",
            ]:
                check_float(issues, f"table4.{row['method']}.{field}", row[field], expected[field], 4)

    table5 = TABLES / "table5_mbfc_external_rescoring.csv"
    if table5.exists():
        baseline = data["table_5_mbfc_external_rescoring"]["rows"]
        for row in load_csv(table5):
            expected = baseline[paper_method(row["method"])]
            for field in [
                "mbfc_high_trust_citation_rate",
                "mbfc_avg_factual_all",
                "mbfc_backbone_high_trust_rate",
                "mbfc_backbone_avg_factual",
                "mbfc_map_coverage",
            ]:
                check_float(issues, f"table5.{row['method']}.{field}", row[field], expected[field], 4)

    table6 = TABLES / "table6_answer_quality_diagnostics.csv"
    if table6.exists():
        baseline = data["table_6_answer_quality_diagnostics"]["rows"]
        for row in load_csv(table6):
            expected = baseline[paper_method(row["method"])]
            for field in [
                "groundedness_gpt4o",
                "groundedness_claude",
                "support_gpt4o",
                "support_claude",
                "utility_gpt4o",
                "utility_claude",
            ]:
                check_float(issues, f"table6.{row['method']}.{field}", row[field], expected[field], 3)

    table7 = TABLES / "table7_top10_trust_advantaged_slice.csv"
    if table7.exists():
        rows = load_csv(table7)
        baseline_rows = data["table_7_top10_trust_advantaged_slice"]["rows"]
        for got, expected in zip(rows[:10], baseline_rows):
            if got["qid"] != expected["qid"]:
                issues.append(f"table7 qid mismatch: got {got['qid']}, expected {expected['qid']}")
            check_float(issues, f"table7.{got['qid']}.delta_ht_cite", got["delta_ht_cite"], expected["delta_ht_cite"], 4)
            check_float(
                issues,
                f"table7.{got['qid']}.delta_cite_coverage",
                got["delta_cite_cov"],
                expected["delta_cite_coverage"],
                4,
            )

    table8 = TABLES / "table8_ablation.csv"
    if table8.exists():
        baseline = data["table_8_ablation"]["rows"]
        for row in load_csv(table8):
            method = paper_method(row["method"])
            expected = baseline[method]
            for field in [
                "groundedness",
                "support",
                "high_trust_citation_rate",
                "utility",
                "backbone_trust",
                "trust_weighted_claim_support",
                "citation_coverage",
            ]:
                check_float(issues, f"table8.{row['method']}.{field}", row[field], expected[field], 3)
            check_float(issues, f"table8.{row['method']}.claims", row["claims"], expected["claims"], 2)

    if any(path.exists() for path in TABLES.glob("table*.csv")):
        print("[audit] generated cached tables match paper_values.yaml")
    return issues


def audit_generated_figures(data: dict) -> list[str]:
    issues: list[str] = []
    figure_data = FIGURES / "figure2_source_prior_mbfc_data.csv"
    if figure_data.exists():
        rows = load_csv(figure_data)
        values = data["figure_2_source_prior_mbfc_validation"]
        covered = sum(int(float(row["total_count"])) for row in rows)
        if len(rows) != values["matched_sources"]:
            issues.append(f"figure2 matched sources mismatch: got {len(rows)}, expected {values['matched_sources']}")
        if covered != values["covered_documents"]:
            issues.append(f"figure2 covered documents mismatch: got {covered}, expected {values['covered_documents']}")
        print("[audit] generated Figure 2 data matches paper_values.yaml")
    return issues


def audit_live_inputs() -> list[str]:
    issues: list[str] = []
    if yaml is None:
        return issues
    if not LIVE_CONFIG.exists():
        return [f"missing live config: {LIVE_CONFIG}"]

    config = yaml.safe_load(LIVE_CONFIG.read_text())
    topics = config.get("topics") or []
    if len(topics) != 7:
        issues.append(f"live config topic count is {len(topics)}, expected 7")

    total_queries = 0
    for topic in topics:
        key = topic.get("key", "<missing>")
        query_file = REPO_ROOT / topic.get("query_file", "")
        graph_file = REPO_ROOT / topic.get("graph_file", "")
        if not query_file.exists():
            issues.append(f"missing live query file for {key}: {query_file}")
            continue
        if not graph_file.exists():
            issues.append(f"missing live graph file for {key}: {graph_file}")
            continue

        try:
            queries = json.loads(query_file.read_text())
        except Exception as exc:
            issues.append(f"could not parse live query file for {key}: {exc}")
            queries = []
        if len(queries) != 30:
            issues.append(f"live query count for {key} is {len(queries)}, expected 30")
        total_queries += len(queries)

        try:
            graph = json.loads(graph_file.read_text())
        except Exception as exc:
            issues.append(f"could not parse live graph file for {key}: {exc}")
            continue
        if not isinstance(graph, dict) or "nodes" not in graph or "links" not in graph:
            issues.append(f"live graph for {key} must contain nodes and links")

    methods = (config.get("methods") or {}).get("include") or []
    if len(methods) != 7:
        issues.append(f"live method count is {len(methods)}, expected 7")
    if total_queries != 210:
        issues.append(f"live total query count is {total_queries}, expected 210")

    if not issues:
        print("[audit] live config, queries, and graph inputs parsed")
    return issues


def audit_supporting_experiments(data: dict) -> list[str]:
    issues: list[str] = []
    support = REPO_ROOT / "artifacts/paper_2026_cikm/supporting_experiments"

    required_paths = [
        support / "ablation_210q/source_summaries",
        support / "ablation_210q/source_judgments",
        support / "ablation_210q/multi_anchor_note.csv",
        support / "sensitivity/compact_sensitivity_values.csv",
        support / "sensitivity/appendix_artifacts/main_compact_table.tex",
    ]
    for path in required_paths:
        if not path.exists():
            issues.append(f"missing supporting experiment artifact: {path.relative_to(REPO_ROOT)}")

    summary_count = len(list((support / "ablation_210q/source_summaries").glob("*.csv")))
    judgment_count = len(list((support / "ablation_210q/source_judgments").glob("*.jsonl")))
    if summary_count != 7:
        issues.append(f"ablation source summary count is {summary_count}, expected 7")
    if judgment_count != 7:
        issues.append(f"ablation source judgment count is {judgment_count}, expected 7")

    note_path = support / "ablation_210q/multi_anchor_note.csv"
    if note_path.exists():
        expected_note = data["table_8_ablation"]["multi_anchor_note"]
        note_rows = {row["metric"]: row["value"] for row in load_csv(note_path)}
        for field, expected in expected_note.items():
            if field not in note_rows:
                issues.append(f"multi_anchor_note missing metric: {field}")
                continue
            check_float(issues, f"multi_anchor_note.{field}", note_rows[field], expected, 4)

    sensitivity = support / "sensitivity/compact_sensitivity_values.csv"
    if sensitivity.exists():
        rows = load_csv(sensitivity)
        expected_count = sum(len(group) for group in data["section_5_3_sensitivity_text_values"].values())
        if len(rows) != expected_count:
            issues.append(f"sensitivity compact row count is {len(rows)}, expected {expected_count}")

    if not issues:
        print("[audit] supporting experiment artifacts parsed")
    return issues


def main() -> int:
    issues = []
    issues.extend(audit_manifest())
    issues.extend(audit_paper_values())

    if yaml is not None and PAPER_VALUES.exists():
        data = yaml.safe_load(PAPER_VALUES.read_text())
        issues.extend(audit_generated_tables(data))
        issues.extend(audit_generated_figures(data))
        issues.extend(audit_supporting_experiments(data))
        issues.extend(audit_live_inputs())

    if issues:
        print("[audit] FAILED")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("[audit] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
