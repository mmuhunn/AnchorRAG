#!/usr/bin/env python3.8
"""Re-inject raw article body text into the released topic graphs.

The public release ships derived topic graphs with the raw third-party article
``text`` body removed (see ``data/graphs/README.md``). The structure, derived
scores, source names, stance labels, and article titles are kept. Live
reproduction needs the article body, so this script reads the AllSides/Qbias
subset CSVs that you obtain separately and writes the ``text`` field back into
each node, matching on the article ``title``.

Usage:
    python3.8 scripts/hydrate_graphs.py --subset-dir /path/to/AllSides_Qbias/data/subsets
    python3.8 scripts/hydrate_graphs.py --subset-dir <dir> --check   # report coverage only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = REPO_ROOT / "data/graphs"

TOPICS = [
    "abortion",
    "climate_change",
    "economy_jobs",
    "free_speech",
    "gun_control",
    "immigration",
    "voting_rights_and_voter_fraud",
]


def load_title_text_map(subset_dir: Path, topic: str) -> dict[str, str]:
    """Build a {title: text} map from the best-matching subset CSV for a topic."""
    candidates = sorted(glob.glob(str(subset_dir / f"{topic}_subset*.csv")))
    if not candidates:
        return {}
    best: dict[str, str] = {}
    for path in candidates:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mapping = {r["title"]: (r.get("text") or "") for r in rows if r.get("title")}
        if len(mapping) > len(best):
            best = mapping
    return best


def hydrate_topic(graph_path: Path, title_text: dict[str, str], write: bool) -> tuple[int, int]:
    graph = json.loads(graph_path.read_text())
    nodes = graph.get("nodes", [])
    matched = 0
    for node in nodes:
        title = node.get("title", "")
        if title in title_text and title_text[title]:
            node["text"] = title_text[title]
            matched += 1
    if write and matched:
        graph_path.write_text(json.dumps(graph, ensure_ascii=False))
    return matched, len(nodes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-dir",
        required=True,
        help="Directory containing AllSides/Qbias <topic>_subset.csv files.",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(GRAPHS_DIR),
        help="Directory with released *_scored_graph.json files (default: data/graphs).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report match coverage without writing the graphs.",
    )
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir).expanduser().resolve()
    graphs_dir = Path(args.graphs_dir).expanduser().resolve()
    if not subset_dir.is_dir():
        print(f"[hydrate] subset dir not found: {subset_dir}")
        return 1

    write = not args.check
    total_matched = 0
    total_nodes = 0
    incomplete = []
    for topic in TOPICS:
        graph_path = graphs_dir / f"{topic}_scored_graph.json"
        if not graph_path.exists():
            print(f"[hydrate] missing graph: {graph_path}")
            return 1
        title_text = load_title_text_map(subset_dir, topic)
        if not title_text:
            print(f"[hydrate] no subset CSV found for topic '{topic}' in {subset_dir}")
            return 1
        matched, n = hydrate_topic(graph_path, title_text, write)
        total_matched += matched
        total_nodes += n
        flag = "" if matched == n else "  <-- INCOMPLETE"
        if matched != n:
            incomplete.append(topic)
        print(f"[hydrate] {topic}: matched {matched}/{n}{flag}")

    action = "checked" if args.check else "wrote"
    print(f"[hydrate] {action} {total_matched}/{total_nodes} node bodies across {len(TOPICS)} graphs")
    if incomplete:
        print(f"[hydrate] WARNING: incomplete coverage for: {', '.join(incomplete)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
