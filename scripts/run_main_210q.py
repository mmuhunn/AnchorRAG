#!/usr/bin/env python3.8
"""Config-driven live runner for the main 210-query AnchorRAG experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/runs/main_210q_live.yaml"
GENERATION_MODULE = REPO_ROOT / "src/anchorrag/exp4/run_exp4_allsides.py"
EVALUATION_MODULE = REPO_ROOT / "src/anchorrag/exp4/eval_exp4_allsides.py"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must parse to a mapping: {}".format(path))
    return data


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import module from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_topics(config: Dict[str, Any], requested: Optional[List[str]]) -> List[Dict[str, Any]]:
    topics = config.get("topics") or []
    if requested is None:
        return topics
    wanted = set(requested)
    selected = [topic for topic in topics if topic.get("key") in wanted]
    missing = sorted(wanted - {topic.get("key") for topic in selected})
    if missing:
        raise ValueError("unknown topic(s): {}".format(", ".join(missing)))
    return selected


def selected_methods(config: Dict[str, Any], requested: Optional[List[str]]) -> List[str]:
    methods = list((config.get("methods") or {}).get("include") or [])
    if requested is None:
        return methods
    missing = sorted(set(requested) - set(methods))
    if missing:
        raise ValueError("requested method(s) not in config: {}".format(", ".join(missing)))
    return requested


def output_paths(config: Dict[str, Any], run_id: Optional[str]) -> Dict[str, Path]:
    outputs = config.get("outputs") or {}
    root = repo_path(outputs.get("root", "outputs/main_210q_live"))
    rid = run_id or outputs.get("run_id") or "main_210q_live"
    run_root = root / rid
    generation_jsonl = run_root / outputs.get("generation_jsonl", "generation/exp4_eval_data_live_7methods_210q.jsonl")
    cache_dir = run_root / outputs.get("cache_dir", "cache/embeddings")
    judged_dir = run_root / outputs.get("judged_dir", "judging")
    return {
        "run_root": run_root,
        "generation_jsonl": generation_jsonl,
        "cache_dir": cache_dir,
        "judged_dir": judged_dir,
        "manifest": run_root / "run_manifest.json",
    }


def validate_inputs(config: Dict[str, Any], topics: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for path in [GENERATION_MODULE, EVALUATION_MODULE]:
        if not path.exists():
            issues.append("missing pipeline module: {}".format(path.relative_to(REPO_ROOT)))

    for topic in topics:
        key = topic.get("key", "<missing>")
        query_path = repo_path(topic.get("query_file", ""))
        graph_path = repo_path(topic.get("graph_file", ""))
        if not query_path.exists():
            issues.append("missing query file for {}: {}".format(key, display_path(query_path)))
        if not graph_path.exists():
            issues.append("missing graph file for {}: {}".format(key, display_path(graph_path)))
            continue
        try:
            query_count = len(json.loads(query_path.read_text(encoding="utf-8")))
        except Exception as exc:
            issues.append("could not parse query file for {}: {}".format(key, exc))
            query_count = None
        if query_count is not None and query_count != 30:
            issues.append("expected 30 queries for {}, found {}".format(key, query_count))
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append("could not parse graph file for {}: {}".format(key, exc))
            continue
        if not isinstance(graph, dict) or "nodes" not in graph or "links" not in graph:
            issues.append("graph for {} must contain nodes and links".format(key))
    return issues


def graph_hydration_status(topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report how many graph nodes carry raw article ``text`` (see hydrate_graphs.py)."""
    per_topic: Dict[str, str] = {}
    unhydrated: List[str] = []
    for topic in topics:
        key = topic.get("key", "<missing>")
        graph_path = repo_path(topic.get("graph_file", ""))
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception:
            per_topic[key] = "unreadable"
            unhydrated.append(key)
            continue
        nodes = graph.get("nodes", [])
        with_text = sum(1 for node in nodes if node.get("text"))
        per_topic[key] = "{}/{}".format(with_text, len(nodes))
        if nodes and with_text == 0:
            unhydrated.append(key)
    return {"per_topic": per_topic, "fully_unhydrated": unhydrated}


def hydrate_topic_configs(module: Any, topics: List[Dict[str, Any]]) -> None:
    configured: Dict[str, Dict[str, Any]] = {}
    for topic in topics:
        key = str(topic["key"])
        configured[key] = {
            "display_name": topic.get("display_name", key),
            "graph_path": str(repo_path(topic["graph_file"])),
            "queries": json.loads(repo_path(topic["query_file"]).read_text(encoding="utf-8")),
        }
    module.TOPIC_CONFIGS.clear()
    module.TOPIC_CONFIGS.update(configured)


def make_generation_args(
    config: Dict[str, Any],
    topics: List[Dict[str, Any]],
    methods: List[str],
    paths: Dict[str, Path],
    limit_queries: Optional[int],
    refresh_cache: bool,
) -> argparse.Namespace:
    retrieval = config.get("retrieval") or {}
    models = config.get("models") or {}
    return argparse.Namespace(
        topics=[topic["key"] for topic in topics],
        methods=methods,
        output_jsonl=str(paths["generation_jsonl"]),
        cache_dir=str(paths["cache_dir"]),
        refresh_cache=refresh_cache,
        top_k=int(retrieval.get("top_k", 10)),
        candidate_pool=int(retrieval.get("candidate_pool", 50)),
        limit_queries=limit_queries,
        embedding_model=models.get("embedding", "text-embedding-3-small"),
        generator_model=models.get("generator", "gpt-4o-2024-08-06"),
        planner_model=models.get("planner", models.get("generator", "gpt-4o-2024-08-06")),
        mmr_lambda=float(retrieval.get("mmr_lambda", 0.70)),
        graph_alpha=float(retrieval.get("graph_alpha", 0.70)),
        graph_gamma=float(retrieval.get("graph_gamma", 0.25)),
        graph_delta=float(retrieval.get("graph_delta", 0.05)),
        trust_alpha=float(retrieval.get("trust_alpha", 0.70)),
        trust_beta=float(retrieval.get("trust_beta", 0.30)),
        trust_div_rho=float(retrieval.get("trust_div_rho", 0.80)),
        trust_div_mode=retrieval.get("trust_div_mode", "community"),
        card_gamma=float(retrieval.get("card_gamma", 0.65)),
        anchor_lambda_structure=float(retrieval.get("anchor_lambda_structure", 0.45)),
        anchor_lambda_relevance=float(retrieval.get("anchor_lambda_relevance", 0.35)),
        anchor_lambda_support=float(retrieval.get("anchor_lambda_support", 0.20)),
        anchor_high_trust_threshold=float(retrieval.get("anchor_high_trust_threshold", 0.80)),
    )


def judge_names(requested: str) -> List[str]:
    if requested == "both":
        return ["primary", "robustness"]
    return [requested]


def make_evaluation_args(
    config: Dict[str, Any],
    methods: List[str],
    paths: Dict[str, Path],
    judge_name: str,
    skip_pairwise_override: Optional[bool],
) -> argparse.Namespace:
    judging = config.get("judging") or {}
    judge = judging.get(judge_name) or {}
    if not judge:
        raise ValueError("missing judging config for {}".format(judge_name))

    reference_method = (config.get("methods") or {}).get("reference_method", "trustrag_anchor")
    pairwise_baselines = [method for method in methods if method != reference_method]
    judge_dir = paths["judged_dir"] / judge_name
    skip_pairwise = bool(judge.get("skip_pairwise", False))
    if skip_pairwise_override is not None:
        skip_pairwise = skip_pairwise_override

    return argparse.Namespace(
        input=str(paths["generation_jsonl"]),
        output_jsonl=str(judge_dir / "eval_main_210q_{}.jsonl".format(judge_name)),
        output_csv=str(judge_dir / "summary_main_210q_{}.csv".format(judge_name)),
        pairwise_output_jsonl=str(judge_dir / "pairwise_main_210q_{}.jsonl".format(judge_name)),
        pairwise_output_csv=str(judge_dir / "pairwise_summary_main_210q_{}.csv".format(judge_name)),
        methods=methods,
        model=judge.get("model", "gpt-4o-2024-08-06"),
        support_model=judge.get("support_model", judge.get("model", "gpt-4o-2024-08-06")),
        pairwise_model=judge.get("pairwise_model", judge.get("model", "gpt-4o-2024-08-06")),
        high_trust_threshold=float(judging.get("high_trust_threshold", 0.75)),
        backbone_k=int(judging.get("backbone_k", 3)),
        pairwise_target=reference_method,
        pairwise_baselines=pairwise_baselines,
        max_pairwise_evidence_docs=int(judging.get("max_pairwise_evidence_docs", 6)),
        skip_pairwise=skip_pairwise,
    )


def write_run_manifest(
    config: Dict[str, Any],
    topics: List[Dict[str, Any]],
    methods: List[str],
    paths: Dict[str, Path],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": display_path(args.config),
        "stage": args.stage,
        "judge": args.judge,
        "topics": [topic["key"] for topic in topics],
        "methods": methods,
        "models": config.get("models", {}),
        "retrieval": config.get("retrieval", {}),
        "outputs": {key: display_path(value) for key, value in paths.items()},
        "resume_note": "Generation resumes by (topic, query_id); evaluation resumes by judged output files.",
    }
    paths["run_root"].mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_plan(
    config: Dict[str, Any],
    topics: List[Dict[str, Any]],
    methods: List[str],
    paths: Dict[str, Path],
    args: argparse.Namespace,
) -> None:
    required_env = []
    if args.stage in ("generate", "all"):
        required_env.append("OPENAI_API_KEY")
    if args.stage in ("evaluate", "all"):
        for judge_name in judge_names(args.judge):
            judge = (config.get("judging") or {}).get(judge_name) or {}
            if judge.get("provider") == "openai":
                required_env.append("OPENAI_API_KEY")
            if judge.get("provider") == "anthropic":
                required_env.append("ANTHROPIC_API_KEY")
    required_env = sorted(set(required_env))
    env_status = {name: bool(os.environ.get(name)) for name in required_env}

    hydration = graph_hydration_status(topics)
    plan = {
        "config": display_path(args.config),
        "stage": args.stage,
        "judge": args.judge,
        "topics": [topic["key"] for topic in topics],
        "methods": methods,
        "output_root": display_path(paths["run_root"]),
        "generation_jsonl": display_path(paths["generation_jsonl"]),
        "cache_dir": display_path(paths["cache_dir"]),
        "required_env_present": env_status,
        "limit_queries": args.limit_queries,
        "refresh_cache": args.refresh_cache,
        "graph_text_hydration": hydration["per_topic"],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if hydration["fully_unhydrated"] and args.stage in ("generate", "all"):
        print(
            "[run_main_210q] WARNING: graphs missing raw article text for: {}\n"
            "  run `python3.8 scripts/hydrate_graphs.py --subset-dir <AllSides_Qbias/data/subsets>` "
            "before live generation.".format(", ".join(hydration["fully_unhydrated"]))
        )


def run_generation(
    config: Dict[str, Any],
    topics: List[Dict[str, Any]],
    methods: List[str],
    paths: Dict[str, Path],
    limit_queries: Optional[int],
    refresh_cache: bool,
) -> None:
    module = load_module("anchorrag_exp4_generation", GENERATION_MODULE)
    hydrate_topic_configs(module, topics)
    generation_args = make_generation_args(config, topics, methods, paths, limit_queries, refresh_cache)
    module.parse_args = lambda: generation_args
    module.main()


def run_evaluation(
    config: Dict[str, Any],
    methods: List[str],
    paths: Dict[str, Path],
    judge_name: str,
    skip_pairwise_override: Optional[bool],
) -> None:
    if not paths["generation_jsonl"].exists():
        raise FileNotFoundError(
            "generation output does not exist yet: {}".format(display_path(paths["generation_jsonl"]))
        )
    module = load_module("anchorrag_exp4_evaluation_{}".format(judge_name), EVALUATION_MODULE)
    evaluation_args = make_evaluation_args(config, methods, paths, judge_name, skip_pairwise_override)
    module.parse_args = lambda: evaluation_args
    module.main()


def parse_cli(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate the main 210-query live reproduction path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["generate", "evaluate", "all"], default="all")
    parser.add_argument("--judge", choices=["primary", "robustness", "both"], default="primary")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--topics", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--include-pairwise", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if args.skip_pairwise and args.include_pairwise:
        parser.error("--skip-pairwise and --include-pairwise are mutually exclusive")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_cli(argv)
    config = load_yaml(args.config)
    topics = selected_topics(config, args.topics)
    methods = selected_methods(config, args.methods)
    paths = output_paths(config, args.run_id)

    issues = validate_inputs(config, topics)
    if issues:
        print("[run_main_210q] input validation failed", file=sys.stderr)
        for issue in issues:
            print("  - {}".format(issue), file=sys.stderr)
        return 1

    print_plan(config, topics, methods, paths, args)
    if args.dry_run:
        print("[run_main_210q] dry run complete; no API calls were made.")
        return 0

    skip_pairwise_override: Optional[bool] = None
    if args.skip_pairwise:
        skip_pairwise_override = True
    if args.include_pairwise:
        skip_pairwise_override = False

    if args.stage in ("generate", "all"):
        unhydrated = graph_hydration_status(topics)["fully_unhydrated"]
        if unhydrated:
            print(
                "[run_main_210q] cannot generate: graphs have no raw article text for: {}\n"
                "  run `python3.8 scripts/hydrate_graphs.py --subset-dir <AllSides_Qbias/data/subsets>` "
                "first.".format(", ".join(unhydrated)),
                file=sys.stderr,
            )
            return 1

    write_run_manifest(config, topics, methods, paths, args)

    if args.stage in ("generate", "all"):
        run_generation(config, topics, methods, paths, args.limit_queries, args.refresh_cache)

    if args.stage in ("evaluate", "all"):
        for judge_name in judge_names(args.judge):
            run_evaluation(config, methods, paths, judge_name, skip_pairwise_override)

    print("[run_main_210q] completed stage={}".format(args.stage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
