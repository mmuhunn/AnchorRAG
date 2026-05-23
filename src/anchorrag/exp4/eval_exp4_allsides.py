import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

try:
    import nltk as _nltk
    _nltk.download("punkt_tab", quiet=True)
    _SENT_TOKENIZE = _nltk.sent_tokenize
except Exception:
    _SENT_TOKENIZE = None

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
    OPENAI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import path depends on local env
    OpenAI = Any  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc

try:
    import anthropic as anthropic_sdk
    ANTHROPIC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    anthropic_sdk = None  # type: ignore[assignment]
    ANTHROPIC_IMPORT_ERROR = exc

# Module-level client registry — populated in main() before any judge calls.
_CLIENTS: Dict[str, Any] = {"openai": None, "anthropic": None}


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_METHODS = [
    "vanilla_rag",
    "mmr_rag",
    "graph_retrieval",
    "r2ag",
    "reclaim",
    "grag",
    "trustrag_anchor",
]
AVAILABLE_METHODS = DEFAULT_METHODS + [
    "trustrag_anchor_top2",
    "claim_graph_only",
    "claim_graph_verifier",
    "claim_graph_policy",
    "trustrag_hybrid_v5",
    "wo_trust",
    "wo_anchor",
    "wo_diversity",
]

DEFAULT_INPUT = BASE_DIR / "exp4_eval_data.jsonl"
DEFAULT_OUTPUT_JSONL = BASE_DIR / "judged_results.jsonl"
DEFAULT_OUTPUT_CSV = BASE_DIR / "summary_report.csv"
DEFAULT_PAIRWISE_JSONL = BASE_DIR / "pairwise_results.jsonl"
DEFAULT_PAIRWISE_CSV = BASE_DIR / "pairwise_summary.csv"
DEFAULT_MODEL = "gpt-4o-2024-08-06"
DEFAULT_SUPPORT_MODEL = "gpt-4o-2024-08-06"
DEFAULT_PAIRWISE_MODEL = DEFAULT_MODEL
DEFAULT_HIGH_TRUST_THRESHOLD = 0.75
DEFAULT_BACKBONE_K = 3

JUDGE_SYSTEM_PROMPT = """You are a rigorous academic evaluator for Retrieval-Augmented Generation (RAG).
Your task is to provide high-precision scores that differentiate subtle answer-quality gaps between methods.

Evaluation priorities:
- grounded_synthesis: Does the answer stay faithful to the provided evidence?
- response_utility: Does the answer adequately address the user's question?

Rules:
- Judge the substance of the answer; do not reward or penalize formatting choices
  (paragraphs vs. bullets, headings vs. plain prose, citation style).
- Use the full 0.00-1.00 scale.
- Prefer precision over coarse rounding.
- Do not use outside knowledge.
- Return ONLY a valid JSON object with fields:
  {"grounded_synthesis": 0.0, "response_utility": 0.0, "explanation": "..."}
"""

SUPPORT_SYSTEM_PROMPT = """You are an evidence-grounding evaluator.
Your job is to determine whether one answer sentence is supported by the provided evidence.

Rules:
- Mark "supported" only when the sentence is clearly justified by the evidence.
- Mark "unsupported" when the sentence introduces a claim not backed by the evidence.
- Mark "unclear" when the sentence is ambiguous, normative, or the evidence is insufficient.
- Do not use outside knowledge.

Return ONLY a valid JSON object:
{
  "label": "supported" | "unsupported" | "unclear",
  "explanation": "<short reason>"
}
"""

MSSR_SYSTEM_PROMPT = """You are an evidence-grounding evaluator for multi-perspective synthesis.
Your job is to determine whether one answer sentence is supported by the provided evidence,
giving credit for valid synthesis claims that emerge from reading multiple sources together.

Rules:
- Mark "supported" when the sentence is justified by the evidence, including:
  * Direct claims that a single source clearly supports.
  * Synthesis claims that emerge from reading multiple sources together — for example,
    claims about conflicting findings ("evidence is mixed", "studies disagree on X"),
    comparative conclusions ("while Source A shows X, Source B finds Y"), or
    epistemic summaries ("empirical evidence for X is limited") that are valid
    integrative conclusions from the cited sources, even if no single source states
    the claim verbatim.
- Mark "unsupported" when the sentence introduces a claim not derivable from any
  combination of the provided evidence.
- Mark "unclear" when the sentence is ambiguous, normative, or the evidence is
  genuinely insufficient to evaluate.
- Do not use outside knowledge.

Return ONLY a valid JSON object:
{
  "label": "supported" | "unsupported" | "unclear",
  "explanation": "<short reason>"
}
"""

PAIRWISE_SYSTEM_PROMPT = """You are a rigorous pairwise evaluator for political RAG answers.
Choose which answer is better overall.

Decision priorities:
1. Groundedness to the provided evidence
2. Evidence use quality and citation discipline
3. Appropriate differentiation of stronger vs. weaker evidence based on content quality and relevance, without false equivalence
4. Balanced treatment of meaningful disagreement without false equivalence
5. Response utility and clarity

Rules:
- Compare Answer A and Answer B only using their paired evidence sets.
- [Card X] references in Answer A refer only to Evidence Set A. The same applies to Answer B.
- Choose "tie" only when the quality difference is genuinely negligible.
- Return ONLY a valid JSON object with fields:
{
  "winner": "A" | "B" | "tie",
  "groundedness_winner": "A" | "B" | "tie",
  "evidence_use_winner": "A" | "B" | "tie",
  "balance_winner": "A" | "B" | "tie",
  "utility_winner": "A" | "B" | "tie",
  "explanation": "..."
}
"""

OPENAI_RUNTIME_HINT = (
    "This script requires the modern OpenAI Python SDK. "
    "Run the public adapter with `python3.8 scripts/run_main_210q.py --stage evaluate ...`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Exp4 outputs for the AllSides/Qbias corpus.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--pairwise-output-jsonl", default=str(DEFAULT_PAIRWISE_JSONL))
    parser.add_argument("--pairwise-output-csv", default=str(DEFAULT_PAIRWISE_CSV))
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--support-model", default=DEFAULT_SUPPORT_MODEL)
    parser.add_argument("--pairwise-model", default=DEFAULT_PAIRWISE_MODEL)
    parser.add_argument("--high-trust-threshold", type=float, default=DEFAULT_HIGH_TRUST_THRESHOLD)
    parser.add_argument("--backbone-k", type=int, default=DEFAULT_BACKBONE_K,
                        help="Number of leading support_checks treated as the answer backbone for trust-weighted metrics.")
    parser.add_argument("--pairwise-target", default="trustrag_anchor", choices=AVAILABLE_METHODS)
    parser.add_argument("--pairwise-baselines", nargs="+", choices=AVAILABLE_METHODS, default=None)
    parser.add_argument("--max-pairwise-evidence-docs", type=int, default=6)
    parser.add_argument("--skip-pairwise", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_openai_client() -> OpenAI:
    if OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("{} Import error: {}".format(OPENAI_RUNTIME_HINT, OPENAI_IMPORT_ERROR))
    return OpenAI()


def create_anthropic_client() -> Any:
    if ANTHROPIC_IMPORT_ERROR is not None:
        raise RuntimeError("anthropic package is required for Claude models. Import error: {}".format(ANTHROPIC_IMPORT_ERROR))
    return anthropic_sdk.Anthropic()


def is_claude_model(model: str) -> bool:
    return model.lower().startswith("claude")


_DEBUG_DUMP_PATH = os.environ.get("CLAUDE_JUDGE_DEBUG_DUMP")


def _extract_schema_block(system_prompt: str) -> str:
    """Pull the JSON schema example out of a system prompt's `Return ONLY ...`
    section. Returns the literal `{...}` block including braces, or "" if none."""
    if not system_prompt:
        return ""
    text = system_prompt
    idx = text.find("Return ONLY")
    if idx == -1:
        idx = text.find("Return only")
    if idx == -1:
        return ""
    start = text.find("{", idx)
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: j + 1]
    return ""


def call_judge_api(model: str, system_prompt: str, user_prompt: str) -> str:
    if is_claude_model(model):
        client = _CLIENTS["anthropic"]
        if client is None:
            raise RuntimeError("Anthropic client not initialized. Use a claude-* model name with --model.")
        schema_block = _extract_schema_block(system_prompt)
        if schema_block:
            schema_reminder = (
                "\n\nYou are an evaluator. Do NOT answer the question yourself. "
                "Output ONLY a single JSON object matching exactly this schema "
                "(same keys, same value types, no extra keys, no markdown fences):\n"
                + schema_block
            )
        else:
            schema_reminder = (
                "\n\nRespond with a single valid JSON object only. "
                "Do not include any prose, preamble, or markdown fences."
            )
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt + schema_reminder},
            ],
            temperature=0.0,
        )
        text = response.content[0].text or ""
        if _DEBUG_DUMP_PATH:
            try:
                with open(_DEBUG_DUMP_PATH, "a", encoding="utf-8") as df:
                    df.write(json.dumps({
                        "model": model,
                        "system_prompt_head": (system_prompt or "")[:200],
                        "user_prompt_head": (user_prompt or "")[:200],
                        "response_text": text,
                        "stop_reason": getattr(response, "stop_reason", None),
                        "usage": getattr(response, "usage", None).__dict__ if getattr(response, "usage", None) else None,
                    }, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass
        return text
    else:
        client = _CLIENTS["openai"]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response.choices[0].message.content


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at line {}: {}".format(line_no, exc)) from exc
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Could not extract valid JSON from model output.")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start: idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    raise ValueError("Could not extract valid JSON from model output.")


def coerce_score(value: Any) -> float:
    try:
        score = float(value)
        return max(0.0, min(1.0, round(score, 4)))
    except Exception:
        return 0.0


def _normalize_structured_answer(text: str) -> str:
    """Flatten bullet-point structured answers (e.g. trustrag_anchor) into prose.

    Lines that are section headers are dropped; bullet markers are stripped so
    each claim line survives as a complete sentence candidate.
    """
    lines = (text or "").splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip().lstrip("-•* \t")
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(_PLAN_HEADER_PATTERNS):
            continue
        cleaned.append(stripped)
    return " ".join(cleaned)


def split_sentences(text: str) -> List[str]:
    normalized = _normalize_structured_answer(text)
    if not normalized.strip():
        return []
    if _SENT_TOKENIZE is not None:
        return [s.strip() for s in _SENT_TOKENIZE(normalized) if s.strip()]
    parts = re.split(r"(?<=[.!?])\s+", normalized.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_doc_refs(text: str) -> List[int]:
    refs = re.findall(r"\[(?:Doc|Card)\s*(\d+)\]", text or "", flags=re.IGNORECASE)
    return [int(ref) for ref in refs]


_PLAN_HEADER_PATTERNS = (
    "most supported points",
    "main disagreement",
    "main disagreements",
    "bottom line",
    "evidence gap",
    "suggested response",
    "response strategy",
)

def is_factual_sentence(sentence: str) -> bool:
    stripped = sentence.strip().lstrip("-•* \t")
    if not stripped:
        return False
    if len(stripped.split()) < 5:
        return False
    lowered = stripped.lower()
    if lowered.startswith(_PLAN_HEADER_PATTERNS):
        return False
    return True


def compute_retrieval_diagnostics(docs: List[Dict[str, Any]], high_trust_threshold: float) -> Dict[str, float]:
    trust_scores = [float(doc.get("S_trust", 0.0) or 0.0) for doc in docs]
    if not trust_scores:
        return {
            "avg_s_trust_at_k": 0.0,
            "top3_avg_s_trust": 0.0,
            "high_trust_ratio": 0.0,
        }

    top3 = trust_scores[:3]
    high_trust = [score for score in trust_scores if score >= high_trust_threshold]
    return {
        "avg_s_trust_at_k": round(mean(trust_scores), 4),
        "top3_avg_s_trust": round(mean(top3), 4),
        "high_trust_ratio": round(len(high_trust) / len(trust_scores), 4),
    }


def compute_citation_metrics(
    answer: str,
    docs: List[Dict[str, Any]],
    high_trust_threshold: float,
) -> Dict[str, Any]:
    sentences = split_sentences(answer)
    factual_sentences = [sentence for sentence in sentences if is_factual_sentence(sentence)]
    cited_factual_sentences = 0
    cited_total_refs = 0
    cited_high_trust_refs = 0
    cited_anchor_refs = 0

    for sentence in factual_sentences:
        refs = extract_doc_refs(sentence)
        if refs:
            cited_factual_sentences += 1
        for ref in refs:
            idx = ref - 1
            if 0 <= idx < len(docs):
                cited_total_refs += 1
                doc = docs[idx]
                if float(doc.get("S_trust", 0.0) or 0.0) >= high_trust_threshold:
                    cited_high_trust_refs += 1
                if bool(doc.get("is_community_anchor", False)):
                    cited_anchor_refs += 1

    factual_count = len(factual_sentences)
    citation_coverage = (cited_factual_sentences / factual_count) if factual_count else 0.0
    high_trust_citation_rate = (cited_high_trust_refs / cited_total_refs) if cited_total_refs else 0.0
    anchor_utilization_rate = (cited_anchor_refs / cited_total_refs) if cited_total_refs else 0.0

    return {
        "factual_sentence_count": factual_count,
        "cited_factual_sentences": cited_factual_sentences,
        "citation_coverage": round(citation_coverage, 4),
        "cited_total_refs": cited_total_refs,
        "high_trust_citation_rate": round(high_trust_citation_rate, 4),
        "anchor_utilization_rate": round(anchor_utilization_rate, 4),
    }


def support_value(label: str) -> float:
    if label == "supported":
        return 1.0
    if label == "unclear":
        return 0.5
    return 0.0


def compute_backbone_metrics(
    support_checks: List[Dict[str, Any]],
    docs: List[Dict[str, Any]],
    high_trust_threshold: float,
    backbone_k: int,
) -> Dict[str, Any]:
    backbone_claims = (support_checks or [])[:backbone_k]
    if not backbone_claims:
        return {
            "backbone_claim_count": 0,
            "backbone_trust_rate": 0.0,
            "trust_weighted_claim_support": 0.0,
            "backbone_avg_claim_trust": 0.0,
            "backbone_support_rate": 0.0,
        }

    claim_scores: List[float] = []
    claim_trusts: List[float] = []
    claim_supports: List[float] = []
    high_trust_ref_count = 0
    total_ref_count = 0

    for item in backbone_claims:
        refs = extract_doc_refs(item.get("sentence", ""))
        ref_trusts: List[float] = []
        for ref in refs:
            idx = ref - 1
            if 0 <= idx < len(docs):
                trust = float(docs[idx].get("S_trust", 0.0) or 0.0)
                ref_trusts.append(trust)
                total_ref_count += 1
                if trust >= high_trust_threshold:
                    high_trust_ref_count += 1
        mean_claim_trust = mean(ref_trusts) if ref_trusts else 0.0
        label_score = support_value(item.get("label", "unsupported"))
        claim_trusts.append(mean_claim_trust)
        claim_supports.append(label_score)
        claim_scores.append(label_score * mean_claim_trust)

    backbone_trust_rate = (high_trust_ref_count / total_ref_count) if total_ref_count else 0.0
    return {
        "backbone_claim_count": len(backbone_claims),
        "backbone_trust_rate": round(backbone_trust_rate, 4),
        "trust_weighted_claim_support": round(mean(claim_scores), 4),
        "backbone_avg_claim_trust": round(mean(claim_trusts), 4),
        "backbone_support_rate": round(mean(claim_supports), 4),
    }


def check_sentence_support(
    client: Any,
    model: str,
    sentence: str,
    evidence_docs: List[Dict[str, Any]],
    system_prompt: str = SUPPORT_SYSTEM_PROMPT,
) -> Dict[str, str]:
    evidence_blocks = []
    for doc in evidence_docs:
        evidence_blocks.append(
            "Doc [{}] (Source: {})\n{}".format(
                int(doc.get("card_id", 0) or 0),
                doc.get("source", "N/A"),
                (doc.get("text") or "")[:900],
            )
        )

    user_prompt = "Answer sentence:\n{}\n\nEvidence:\n{}".format(sentence, "\n\n".join(evidence_blocks))

    try:
        raw = call_judge_api(model, system_prompt, user_prompt)
        parsed = extract_json_object(raw)
        label = str(parsed.get("label", "unclear")).strip().lower()
        if label not in {"supported", "unsupported", "unclear"}:
            label = "unclear"
        return {"label": label, "explanation": parsed.get("explanation", "")}
    except Exception as exc:
        return {"label": "unclear", "explanation": str(exc)}


def compute_evidence_support(
    client: OpenAI,
    model: str,
    answer: str,
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    support_checks = []
    for sentence in split_sentences(answer):
        if not is_factual_sentence(sentence):
            continue
        doc_refs = extract_doc_refs(sentence)
        if not doc_refs:
            continue

        referenced_docs = []
        for ref in doc_refs:
            idx = ref - 1
            if 0 <= idx < len(docs):
                referenced_docs.append(docs[idx])

        if not referenced_docs:
            support_checks.append(
                {
                    "sentence": sentence,
                    "label": "unsupported",
                    "explanation": "Cited document index not found in retrieval set.",
                }
            )
            continue

        result = check_sentence_support(client, model, sentence, referenced_docs)
        support_checks.append(
            {
                "sentence": sentence,
                "label": result["label"],
                "explanation": result["explanation"],
            }
        )

    evaluable = [item for item in support_checks if item["label"] in {"supported", "unsupported", "unclear"}]
    supported = [item for item in evaluable if item["label"] == "supported"]
    unsupported = [item for item in evaluable if item["label"] == "unsupported"]
    unclear = [item for item in evaluable if item["label"] == "unclear"]
    denominator = len(evaluable) if evaluable else 0
    support_rate = (len(supported) / denominator) if denominator else 0.0

    return {
        "evidence_support_rate": round(support_rate, 4),
        "supported_sentences": len(supported),
        "unsupported_sentences": len(unsupported),
        "unclear_sentences": len(unclear),
        "evaluable_sentences": denominator,
        "support_checks": support_checks,
    }


def compute_mssr(
    client: OpenAI,
    model: str,
    answer: str,
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    support_checks = []
    for sentence in split_sentences(answer):
        if not is_factual_sentence(sentence):
            continue
        doc_refs = extract_doc_refs(sentence)
        if not doc_refs:
            continue

        referenced_docs = []
        for ref in doc_refs:
            idx = ref - 1
            if 0 <= idx < len(docs):
                referenced_docs.append(docs[idx])

        if not referenced_docs:
            support_checks.append(
                {
                    "sentence": sentence,
                    "label": "unsupported",
                    "explanation": "Cited document index not found in retrieval set.",
                }
            )
            continue

        result = check_sentence_support(client, model, sentence, referenced_docs, system_prompt=MSSR_SYSTEM_PROMPT)
        support_checks.append(
            {
                "sentence": sentence,
                "label": result["label"],
                "explanation": result["explanation"],
            }
        )

    evaluable = [item for item in support_checks if item["label"] in {"supported", "unsupported", "unclear"}]
    supported = [item for item in evaluable if item["label"] == "supported"]
    unsupported = [item for item in evaluable if item["label"] == "unsupported"]
    unclear = [item for item in evaluable if item["label"] == "unclear"]
    denominator = len(evaluable) if evaluable else 0
    mssr = (len(supported) / denominator) if denominator else 0.0

    return {
        "mssr": round(mssr, 4),
        "mssr_supported": len(supported),
        "mssr_unsupported": len(unsupported),
        "mssr_unclear": len(unclear),
        "mssr_evaluable": denominator,
        "mssr_checks": support_checks,
    }


def judge_one(
    client: OpenAI,
    model: str,
    question: str,
    docs: List[Dict[str, Any]],
    answer: str,
) -> Dict[str, Any]:
    doc_items = []
    missing_text_docs = 0
    for idx, doc in enumerate(docs, start=1):
        content = doc.get("text") or ""
        if not content:
            missing_text_docs += 1
        doc_items.append(
            "Doc [{}] (Source: {} | Perspective: {})\n{}".format(
                idx,
                doc.get("source", "N/A"),
                doc.get("stance_label", "unknown"),
                content[:900],
            )
        )

    user_prompt = "Question: {}\n\nEvidence:\n{}\n\nAnswer: {}".format(question, "\n\n".join(doc_items), answer)

    try:
        raw = call_judge_api(model, JUDGE_SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_object(raw)
        return {
            "groundedness": coerce_score(parsed.get("grounded_synthesis") or parsed.get("grounding")),
            "response_utility": coerce_score(parsed.get("response_utility")),
            "explanation": parsed.get("explanation", ""),
            "doc_count": len(docs),
            "missing_text_docs": missing_text_docs,
        }
    except Exception as exc:
        return {
            "groundedness": 0.0,
            "response_utility": 0.0,
            "explanation": str(exc),
            "doc_count": len(docs),
            "missing_text_docs": missing_text_docs,
        }


def unique_doc_refs(answer: str) -> List[int]:
    refs = []
    seen = set()
    for ref in extract_doc_refs(answer):
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def select_pairwise_docs(answer: str, docs: List[Dict[str, Any]], max_docs: int) -> List[Dict[str, Any]]:
    by_card_id = {}
    for doc in docs:
        try:
            by_card_id[int(doc.get("card_id"))] = doc
        except Exception:
            continue

    selected = []
    selected_ids = set()
    for ref in unique_doc_refs(answer):
        doc = by_card_id.get(ref)
        if doc is not None and ref not in selected_ids:
            selected.append(doc)
            selected_ids.add(ref)
        if len(selected) >= max_docs:
            return selected

    for doc in docs:
        card_id = int(doc.get("card_id", 0) or 0)
        if card_id in selected_ids:
            continue
        selected.append(doc)
        selected_ids.add(card_id)
        if len(selected) >= max_docs:
            break
    return selected


def format_pairwise_evidence_set(label: str, docs: List[Dict[str, Any]]) -> str:
    lines = ["Evidence Set {}:".format(label)]
    for doc in docs:
        lines.append(
            "Card [{}] (Source: {} | Perspective: {} | Rank: {})\n{}".format(
                int(doc.get("card_id", 0) or 0),
                doc.get("source", "N/A"),
                doc.get("stance_label", "unknown"),
                doc.get("retrieval_rank", "N/A"),
                (doc.get("text") or "")[:900],
            )
        )
    return "\n\n".join(lines)


def map_pairwise_label(label: Any, method_at_a: str, method_at_b: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"a", "answer_a", "answer a", method_at_a.lower()}:
        return method_at_a
    if normalized in {"b", "answer_b", "answer b", method_at_b.lower()}:
        return method_at_b
    return "tie"


def _pairwise_swap_decision(query_id: Any, target_method: str, baseline_method: str) -> bool:
    """Deterministic 50/50 swap decision keyed on (query_id, target, baseline).
    Returns True if the target should be placed at position B (i.e., swapped)."""
    seed_str = "{}|{}|{}".format(query_id, target_method, baseline_method)
    digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return (digest[0] & 1) == 1


def judge_pairwise(
    client: OpenAI,
    model: str,
    question: str,
    target_method: str,
    baseline_method: str,
    target_answer: str,
    baseline_answer: str,
    target_docs: List[Dict[str, Any]],
    baseline_docs: List[Dict[str, Any]],
    query_id: Any = None,
) -> Dict[str, Any]:
    swap = _pairwise_swap_decision(query_id, target_method, baseline_method)
    if swap:
        method_at_a, method_at_b = baseline_method, target_method
        answer_at_a, answer_at_b = baseline_answer, target_answer
        docs_at_a, docs_at_b = baseline_docs, target_docs
    else:
        method_at_a, method_at_b = target_method, baseline_method
        answer_at_a, answer_at_b = target_answer, baseline_answer
        docs_at_a, docs_at_b = target_docs, baseline_docs

    user_prompt = (
        "Question: {question}\n\n"
        "{evidence_a}\n\n"
        "Answer A:\n{answer_a}\n\n"
        "{evidence_b}\n\n"
        "Answer B:\n{answer_b}\n"
    ).format(
        question=question,
        evidence_a=format_pairwise_evidence_set("A", docs_at_a),
        answer_a=answer_at_a,
        evidence_b=format_pairwise_evidence_set("B", docs_at_b),
        answer_b=answer_at_b,
    )

    try:
        raw = call_judge_api(model, PAIRWISE_SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_object(raw)
    except Exception as exc:
        return {
            "winner": "tie",
            "groundedness_winner": "tie",
            "evidence_use_winner": "tie",
            "balance_winner": "tie",
            "utility_winner": "tie",
            "explanation": str(exc),
            "position_swap": swap,
            "target_position": "B" if swap else "A",
        }

    return {
        "winner": map_pairwise_label(parsed.get("winner"), method_at_a, method_at_b),
        "groundedness_winner": map_pairwise_label(parsed.get("groundedness_winner"), method_at_a, method_at_b),
        "evidence_use_winner": map_pairwise_label(parsed.get("evidence_use_winner"), method_at_a, method_at_b),
        "balance_winner": map_pairwise_label(parsed.get("balance_winner"), method_at_a, method_at_b),
        "utility_winner": map_pairwise_label(parsed.get("utility_winner"), method_at_a, method_at_b),
        "explanation": parsed.get("explanation", ""),
        "position_swap": swap,
        "target_position": "B" if swap else "A",
    }


def write_summary_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_absolute_summary(rows: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    topics = sorted({row.get("topic", "unknown") for row in rows})
    summary_rows = []

    for topic in ["all"] + topics:
        for method in methods:
            items = [row for row in rows if row["method"] == method and (topic == "all" or row["topic"] == topic)]
            if not items:
                continue
            n = len(items)
            summary_rows.append(
                {
                    "topic": topic,
                    "method": method,
                    "n": n,
                    "groundedness": round(sum(item["groundedness"] for item in items) / n, 4),
                    "evidence_support_rate": round(sum(item["evidence_support_rate"] for item in items) / n, 4),
                    "mssr": round(sum(item.get("mssr", 0.0) for item in items) / n, 4),
                    "citation_coverage": round(sum(item["citation_coverage"] for item in items) / n, 4),
                    "high_trust_citation_rate": round(sum(item["high_trust_citation_rate"] for item in items) / n, 4),
                    "response_utility": round(sum(item["response_utility"] for item in items) / n, 4),
                    "anchor_utilization_rate": round(sum(item["anchor_utilization_rate"] for item in items) / n, 4),
                    "avg_s_trust_at_k": round(sum(item["avg_s_trust_at_k"] for item in items) / n, 4),
                    "top3_avg_s_trust": round(sum(item["top3_avg_s_trust"] for item in items) / n, 4),
                    "high_trust_ratio": round(sum(item["high_trust_ratio"] for item in items) / n, 4),
                    "trust_weighted_claim_support": round(sum(item.get("trust_weighted_claim_support", 0.0) for item in items) / n, 4),
                    "backbone_trust_rate": round(sum(item.get("backbone_trust_rate", 0.0) for item in items) / n, 4),
                    "backbone_avg_claim_trust": round(sum(item.get("backbone_avg_claim_trust", 0.0) for item in items) / n, 4),
                    "backbone_support_rate": round(sum(item.get("backbone_support_rate", 0.0) for item in items) / n, 4),
                    "backbone_claim_count": round(sum(item.get("backbone_claim_count", 0) for item in items) / n, 4),
                }
            )
    return summary_rows


def build_pairwise_summary(rows: List[Dict[str, Any]], target_method: str, baselines: List[str]) -> List[Dict[str, Any]]:
    topics = sorted({row.get("topic", "unknown") for row in rows})
    summary_rows = []

    for topic in ["all"] + topics:
        for baseline in baselines:
            items = [
                row
                for row in rows
                if row["baseline_method"] == baseline and (topic == "all" or row["topic"] == topic)
            ]
            if not items:
                continue

            n = len(items)
            target_wins = sum(1 for item in items if item["winner"] == target_method)
            baseline_wins = sum(1 for item in items if item["winner"] == baseline)
            ties = sum(1 for item in items if item["winner"] == "tie")
            non_tie = target_wins + baseline_wins
            summary_rows.append(
                {
                    "topic": topic,
                    "target_method": target_method,
                    "baseline_method": baseline,
                    "n": n,
                    "target_wins": target_wins,
                    "baseline_wins": baseline_wins,
                    "ties": ties,
                    "target_win_rate": round(target_wins / n, 4),
                    "non_tie_target_win_rate": round((target_wins / non_tie), 4) if non_tie else 0.0,
                }
            )

    return summary_rows


def main() -> None:
    args = parse_args()

    # Initialize API clients based on which models are requested.
    all_models = {args.model, args.support_model, getattr(args, "pairwise_model", args.model)}
    needs_anthropic = any(is_claude_model(m) for m in all_models if m)
    needs_openai = any(not is_claude_model(m) for m in all_models if m)

    if needs_openai:
        _CLIENTS["openai"] = create_openai_client()
    if needs_anthropic:
        _CLIENTS["anthropic"] = create_anthropic_client()

    client = _CLIENTS["openai"] or _CLIENTS["anthropic"]  # kept for legacy call sites

    input_rows = load_jsonl(Path(args.input))

    methods = list(args.methods)
    pairwise_target = args.pairwise_target
    pairwise_baselines = args.pairwise_baselines or [method for method in methods if method != pairwise_target]
    pairwise_baselines = [method for method in pairwise_baselines if method != pairwise_target]

    # Resume: load already-judged (query_id, method) pairs from existing absolute output.
    judged_path = Path(args.output_jsonl)
    judged_rows: List[Dict[str, Any]] = []
    judged_done: set = set()
    if judged_path.exists():
        try:
            judged_rows = load_jsonl(judged_path)
            judged_done = {(r.get("query_id"), r.get("method")) for r in judged_rows}
            if judged_done:
                print("[resume] absolute eval: {} entries already done.".format(len(judged_done)))
        except Exception:
            judged_rows = []

    for row in tqdm(input_rows, desc="Absolute evaluation"):
        for method in methods:
            if (row.get("query_id"), method) in judged_done:
                continue

            answer = row.get("answers", {}).get(method)
            if answer is None:
                continue

            retrieval_docs = row.get("retrievals", {}).get(method, [])
            judged = judge_one(
                client=client,
                model=args.model,
                question=row.get("question", ""),
                docs=retrieval_docs,
                answer=answer,
            )
            judged.update(
                compute_retrieval_diagnostics(
                    retrieval_docs,
                    high_trust_threshold=args.high_trust_threshold,
                )
            )
            judged.update(
                compute_evidence_support(
                    client=client,
                    model=args.support_model,
                    answer=answer,
                    docs=retrieval_docs,
                )
            )
            judged.update(
                compute_backbone_metrics(
                    support_checks=judged.get("support_checks", []),
                    docs=retrieval_docs,
                    high_trust_threshold=args.high_trust_threshold,
                    backbone_k=args.backbone_k,
                )
            )
            judged.update(
                compute_mssr(
                    client=client,
                    model=args.support_model,
                    answer=answer,
                    docs=retrieval_docs,
                )
            )
            judged.update(
                compute_citation_metrics(
                    answer=answer,
                    docs=retrieval_docs,
                    high_trust_threshold=args.high_trust_threshold,
                )
            )
            judged["method"] = method
            judged["query_id"] = row.get("query_id")
            judged["topic"] = row.get("topic", "unknown")
            judged["topic_display"] = row.get("topic_display", row.get("topic", "unknown"))
            judged["question"] = row.get("question", "")
            judged_rows.append(judged)
            with open(judged_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(judged, ensure_ascii=False) + "\n")

    # final dedup write (resume may have loaded pre-existing rows)
    write_jsonl(judged_path, judged_rows)

    summary_rows = build_absolute_summary(judged_rows, methods)
    write_summary_csv(
        Path(args.output_csv),
        summary_rows,
        fieldnames=[
            "topic",
            "method",
            "n",
            "groundedness",
            "evidence_support_rate",
            "mssr",
            "citation_coverage",
            "high_trust_citation_rate",
            "response_utility",
            "anchor_utilization_rate",
            "avg_s_trust_at_k",
            "top3_avg_s_trust",
            "high_trust_ratio",
            "trust_weighted_claim_support",
            "backbone_trust_rate",
            "backbone_avg_claim_trust",
            "backbone_support_rate",
            "backbone_claim_count",
        ],
    )

    # Resume: load already-judged (query_id, target_method, baseline_method) triples.
    pairwise_path = Path(args.pairwise_output_jsonl)
    pairwise_rows: List[Dict[str, Any]] = []
    pairwise_done: set = set()
    if not args.skip_pairwise and pairwise_path.exists():
        try:
            pairwise_rows = load_jsonl(pairwise_path)
            pairwise_done = {
                (r.get("query_id"), r.get("target_method"), r.get("baseline_method"))
                for r in pairwise_rows
            }
            if pairwise_done:
                print("[resume] pairwise eval: {} entries already done.".format(len(pairwise_done)))
        except Exception:
            pairwise_rows = []

    if not args.skip_pairwise:
        for row in tqdm(input_rows, desc="Pairwise evaluation"):
            target_answer = row.get("answers", {}).get(pairwise_target)
            target_docs_all = row.get("retrievals", {}).get(pairwise_target, [])
            if target_answer is None:
                continue

            for baseline in pairwise_baselines:
                if (row.get("query_id"), pairwise_target, baseline) in pairwise_done:
                    continue

                baseline_answer = row.get("answers", {}).get(baseline)
                baseline_docs_all = row.get("retrievals", {}).get(baseline, [])
                if baseline_answer is None:
                    continue

                target_docs = select_pairwise_docs(
                    target_answer,
                    target_docs_all,
                    max_docs=args.max_pairwise_evidence_docs,
                )
                baseline_docs = select_pairwise_docs(
                    baseline_answer,
                    baseline_docs_all,
                    max_docs=args.max_pairwise_evidence_docs,
                )
                judged = judge_pairwise(
                    client=client,
                    model=args.pairwise_model,
                    question=row.get("question", ""),
                    target_method=pairwise_target,
                    baseline_method=baseline,
                    target_answer=target_answer,
                    baseline_answer=baseline_answer,
                    target_docs=target_docs,
                    baseline_docs=baseline_docs,
                    query_id=row.get("query_id"),
                )
                judged["query_id"] = row.get("query_id")
                judged["topic"] = row.get("topic", "unknown")
                judged["topic_display"] = row.get("topic_display", row.get("topic", "unknown"))
                judged["question"] = row.get("question", "")
                judged["target_method"] = pairwise_target
                judged["baseline_method"] = baseline
                pairwise_rows.append(judged)

    write_jsonl(pairwise_path, pairwise_rows)

    pairwise_summary_rows = build_pairwise_summary(pairwise_rows, pairwise_target, pairwise_baselines)
    write_summary_csv(
        Path(args.pairwise_output_csv),
        pairwise_summary_rows,
        fieldnames=[
            "topic",
            "target_method",
            "baseline_method",
            "n",
            "target_wins",
            "baseline_wins",
            "ties",
            "target_win_rate",
            "non_tie_target_win_rate",
        ],
    )

    print("Saved judged per-query results to {}".format(args.output_jsonl))
    print("Saved summary CSV to {}".format(args.output_csv))
    print("Saved pairwise results to {}".format(args.pairwise_output_jsonl))
    print("Saved pairwise summary to {}".format(args.pairwise_output_csv))


if __name__ == "__main__":
    main()
