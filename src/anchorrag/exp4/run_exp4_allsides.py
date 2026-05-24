import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import tiktoken
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

_TOKENIZER = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

try:
    from openai import OpenAI
    OPENAI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import path depends on local env
    OpenAI = Any  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DEFAULT_OUTPUT_JSONL = BASE_DIR / "exp4_eval_data.jsonl"
DEFAULT_CACHE_DIR = BASE_DIR / "embed_cache"
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
]

MODEL_GEN = "gpt-4o-2024-08-06"
PLANNER_MODEL = MODEL_GEN
EMBED_MODEL = "text-embedding-3-small"
RETRIEVAL_TEXT_LIMIT = 225  # 900 chars → 225 tokens
EVIDENCE_SNIPPET_LIMIT_TOKENS = 80  # ~320 chars → 80 tokens (token-based truncation)
ANCHOR_HIGH_TRUST_THRESHOLD = 0.8
TRUST_DIV_RHO = 0.8
TRUST_DIV_MODE = "community"
CARD_GAMMA = 0.65
ANCHOR_LAMBDA_STRUCTURE = 0.45
ANCHOR_LAMBDA_RELEVANCE = 0.35
ANCHOR_LAMBDA_SUPPORT = 0.20
RECLAIM_MAX_STEPS = 5
RECLAIM_MAX_SENTENCES_PER_CARD = 3
RECLAIM_MAX_SENTENCE_POOL = 24
GRAG_EGO_HOPS = 2
GRAG_TOP_SUBGRAPHS = 4
TRUSTRAG_MAX_CLAIM_UNITS = 5
TRUSTRAG_SELECTED_CARD_LIMIT = 8
QUESTION_DIRECTNESS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "do",
    "does",
    "different",
    "either",
    "exists",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "interpret",
    "is",
    "main",
    "of",
    "on",
    "or",
    "regarding",
    "the",
    "their",
    "there",
    "to",
    "used",
    "what",
    "whether",
    "which",
    "who",
    "why",
    "with",
}
CLAIM_SUPPORT_STRENGTH = {
    "high": 1.0,
    "medium": 0.65,
    "low": 0.35,
}
CLAIM_GRAPH_MAX_CARD_CLAIMS = 2
CLAIM_GRAPH_MAX_TOTAL_CLAIMS = 12
POLICY_MAX_CLAIM_UNITS = 4
POLICY_DIRECTNESS_THRESHOLD = 0.16
POLICY_CONFLICT_TERMS = {
    "conflicting",
    "against",
    "disagree",
    "different",
    "trade-off",
    "tradeoff",
    "balance",
    "for and against",
    "either",
    "versus",
    "vs",
}
POLICY_DIRECT_EVIDENCE_TERMS = {
    "impact",
    "effects",
    "effect",
    "cause",
    "causes",
    "causal",
    "relationship",
    "reduce",
    "reduces",
    "increase",
    "increases",
    "improve",
    "improves",
    "worsen",
    "worsens",
    "deters",
    "deter",
    "growth",
    "employment",
    "jobs",
    "wages",
    "crime",
    "violence",
    "finances",
    "public safety",
}
POLICY_HIGH_RISK_TERMS = {
    "cause",
    "causes",
    "caused",
    "lead",
    "leads",
    "led",
    "result",
    "results",
    "boost",
    "boosts",
    "improve",
    "improves",
    "harm",
    "harms",
    "reduce",
    "reduces",
    "increase",
    "increases",
    "decrease",
    "decreases",
    "prove",
    "proves",
}
HYBRID_V5_SELECTED_CARD_LIMIT = 6
HYBRID_V5_MAX_SUPPORTED = 2
HYBRID_V5_MAX_UNCERTAINTIES = 2
HYBRID_V5_AUDIT_DOC_LIMIT = 3

SUPPORT_SYSTEM_PROMPT = """You are checking whether one answer sentence is supported by evidence.

Your job is to determine whether one answer sentence is supported by the provided evidence.

Rules:
- Mark "supported" only when the sentence is clearly justified by the evidence.
- Mark "unsupported" when the sentence introduces a claim not backed by the evidence.
- Mark "unclear" when the evidence is only partial, ambiguous, or too indirect.
- Be strict about causal or quantitative claims.

Return ONLY valid JSON:
{
  "label": "supported" | "unsupported" | "unclear",
  "explanation": "brief reason"
}
"""

TOPIC_CONFIGS = {
    "immigration": {
        "display_name": "Immigration",
        "graph_path": str(PROJECT_DIR / "graphs" / "immigration_scored_graph.json"),
        "queries": [
            {
                "id": "imm-01",
                "q": "What conflicting evidence exists regarding the impact of immigration on native-born wages and employment?",
            },
            {
                "id": "imm-02",
                "q": "How do different political perspectives evaluate the fiscal effects of immigration on public services such as healthcare and education?",
            },
            {
                "id": "imm-03",
                "q": "What are the main arguments for and against stricter border enforcement, and what evidence is used to support each side?",
            },
            {
                "id": "imm-04",
                "q": "How do different sources interpret the relationship between immigration policy and public safety?",
            },
            {
                "id": "imm-05",
                "q": "What conflicting evidence is presented on whether immigration contributes to long-term economic growth and labor market resilience?",
            },
            {
                "id": "imm-06",
                "q": "How do political actors disagree on the humanitarian and legal trade-offs in asylum and border-control policy?",
            },
            {
                "id": "imm-07",
                "q": "What are the most disputed arguments about immigration’s role in national identity, multiculturalism, and social cohesion?",
            },
            {
                "id": "imm-08",
                "q": "How do commentators differ in their explanations of undocumented immigration as either a labor-market issue or a policy-enforcement failure?",
            },
            {
                "id": "imm-09",
                "q": "What evidence is used to argue that expanding legal immigration pathways would either reduce or worsen irregular migration?",
            },
            {
                "id": "imm-10",
                "q": "How do different ideological perspectives frame the balance between national sovereignty and the rights of migrants and asylum seekers?",
            },
        ],
    },
    "gun_control": {
        "display_name": "Gun Control And Gun Rights",
        "graph_path": str(PROJECT_DIR / "graphs" / "gun_control_scored_graph.json"),
        "queries": [
            {
                "id": "gun-01",
                "q": "What conflicting evidence exists regarding whether stricter gun control laws reduce gun violence?",
            },
            {
                "id": "gun-02",
                "q": "How do different perspectives interpret the relationship between gun ownership rates and public safety?",
            },
            {
                "id": "gun-03",
                "q": "What are the main arguments for and against assault weapon bans, and what evidence is used to support each side?",
            },
            {
                "id": "gun-04",
                "q": "How do different sources interpret the effects of background checks on gun-related crime and mass shootings?",
            },
            {
                "id": "gun-05",
                "q": "What conflicting evidence is presented on whether permissive concealed-carry laws deter or increase violent crime?",
            },
            {
                "id": "gun-06",
                "q": "How do political actors disagree on the constitutional and public-safety trade-offs in firearm regulation?",
            },
            {
                "id": "gun-07",
                "q": "What are the most disputed arguments about the role of firearms in self-defense versus public risk?",
            },
            {
                "id": "gun-08",
                "q": "How do commentators differ in explaining school shootings as a gun-access problem versus a broader mental-health or security failure?",
            },
            {
                "id": "gun-09",
                "q": "What evidence is used to argue that waiting periods and red flag laws either prevent harm or unfairly restrict lawful gun owners?",
            },
            {
                "id": "gun-10",
                "q": "How do different ideological perspectives frame the balance between Second Amendment rights and collective safety concerns?",
            },
        ],
    },
    "economy_jobs": {
        "display_name": "Economy And Jobs",
        "graph_path": str(PROJECT_DIR / "graphs" / "economy_jobs_scored_graph.json"),
        "queries": [
            {
                "id": "eco-01",
                "q": "What conflicting evidence exists regarding whether recent economic growth has broadly improved wages, jobs, and household finances?",
            },
            {
                "id": "eco-02",
                "q": "How do different political perspectives interpret the causes of inflation and the effectiveness of government responses?",
            },
            {
                "id": "eco-03",
                "q": "What are the main arguments for and against raising the minimum wage, and what evidence is used to support each side?",
            },
            {
                "id": "eco-04",
                "q": "How do different sources interpret the relationship between unemployment, job creation, and government economic policy?",
            },
            {
                "id": "eco-05",
                "q": "What conflicting evidence is presented on whether tax cuts stimulate broad-based job growth and investment?",
            },
            {
                "id": "eco-06",
                "q": "How do political actors disagree on the trade-off between deficit reduction and economic stimulus spending?",
            },
            {
                "id": "eco-07",
                "q": "What are the most disputed claims about globalization and trade as drivers of job loss versus consumer and productivity gains?",
            },
            {
                "id": "eco-08",
                "q": "How do commentators differ in explaining manufacturing decline as a trade problem, an automation problem, or a policy failure?",
            },
            {
                "id": "eco-09",
                "q": "What evidence is used to argue that labor shortages reflect excessive worker bargaining power versus structural mismatches in the labor market?",
            },
            {
                "id": "eco-10",
                "q": "How do different ideological perspectives frame the balance between market freedom, worker protection, and government intervention in the economy?",
            },
        ],
    },
    "abortion": {
        "display_name": "Abortion",
        "graph_path": str(PROJECT_DIR / "graphs" / "abortion_scored_graph.json"),
        "queries": [
            {
                "id": "abo-01",
                "q": "What conflicting evidence exists regarding the health impacts of abortion access versus restrictions on maternal outcomes?",
            },
            {
                "id": "abo-02",
                "q": "How do different sources interpret evidence about late-term abortions and medical necessity exceptions?",
            },
            {
                "id": "abo-03",
                "q": "What evidence is used to argue that abortion restrictions change overall abortion rates versus only shifting timing or geography?",
            },
            {
                "id": "abo-04",
                "q": "How do different perspectives evaluate the effects of abortion policy on low-income or rural access to healthcare?",
            },
            {
                "id": "abo-05",
                "q": "What evidence is cited to support or challenge claims about fetal development thresholds used in abortion policy debates?",
            },
            {
                "id": "abo-06",
                "q": "How do political actors disagree on the legal and constitutional trade-offs in abortion regulation?",
            },
            {
                "id": "abo-07",
                "q": "What evidence is used to argue that abortion policy affects adoption rates or child welfare outcomes?",
            },
            {
                "id": "abo-08",
                "q": "How do different sources interpret evidence on the relationship between abortion access and women’s economic outcomes?",
            },
            {
                "id": "abo-09",
                "q": "What conflicting evidence is presented on whether parental notification or consent laws change health outcomes?",
            },
            {
                "id": "abo-10",
                "q": "How do ideological perspectives differ in interpreting the balance between bodily autonomy and fetal rights, and what evidence is used to support each side?",
            },
        ],
    },
    "free_speech": {
        "display_name": "Free Speech",
        "graph_path": str(PROJECT_DIR / "graphs" / "free_speech_scored_graph.json"),
        "queries": [
            {
                "id": "speech-01",
                "q": "What evidence is used to argue that content moderation reduces harm versus suppresses legitimate speech?",
            },
            {
                "id": "speech-02",
                "q": "How do different sources interpret evidence about deplatforming and its effects on extremist activity or misinformation spread?",
            },
            {
                "id": "speech-03",
                "q": "What are the main arguments for and against hate speech restrictions, and what evidence is used to support each side?",
            },
            {
                "id": "speech-04",
                "q": "How do commentators differ in interpreting the impact of campus speech policies on academic freedom and student safety?",
            },
            {
                "id": "speech-05",
                "q": "What evidence is used to argue that government pressure on platforms does or does not chill speech?",
            },
            {
                "id": "speech-06",
                "q": "How do different perspectives evaluate the role of private platforms versus public regulation in shaping speech norms?",
            },
            {
                "id": "speech-07",
                "q": "What conflicting evidence exists regarding whether misinformation labeling improves or polarizes public discourse?",
            },
            {
                "id": "speech-08",
                "q": "How do sources interpret the trade-off between rapid moderation and viewpoint neutrality?",
            },
            {
                "id": "speech-09",
                "q": "What evidence is cited about the effects of algorithmic amplification on controversial speech?",
            },
            {
                "id": "speech-10",
                "q": "How do ideological perspectives frame the balance between free expression and protection from harassment or threats?",
            },
        ],
    },
    "voting_rights_and_voter_fraud": {
        "display_name": "Voting Rights And Voter Fraud",
        "graph_path": str(PROJECT_DIR / "graphs" / "voting_rights_and_voter_fraud_scored_graph.json"),
        "queries": [
            {
                "id": "vote-01",
                "q": "What evidence is used to argue that voter fraud is widespread versus rare, and how reliable are those sources?",
            },
            {
                "id": "vote-02",
                "q": "How do different perspectives interpret the effects of voter ID laws on turnout and election integrity?",
            },
            {
                "id": "vote-03",
                "q": "What conflicting evidence exists regarding the security and accessibility of mail-in voting?",
            },
            {
                "id": "vote-04",
                "q": "How do sources interpret evidence about ballot drop boxes and their impact on participation or fraud risk?",
            },
            {
                "id": "vote-05",
                "q": "What evidence is used to argue that voter roll maintenance prevents fraud versus suppresses eligible voters?",
            },
            {
                "id": "vote-06",
                "q": "How do political actors disagree on the trade-offs between election security measures and voting access?",
            },
            {
                "id": "vote-07",
                "q": "What evidence is cited about the effects of early voting or same-day registration on turnout?",
            },
            {
                "id": "vote-08",
                "q": "How do commentators differ in assessing partisan gerrymandering and its impact on representation?",
            },
            {
                "id": "vote-09",
                "q": "What conflicting evidence is presented on whether election administration changes affect public trust?",
            },
            {
                "id": "vote-10",
                "q": "How do ideological perspectives frame the balance between federal oversight and state control in election policy?",
            },
        ],
    },
}

METHOD_SPECS = {
    "vanilla_rag": {
        "display_name": "vanilla_rag",
        "retrieval_family": "semantic_topk",
        "generation_family": "basic_grounded_synthesis",
    },
    "mmr_rag": {
        "display_name": "mmr_rag",
        "retrieval_family": "mmr_diversity",
        "generation_family": "basic_grounded_synthesis",
    },
    "graph_retrieval": {
        "display_name": "graph_retrieval",
        "retrieval_family": "graph_diversity",
        "generation_family": "basic_grounded_synthesis",
    },
    "r2ag": {
        "display_name": "R2AG",
        "retrieval_family": "semantic_topk",
        "generation_family": "retrieval_info_prefixed_prompting",
    },
    "reclaim": {
        "display_name": "ReClaim",
        "retrieval_family": "semantic_topk",
        "generation_family": "interleaving_reference_claim_generation",
    },
    "grag": {
        "display_name": "GRAG",
        "retrieval_family": "textual_subgraph_retrieval",
        "generation_family": "two_view_graph_context_generation",
    },
    "trustrag_anchor": {
        "display_name": "trustrag_anchor",
        "retrieval_family": "trust_diversity",
        "generation_family": "anchor_aware_generation",
    },
    "trustrag_anchor_top2": {
        "display_name": "trustrag_anchor_top2",
        "retrieval_family": "trust_diversity",
        "generation_family": "anchor_aware_generation_top2",
    },
    "claim_graph_only": {
        "display_name": "claim_graph_only",
        "retrieval_family": "trust_diversity",
        "generation_family": "claim_graph_only",
    },
    "claim_graph_verifier": {
        "display_name": "claim_graph_verifier",
        "retrieval_family": "trust_diversity",
        "generation_family": "claim_graph_verifier",
    },
    "claim_graph_policy": {
        "display_name": "claim_graph_policy",
        "retrieval_family": "trust_diversity",
        "generation_family": "claim_graph_policy",
    },
    "trustrag_hybrid_v5": {
        "display_name": "trustrag_hybrid_v5",
        "retrieval_family": "trust_diversity",
        "generation_family": "hybrid_trust_audit_v5",
    },
}

OPENAI_RUNTIME_HINT = (
    "This script requires the modern OpenAI Python SDK. "
    "Run the public adapter with `python3.8 scripts/run_main_210q.py --stage generate ...`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp4 end-to-end comparison on the AllSides/Qbias corpus.")
    parser.add_argument("--topics", nargs="+", choices=list(TOPIC_CONFIGS.keys()), default=list(TOPIC_CONFIGS.keys()))
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--embedding-model", default=EMBED_MODEL)
    parser.add_argument("--generator-model", default=MODEL_GEN)
    parser.add_argument("--planner-model", default=PLANNER_MODEL)
    parser.add_argument("--mmr-lambda", type=float, default=0.70)
    parser.add_argument("--graph-alpha", type=float, default=0.70)
    parser.add_argument("--graph-gamma", type=float, default=0.25)
    parser.add_argument("--graph-delta", type=float, default=0.05)
    parser.add_argument("--trust-alpha", type=float, default=0.70)
    parser.add_argument("--trust-beta", type=float, default=0.30)
    parser.add_argument("--trust-div-rho", type=float, default=TRUST_DIV_RHO,
                        help="Penalty multiplier applied to nodes from already-represented communities (default 0.8).")
    parser.add_argument("--trust-div-mode", choices=["community", "progressive", "off"], default=TRUST_DIV_MODE,
                        help="'community' = single-step penalty (m_div=rho if community already chosen); "
                             "'progressive' = exponential penalty (m_div=rho^n_c, where n_c is repeat count); "
                             "'off' = no diversity penalty.")
    parser.add_argument("--card-gamma", type=float, default=CARD_GAMMA,
                        help="Weight on query relevance in evidence-card weight; (1-gamma) goes to S_trust (default 0.65).")
    parser.add_argument("--anchor-lambda-structure", type=float, default=ANCHOR_LAMBDA_STRUCTURE,
                        help="Anchor-score weight on internal_degree_norm (lambda_1, default 0.45).")
    parser.add_argument("--anchor-lambda-relevance", type=float, default=ANCHOR_LAMBDA_RELEVANCE,
                        help="Anchor-score weight on query_norm (lambda_2, default 0.35).")
    parser.add_argument("--anchor-lambda-support", type=float, default=ANCHOR_LAMBDA_SUPPORT,
                        help="Anchor-score weight on high_trust_support_ratio (lambda_3, default 0.20).")
    parser.add_argument("--anchor-high-trust-threshold", type=float, default=ANCHOR_HIGH_TRUST_THRESHOLD,
                        help="S_trust threshold (tau_anchor) for counting a neighbor as 'high-trust support' (default 0.8).")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_openai_client() -> OpenAI:
    if OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("{} Import error: {}".format(OPENAI_RUNTIME_HINT, OPENAI_IMPORT_ERROR))
    return OpenAI()


def cache_key(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return "{}_{}.npy".format(prefix, digest)


def embed_texts(
    client: OpenAI,
    items: Sequence[str],
    model: str,
    cache_dir: Path,
    prefix: str,
    refresh_cache: bool = False,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(prefix, "{}::{}".format(model, "\n".join(items)))
    cache_path = cache_dir / key
    if cache_path.exists() and not refresh_cache:
        return np.load(cache_path)

    vectors = []
    for start in tqdm(range(0, len(items), 100), desc="Embedding {}".format(prefix)):
        batch = [item.replace("\n", " ") for item in items[start : start + 100]]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    arr = np.array(vectors, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


def clean_snippet(text: str, limit: int = EVIDENCE_SNIPPET_LIMIT_TOKENS) -> str:
    collapsed = " ".join((text or "").split())
    tokens = _TOKENIZER.encode(collapsed)
    if len(tokens) <= limit:
        return collapsed
    return _TOKENIZER.decode(tokens[: limit - 1]) + "…"


def normalize_focus_token(token: str) -> str:
    token = token.lower().strip("-_' ")
    if len(token) <= 2:
        return ""
    if token.endswith("'s"):
        token = token[:-2]
    for suffix in ("ingly", "edly", "ing", "ers", "ies", "ied", "ed", "es", "s"):
        if len(token) >= 6 and token.endswith(suffix):
            if suffix in {"ies", "ied"}:
                token = token[:-3] + "y"
            else:
                token = token[: -len(suffix)]
            break
    return token


def extract_focus_terms(text: str) -> List[str]:
    terms = []
    seen = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z\-']+", text or ""):
        term = normalize_focus_token(raw)
        if not term or term in QUESTION_DIRECTNESS_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def directness_band(score: float) -> str:
    if score >= 0.32:
        return "high"
    if score >= 0.18:
        return "medium"
    return "low"


def compute_direct_answer_score(question: str, node: Dict[str, Any]) -> float:
    question_terms = extract_focus_terms(question)
    if not question_terms:
        return 0.0

    heading = node.get("heading") or node.get("title") or ""
    body_text = "{} {} {}".format(
        heading,
        node.get("summary") or "",
        clean_snippet(node.get("text") or "", limit=175),
    )
    title_terms = set(extract_focus_terms(heading))
    body_terms = set(extract_focus_terms(body_text))
    if not body_terms:
        return 0.0

    overlap = sum(1 for term in question_terms if term in body_terms) / len(question_terms)
    title_overlap = sum(1 for term in question_terms if term in title_terms) / len(question_terms)
    return round(min(1.0, (0.7 * overlap) + (0.25 * title_overlap)), 4)


def annotate_question_directness(
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
) -> None:
    score_by_card_id: Dict[int, float] = {}
    band_by_card_id: Dict[int, str] = {}
    for node in retrieved_nodes:
        card_id = int(node.get("card_id", 0) or 0)
        score = compute_direct_answer_score(question, node)
        band = directness_band(score)
        node["direct_answer_score"] = score
        node["direct_answer_band"] = band
        score_by_card_id[card_id] = score
        band_by_card_id[card_id] = band

    for anchor in community_anchors:
        card_id = int(anchor.get("card_id", 0) or 0)
        anchor["direct_answer_score"] = score_by_card_id.get(card_id, 0.0)
        anchor["direct_answer_band"] = band_by_card_id.get(card_id, "low")


def split_text_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def extract_doc_refs(text: str) -> List[int]:
    refs = []
    for match in re.findall(r"\[Card\s+(\d+)\]", text or ""):
        try:
            refs.append(int(match))
        except Exception:
            continue
    return refs


def edge_signature(link: Dict[str, Any]) -> str:
    src = int(link.get("source"))
    tgt = int(link.get("target"))
    relation = str(link.get("relation", "related"))
    reason = str(link.get("reason", ""))
    return "{}|{}|{}|{}".format(min(src, tgt), max(src, tgt), relation, reason)


def edge_text(link: Dict[str, Any]) -> str:
    relation = str(link.get("relation", "related"))
    reason = str(link.get("reason", "")).strip()
    if reason:
        return "{}: {}".format(relation, reason)
    return relation


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-8:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def load_graph_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "nodes": data.get("nodes", []),
        "links": data.get("links", data.get("edges", [])),
    }


def build_anchor_feature_map(nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    node_by_id = {int(node["id"]): node for node in nodes}
    internal_degree = {node_id: 0 for node_id in node_by_id}
    support_total = {node_id: 0 for node_id in node_by_id}
    support_high_trust = {node_id: 0 for node_id in node_by_id}

    for link in links:
        src = int(link.get("source"))
        tgt = int(link.get("target"))
        relation = link.get("relation")
        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)
        if src_node is None or tgt_node is None:
            continue

        if src_node.get("community") == tgt_node.get("community"):
            internal_degree[src] += 1
            internal_degree[tgt] += 1

        if relation == "Support":
            support_total[src] += 1
            support_total[tgt] += 1
            if float(tgt_node.get("S_trust", 0.0) or 0.0) >= ANCHOR_HIGH_TRUST_THRESHOLD:
                support_high_trust[src] += 1
            if float(src_node.get("S_trust", 0.0) or 0.0) >= ANCHOR_HIGH_TRUST_THRESHOLD:
                support_high_trust[tgt] += 1

    max_degree_by_community: Dict[int, int] = {}
    for node_id, degree in internal_degree.items():
        community = int(node_by_id[node_id].get("community", -1))
        max_degree_by_community[community] = max(max_degree_by_community.get(community, 0), degree)

    feature_map: Dict[int, Dict[str, float]] = {}
    for node_id, node in node_by_id.items():
        community = int(node.get("community", -1))
        max_degree = max_degree_by_community.get(community, 0)
        internal_degree_norm = (internal_degree[node_id] / max_degree) if max_degree else 0.0
        total_support = support_total[node_id]
        high_trust_support_ratio = (support_high_trust[node_id] / total_support) if total_support else 0.0
        feature_map[node_id] = {
            "internal_degree_norm": round(internal_degree_norm, 4),
            "high_trust_support_ratio": round(high_trust_support_ratio, 4),
        }
    return feature_map


def annotate_retrieval(
    nodes: List[Dict[str, Any]],
    sims: np.ndarray,
    selected_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    retrieved_nodes = []
    for rank, idx in enumerate(selected_indices, start=1):
        node = dict(nodes[idx])
        node["query_relevance"] = float(sims[idx])
        node["retrieval_rank"] = rank
        node["_node_index"] = idx
        retrieved_nodes.append(node)
    return retrieved_nodes


def annotate_r2ag_features(retrieved_nodes: List[Dict[str, Any]], node_embs: np.ndarray) -> None:
    if not retrieved_nodes:
        return

    selected_indices = [int(node.get("_node_index")) for node in retrieved_nodes]
    selected_embs = node_embs[selected_indices]
    pairwise_sims = cosine_similarity(selected_embs)
    relevance_values = [float(node.get("query_relevance", 0.0) or 0.0) for node in retrieved_nodes]
    normalized_rel = normalize_scores(relevance_values)
    total = len(retrieved_nodes)

    for idx, node in enumerate(retrieved_nodes):
        precedent = max(pairwise_sims[idx][:idx]) if idx > 0 else 0.0
        neighbor_values = []
        if idx > 0:
            neighbor_values.append(pairwise_sims[idx][idx - 1])
        if idx + 1 < total:
            neighbor_values.append(pairwise_sims[idx][idx + 1])
        neighbor_similarity = (sum(neighbor_values) / len(neighbor_values)) if neighbor_values else 0.0
        rank_percentile = 1.0 if total == 1 else 1.0 - (idx / (total - 1))
        listwise_coherence = float(np.mean(pairwise_sims[idx])) if total else 0.0
        node["r2ag_rel_norm"] = round(normalized_rel[idx], 4)
        node["r2ag_rank_percentile"] = round(rank_percentile, 4)
        node["r2ag_precedent_similarity"] = round(float(precedent), 4)
        node["r2ag_neighbor_similarity"] = round(float(neighbor_similarity), 4)
        node["r2ag_listwise_coherence"] = round(float(listwise_coherence), 4)


def semantic_topk_retrieval(
    query_emb: np.ndarray,
    nodes: List[Dict[str, Any]],
    node_embs: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    sims = cosine_similarity([query_emb], node_embs)[0]
    indices = list(sims.argsort()[-top_k:][::-1])
    return annotate_retrieval(nodes, sims, indices)


def mmr_select(query_emb: np.ndarray, node_embs: np.ndarray, top_k: int, lambda_param: float) -> List[int]:
    similarities = cosine_similarity([query_emb], node_embs)[0]
    selected = [int(np.argmax(similarities))]
    unselected = set(range(len(node_embs))) - set(selected)

    while len(selected) < top_k and unselected:
        best_idx = None
        best_score = -1e9
        for idx in unselected:
            sim_q = similarities[idx]
            sim_doc = max(cosine_similarity([node_embs[idx]], [node_embs[j]])[0][0] for j in selected)
            score = lambda_param * sim_q - (1 - lambda_param) * sim_doc
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(int(best_idx))
        unselected.remove(best_idx)
    return selected


def mmr_retrieval(
    query_emb: np.ndarray,
    nodes: List[Dict[str, Any]],
    node_embs: np.ndarray,
    top_k: int,
    lambda_param: float,
) -> List[Dict[str, Any]]:
    sims = cosine_similarity([query_emb], node_embs)[0]
    indices = mmr_select(query_emb, node_embs, top_k, lambda_param)
    return annotate_retrieval(nodes, sims, indices)


def redundancy_penalty(candidate_idx: int, selected_indices: List[int], node_embs: np.ndarray) -> float:
    if not selected_indices:
        return 0.0
    sims = [cosine_similarity([node_embs[candidate_idx]], [node_embs[idx]])[0][0] for idx in selected_indices]
    return max(sims)


def diversity_bonus(community: int, community_counts: Dict[int, int]) -> float:
    count = community_counts.get(community, 0)
    return 1.0 / math.sqrt(count + 1)


def graph_retrieval(
    query_emb: np.ndarray,
    nodes: List[Dict[str, Any]],
    node_embs: np.ndarray,
    top_k: int,
    candidate_pool: int,
    alpha: float,
    gamma: float,
    delta: float,
) -> List[Dict[str, Any]]:
    sims = cosine_similarity([query_emb], node_embs)[0]
    candidate_indices = list(sims.argsort()[-candidate_pool:][::-1])
    selected_indices: List[int] = []
    community_counts: Dict[int, int] = {}

    while len(selected_indices) < top_k and candidate_indices:
        scored = []
        for idx in candidate_indices:
            rel = float(sims[idx])
            community = int(nodes[idx].get("community", -1))
            bonus = diversity_bonus(community, community_counts)
            redundancy = redundancy_penalty(idx, selected_indices, node_embs)
            score = (alpha * rel) + (gamma * bonus) - (delta * redundancy)
            scored.append((score, idx))

        scored.sort(reverse=True, key=lambda item: item[0])
        _, best_idx = scored[0]
        selected_indices.append(best_idx)
        community = int(nodes[best_idx].get("community", -1))
        community_counts[community] = community_counts.get(community, 0) + 1
        candidate_indices.remove(best_idx)

    return annotate_retrieval(nodes, sims, selected_indices)


def trustrag_retrieval(
    query_emb: np.ndarray,
    nodes: List[Dict[str, Any]],
    node_embs: np.ndarray,
    top_k: int,
    candidate_pool: int,
    alpha: float,
    beta: float,
    use_diversity: bool = True,
) -> List[Dict[str, Any]]:
    sims = cosine_similarity([query_emb], node_embs)[0]
    candidate_indices = list(sims.argsort()[-candidate_pool:][::-1])
    selected_indices: List[int] = []
    selected_community_counts: Dict[int, int] = {}

    while len(selected_indices) < top_k and candidate_indices:
        scored = []
        for idx in candidate_indices:
            rel = float(sims[idx])
            trust = float(nodes[idx].get("S_trust", 0.0) or 0.0)
            community = int(nodes[idx].get("community", -1))
            n_c = selected_community_counts.get(community, 0)
            if not use_diversity or TRUST_DIV_MODE == "off":
                div_multiplier = 1.0
            elif TRUST_DIV_MODE == "progressive":
                div_multiplier = TRUST_DIV_RHO ** n_c
            else:  # "community" (single-step)
                div_multiplier = 1.0 if n_c == 0 else TRUST_DIV_RHO
            score = (alpha * rel) + (beta * trust * div_multiplier)
            scored.append((score, idx, community))

        scored.sort(reverse=True, key=lambda item: item[0])
        _, best_idx, best_community = scored[0]
        selected_indices.append(best_idx)
        selected_community_counts[best_community] = selected_community_counts.get(best_community, 0) + 1
        candidate_indices.remove(best_idx)

    return annotate_retrieval(nodes, sims, selected_indices)


def collect_k_hop_ego_graph(
    center_node_id: int,
    links_by_node_id: Dict[int, List[Dict[str, Any]]],
    hops: int,
) -> Dict[str, Any]:
    frontier = {center_node_id}
    visited_nodes = {center_node_id}
    node_depths = {center_node_id: 0}
    visited_edges = {}

    for hop in range(1, hops + 1):
        next_frontier = set()
        for node_id in frontier:
            for link in links_by_node_id.get(node_id, []):
                src = int(link.get("source"))
                tgt = int(link.get("target"))
                other = tgt if src == node_id else src
                edge_key = (
                    min(src, tgt),
                    max(src, tgt),
                    str(link.get("relation", "")),
                    str(link.get("reason", "")),
                )
                if edge_key not in visited_edges:
                    visited_edges[edge_key] = link
                if other not in visited_nodes:
                    visited_nodes.add(other)
                    next_frontier.add(other)
                    node_depths[other] = hop
        frontier = next_frontier
        if not frontier:
            break

    return {
        "center_node_id": center_node_id,
        "node_ids": sorted(visited_nodes),
        "edges": list(visited_edges.values()),
        "node_depths": node_depths,
    }


def grag_subgraph_retrieval(
    query_emb: np.ndarray,
    topic_payload: Dict[str, Any],
    top_k: int,
    candidate_pool: int,
) -> Dict[str, Any]:
    nodes = topic_payload["nodes"]
    node_embs = topic_payload["node_embs"]
    edge_embs = topic_payload["edge_embs"]
    edge_key_to_index = topic_payload["edge_key_to_index"]
    node_id_to_index = topic_payload["node_id_to_index"]
    links_by_node_id = topic_payload["links_by_node_id"]

    sims = cosine_similarity([query_emb], node_embs)[0]
    edge_sims = cosine_similarity([query_emb], edge_embs)[0] if len(edge_embs) else np.array([], dtype=np.float32)
    candidate_indices = list(sims.argsort()[-candidate_pool:][::-1])
    ego_graphs = []

    for idx in candidate_indices:
        center_node = nodes[idx]
        center_node_id = int(center_node.get("id"))
        ego = collect_k_hop_ego_graph(center_node_id, links_by_node_id, GRAG_EGO_HOPS)
        ego_indices = [node_id_to_index[node_id] for node_id in ego["node_ids"] if node_id in node_id_to_index]
        if not ego_indices:
            continue

        edge_indices = []
        edge_query_sims = []
        for link in ego["edges"]:
            key = edge_signature(link)
            edge_idx = edge_key_to_index.get(key)
            if edge_idx is None:
                continue
            edge_indices.append(edge_idx)
            edge_query_sims.append(float(edge_sims[edge_idx]))

        pooled_vectors = [node_embs[ego_indices]]
        if edge_indices:
            pooled_vectors.append(edge_embs[edge_indices])
        mean_embedding = np.mean(np.vstack(pooled_vectors), axis=0)
        subgraph_query_similarity = float(cosine_similarity([query_emb], [mean_embedding])[0][0])
        avg_trust = sum(float(nodes[i].get("S_trust", 0.0) or 0.0) for i in ego_indices) / len(ego_indices)
        center_similarity = float(sims[idx])
        mean_edge_similarity = (sum(edge_query_sims) / len(edge_query_sims)) if edge_query_sims else 0.0
        structural_density = (len(ego["edges"]) / max(len(ego["node_ids"]), 1))
        ego_score = (
            (0.50 * subgraph_query_similarity)
            + (0.30 * center_similarity)
            + (0.13 * mean_edge_similarity)
            + (0.07 * min(structural_density, 1.0))
        )
        ego_graphs.append(
            {
                "center_node_id": center_node_id,
                "center_node_index": idx,
                "node_ids": ego["node_ids"],
                "edges": ego["edges"],
                "node_depths": ego["node_depths"],
                "subgraph_query_similarity": round(subgraph_query_similarity, 4),
                "center_query_similarity": round(center_similarity, 4),
                "avg_trust": round(avg_trust, 4),
                "mean_edge_similarity": round(mean_edge_similarity, 4),
                "ego_score": round(ego_score, 4),
            }
        )

    ego_graphs.sort(key=lambda item: item["ego_score"], reverse=True)
    selected_ego_graphs = ego_graphs[: min(GRAG_TOP_SUBGRAPHS, len(ego_graphs))]

    selected_node_ids = set()
    selected_edges_map = {}
    membership_count = Counter()
    min_depth_by_node = {}

    for ego_rank, ego in enumerate(selected_ego_graphs, start=1):
        ego["ego_rank"] = ego_rank
        for node_id in ego["node_ids"]:
            selected_node_ids.add(node_id)
            membership_count[node_id] += 1
            current_depth = ego["node_depths"].get(node_id, GRAG_EGO_HOPS + 1)
            min_depth_by_node[node_id] = min(min_depth_by_node.get(node_id, current_depth), current_depth)
        for link in ego["edges"]:
            src = int(link.get("source"))
            tgt = int(link.get("target"))
            edge_key = (
                min(src, tgt),
                max(src, tgt),
                str(link.get("relation", "")),
                str(link.get("reason", "")),
            )
            if edge_key not in selected_edges_map:
                selected_edges_map[edge_key] = link

    selected_edges = list(selected_edges_map.values())
    max_membership = max(membership_count.values()) if membership_count else 1

    edge_query_similarity_map = {}
    for link in selected_edges:
        key = edge_signature(link)
        edge_idx = edge_key_to_index.get(key)
        edge_query_similarity_map[key] = float(edge_sims[edge_idx]) if edge_idx is not None else 0.0

    node_scores = []

    for node_id in selected_node_ids:
        idx = node_id_to_index[node_id]
        node = dict(nodes[idx])
        node_rel = float(sims[idx])
        trust = float(node.get("S_trust", 0.0) or 0.0)
        membership = membership_count[node_id] / max_membership
        depth = min_depth_by_node.get(node_id, GRAG_EGO_HOPS + 1)
        edge_weights = []
        edge_query_values = []
        for link in selected_edges:
            if int(link.get("source")) == node_id or int(link.get("target")) == node_id:
                edge_weights.append(float(link.get("weight", 0.0) or 0.0))
                edge_query_values.append(edge_query_similarity_map.get(edge_signature(link), 0.0))
        mean_edge_weight = (sum(edge_weights) / len(edge_weights)) if edge_weights else 0.0
        mean_edge_query = (sum(edge_query_values) / len(edge_query_values)) if edge_query_values else 0.0
        soft_prune_score = (
            (0.48 * node_rel)
            + (0.20 * membership)
            + (0.15 * mean_edge_weight)
            + (0.17 * mean_edge_query)
        )
        soft_prune_score -= 0.05 * max(depth - 1, 0)
        node["grag_membership_count"] = membership_count[node_id]
        node["grag_min_depth"] = depth
        node["grag_soft_prune_score"] = round(float(soft_prune_score), 4)
        node["grag_is_key_node"] = any(node_id == ego["center_node_id"] for ego in selected_ego_graphs)
        node_scores.append((soft_prune_score, idx, node))

    node_scores.sort(reverse=True, key=lambda item: item[0])
    selected_indices = []
    key_center_indices = []
    for ego in selected_ego_graphs:
        center_idx = int(ego.get("center_node_index"))
        if center_idx not in key_center_indices:
            key_center_indices.append(center_idx)
    for idx in key_center_indices:
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break
    for _, idx, _ in node_scores:
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break
    retrieved_nodes = annotate_retrieval(nodes, sims, selected_indices)
    by_id = {int(node.get("id")): node for node in retrieved_nodes}
    for _, _, scored_node in node_scores[:top_k]:
        node = by_id.get(int(scored_node.get("id")))
        if node is not None:
            node["grag_membership_count"] = scored_node.get("grag_membership_count")
            node["grag_min_depth"] = scored_node.get("grag_min_depth")
            node["grag_soft_prune_score"] = scored_node.get("grag_soft_prune_score")
            node["grag_is_key_node"] = scored_node.get("grag_is_key_node")

    return {
        "retrieved_nodes": retrieved_nodes,
        "method_context": {
            "selected_ego_graphs": selected_ego_graphs,
            "selected_edges": selected_edges,
            "edge_query_similarity_map": edge_query_similarity_map,
        },
    }


def attach_anchor_metadata(
    retrieved_nodes: List[Dict[str, Any]],
    anchor_feature_map: Dict[int, Dict[str, float]],
    anchors_per_community: int = 1,
) -> List[Dict[str, Any]]:
    communities: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in retrieved_nodes:
        communities[int(node.get("community", -1))].append(node)

    community_anchors = []
    selected_anchor_card_ids = set()
    for community_id, community_nodes in communities.items():
        query_values = [float(node.get("query_relevance", 0.0) or 0.0) for node in community_nodes]
        normalized_query_values = normalize_scores(query_values)

        scored_nodes = []
        for node, query_norm in zip(community_nodes, normalized_query_values):
            features = anchor_feature_map.get(int(node.get("id")), {})
            internal_degree_norm = float(features.get("internal_degree_norm", 0.0) or 0.0)
            high_trust_support_ratio = float(features.get("high_trust_support_ratio", 0.0) or 0.0)
            anchor_score = (
                (ANCHOR_LAMBDA_STRUCTURE * internal_degree_norm)
                + (ANCHOR_LAMBDA_RELEVANCE * query_norm)
                + (ANCHOR_LAMBDA_SUPPORT * high_trust_support_ratio)
            )
            node["anchor_internal_degree_norm"] = round(internal_degree_norm, 4)
            node["anchor_query_norm"] = round(query_norm, 4)
            node["anchor_high_trust_support_ratio"] = round(high_trust_support_ratio, 4)
            node["anchor_score"] = round(anchor_score, 4)
            scored_nodes.append(node)

        top_anchor_count = max(1, anchors_per_community)
        top_nodes = sorted(scored_nodes, key=lambda item: item["anchor_score"], reverse=True)[:top_anchor_count]
        for anchor_rank, anchor_node in enumerate(top_nodes, start=1):
            anchor_node["anchor_rank_in_community"] = anchor_rank
            anchor_node["anchors_in_community"] = len(top_nodes)
            selected_anchor_card_ids.add(anchor_node.get("card_id"))
            community_anchors.append(
                {
                    "community": community_id,
                    "card_id": anchor_node.get("card_id"),
                    "node_id": anchor_node.get("id"),
                    "heading": anchor_node.get("heading") or anchor_node.get("title") or "Untitled",
                    "source": anchor_node.get("source") or "Unknown",
                    "stance_label": anchor_node.get("stance_label") or "unknown",
                    "anchor_score": anchor_node.get("anchor_score"),
                    "anchor_internal_degree_norm": anchor_node.get("anchor_internal_degree_norm"),
                    "anchor_query_norm": anchor_node.get("anchor_query_norm"),
                    "anchor_high_trust_support_ratio": anchor_node.get("anchor_high_trust_support_ratio"),
                    "anchor_rank_in_community": anchor_rank,
                    "anchors_in_community": len(top_nodes),
                    "anchor_reason": (
                        "Selected as community anchor #{rank} because it is central in-cluster, "
                        "well aligned with the query, and connected to relatively trustworthy support."
                    ).format(rank=anchor_rank),
                }
            )

    for node in retrieved_nodes:
        node["is_community_anchor"] = node.get("card_id") in selected_anchor_card_ids

    community_anchors.sort(key=lambda item: item["anchor_score"], reverse=True)
    return community_anchors


def finalize_retrieval(
    retrieved_nodes: List[Dict[str, Any]],
    anchor_feature_map: Dict[int, Dict[str, float]],
    anchors_per_community: int = 1,
) -> List[Dict[str, Any]]:
    for card_id, node in enumerate(retrieved_nodes, start=1):
        node["card_id"] = card_id
    return attach_anchor_metadata(
        retrieved_nodes,
        anchor_feature_map,
        anchors_per_community=anchors_per_community,
    )


def build_basic_evidence_cards(question: str, retrieved_nodes: List[Dict[str, Any]]) -> str:
    cards = []
    for idx, node in enumerate(retrieved_nodes, start=1):
        heading = node.get("heading") or node.get("title") or "Untitled"
        cards.append(
            (
                "Evidence Card [{idx}]\n"
                "- Heading: {heading}\n"
                "- Source: {source}\n"
                "- Perspective: {stance}\n"
                "- Evidence excerpt: {excerpt}\n"
            ).format(
                idx=idx,
                heading=heading,
                source=node.get("source") or "Unknown",
                stance=node.get("stance_label") or "unknown",
                excerpt=clean_snippet(node.get("text") or ""),
            )
        )
    return "Question Focus:\n{}\n\nEvidence Cards:\n{}".format(question, "\n".join(cards))


def build_retrieval_metadata_cards(question: str, retrieved_nodes: List[Dict[str, Any]]) -> str:
    cards = []
    for idx, node in enumerate(retrieved_nodes, start=1):
        heading = node.get("heading") or node.get("title") or "Untitled"
        cards.append(
            (
                "Evidence Card [{idx}]\n"
                "- Heading: {heading}\n"
                "- Source: {source}\n"
                "- Perspective: {stance}\n"
                "- Retrieval rank: {rank}\n"
                "- Query relevance: {rel:.3f}\n"
                "- Community: {community}\n"
                "- Evidence excerpt: {excerpt}\n"
            ).format(
                idx=idx,
                heading=heading,
                source=node.get("source") or "Unknown",
                stance=node.get("stance_label") or "unknown",
                rank=int(node.get("retrieval_rank", idx)),
                rel=float(node.get("query_relevance", 0.0) or 0.0),
                community=node.get("community", -1),
                excerpt=clean_snippet(node.get("text") or ""),
            )
        )
    return "Question Focus:\n{}\n\nEvidence Cards With Retrieval Signals:\n{}".format(question, "\n".join(cards))


def build_trust_evidence_cards(question: str, retrieved_nodes: List[Dict[str, Any]]) -> str:
    cards = []
    for idx, node in enumerate(retrieved_nodes, start=1):
        heading = node.get("heading") or node.get("title") or "Untitled"
        anchor_tag = "Yes" if node.get("is_community_anchor") else "No"
        evidence_weight = (CARD_GAMMA * float(node.get("query_relevance", 0.0) or 0.0)) + (
            (1.0 - CARD_GAMMA) * float(node.get("S_trust", 0.0) or 0.0)
        )
        cards.append(
            (
                "Evidence Card [{idx}]\n"
                "- Heading: {heading}\n"
                "- Source: {source}\n"
                "- Perspective: {stance}\n"
                "- S_trust: {trust:.3f}\n"
                "- Query relevance: {rel:.3f}\n"
                "- Evidence weight: {weight:.3f}\n"
                "- Direct answer fit: {directness:.3f} ({directness_band})\n"
                "- Community anchor: {anchor_tag}\n"
                "- AnchorScore: {anchor_score:.3f}\n"
                "- Community: {community}\n"
                "- Evidence excerpt: {excerpt}\n"
                "- Use guidance: Prefer cards with stronger direct-answer fit for the main answer. Treat anchors as candidate backbones, not mandatory backbones. Use lower-trust or indirect cards mainly for disagreement, caveats, or context.\n"
            ).format(
                idx=idx,
                heading=heading,
                source=node.get("source") or "Unknown",
                stance=node.get("stance_label") or "unknown",
                trust=float(node.get("S_trust", 0.0) or 0.0),
                rel=float(node.get("query_relevance", 0.0) or 0.0),
                weight=evidence_weight,
                directness=float(node.get("direct_answer_score", 0.0) or 0.0),
                directness_band=node.get("direct_answer_band", "low"),
                anchor_tag=anchor_tag,
                anchor_score=float(node.get("anchor_score", 0.0) or 0.0),
                community=node.get("community", -1),
                excerpt=clean_snippet(node.get("text") or ""),
            )
        )
    return "Question Focus:\n{}\n\nTrust-Aware Evidence Cards:\n{}".format(question, "\n".join(cards))


def build_anchor_block(community_anchors: List[Dict[str, Any]]) -> str:
    lines = ["Community Anchors:", "Anchors are candidate backbone cards, not mandatory backbone cards."]
    for item in community_anchors:
        lines.append(
            "- Community {community} Anchor {anchor_rank}/{anchor_total}: Card [{card_id}] | {heading} | source={source} | stance={stance} | "
            "AnchorScore={score:.3f} | centrality={centrality:.3f} | query_align={query_align:.3f} | "
            "trusted_support={trusted_support:.3f} | direct_fit={direct_fit:.3f} ({direct_band})".format(
                community=item.get("community"),
                anchor_rank=item.get("anchor_rank_in_community", 1),
                anchor_total=item.get("anchors_in_community", 1),
                card_id=item.get("card_id"),
                heading=item.get("heading"),
                source=item.get("source"),
                stance=item.get("stance_label"),
                score=float(item.get("anchor_score", 0.0) or 0.0),
                centrality=float(item.get("anchor_internal_degree_norm", 0.0) or 0.0),
                query_align=float(item.get("anchor_query_norm", 0.0) or 0.0),
                trusted_support=float(item.get("anchor_high_trust_support_ratio", 0.0) or 0.0),
                direct_fit=float(item.get("direct_answer_score", 0.0) or 0.0),
                direct_band=item.get("direct_answer_band", "low"),
            )
        )
        lines.append("  Reason: {}".format(item.get("anchor_reason", "")))
    return "\n".join(lines)


def build_graph_cluster_block(retrieved_nodes: List[Dict[str, Any]]) -> str:
    communities: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in retrieved_nodes:
        communities[int(node.get("community", -1))].append(node)

    lines = ["Retrieved Community Structure:"]
    for community_id, group in sorted(
        communities.items(),
        key=lambda item: max(float(node.get("query_relevance", 0.0) or 0.0) for node in item[1]),
        reverse=True,
    ):
        representative = max(group, key=lambda node: float(node.get("query_relevance", 0.0) or 0.0))
        stance_mix = ", ".join(sorted({str(node.get("stance_label", "unknown")) for node in group}))
        lines.append(
            "- Community {community}: representative Card [{card_id}] ({heading}) | docs={count} | stances={stances}".format(
                community=community_id,
                card_id=representative.get("card_id"),
                heading=representative.get("heading") or representative.get("title") or "Untitled",
                count=len(group),
                stances=stance_mix,
            )
        )
        for node in sorted(group, key=lambda item: float(item.get("query_relevance", 0.0) or 0.0), reverse=True)[:3]:
            lines.append(
                "  - Card [{card_id}] | {source} | rel={rel:.3f} | {heading}".format(
                    card_id=node.get("card_id"),
                    source=node.get("source") or "Unknown",
                    rel=float(node.get("query_relevance", 0.0) or 0.0),
                    heading=node.get("heading") or node.get("title") or "Untitled",
                )
            )
    return "\n".join(lines)


def build_r2ag_signal_block(retrieved_nodes: List[Dict[str, Any]]) -> str:
    lines = ["Retriever Features:"]
    for node in retrieved_nodes:
        lines.append(
            "<R_{card_id}> rank_pct={rank_pct:.3f} rel={rel:.3f} rel_norm={rel_norm:.3f} "
            "precedent_sim={precedent:.3f} neighbor_sim={neighbor:.3f} coherence={coherence:.3f}".format(
                card_id=int(node.get("card_id", 0) or 0),
                rank_pct=float(node.get("r2ag_rank_percentile", 0.0) or 0.0),
                rel=float(node.get("query_relevance", 0.0) or 0.0),
                rel_norm=float(node.get("r2ag_rel_norm", 0.0) or 0.0),
                precedent=float(node.get("r2ag_precedent_similarity", 0.0) or 0.0),
                neighbor=float(node.get("r2ag_neighbor_similarity", 0.0) or 0.0),
                coherence=float(node.get("r2ag_listwise_coherence", 0.0) or 0.0),
            )
        )
        lines.append(
            "  Card [{card_id}] | {source} | {heading}".format(
                card_id=int(node.get("card_id", 0) or 0),
                source=node.get("source") or "Unknown",
                heading=node.get("heading") or node.get("title") or "Untitled",
            )
        )
    return "\n".join(lines)


def materialize_grag_context(
    method_context: Dict[str, Any],
    retrieved_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not method_context:
        return {}

    node_by_id = {int(node.get("id")): node for node in retrieved_nodes}
    card_by_node_id = {int(node.get("id")): int(node.get("card_id")) for node in retrieved_nodes}
    selected_node_ids = set(card_by_node_id)
    selected_edges = []
    for link in method_context.get("selected_edges", []):
        src = int(link.get("source"))
        tgt = int(link.get("target"))
        if src in selected_node_ids and tgt in selected_node_ids:
            edge_key = edge_signature(link)
            selected_edges.append(
                {
                    "edge_key": edge_key,
                    "source_node_id": src,
                    "target_node_id": tgt,
                    "source_card_id": card_by_node_id[src],
                    "target_card_id": card_by_node_id[tgt],
                    "relation": link.get("relation", "related"),
                    "weight": float(link.get("weight", 0.0) or 0.0),
                    "query_similarity": float(method_context.get("edge_query_similarity_map", {}).get(edge_key, 0.0) or 0.0),
                    "reason": clean_snippet(link.get("reason") or "", 55),
                }
            )

    selected_ego_graphs = []
    for ego in method_context.get("selected_ego_graphs", []):
        selected_ego_graphs.append(
            {
                "ego_rank": ego.get("ego_rank"),
                "center_node_id": ego.get("center_node_id"),
                "center_card_id": card_by_node_id.get(int(ego.get("center_node_id"))),
                "center_query_similarity": ego.get("center_query_similarity"),
                "subgraph_query_similarity": ego.get("subgraph_query_similarity"),
                "avg_trust": ego.get("avg_trust"),
                "mean_edge_similarity": ego.get("mean_edge_similarity"),
                "ego_score": ego.get("ego_score"),
                "node_card_ids": [card_by_node_id[node_id] for node_id in ego.get("node_ids", []) if node_id in card_by_node_id],
                "node_depths_by_card": {
                    card_by_node_id[node_id]: depth
                    for node_id, depth in ego.get("node_depths", {}).items()
                    if node_id in card_by_node_id
                },
            }
        )

    text_view_lines = ["Hierarchical Evidence View (key documents with k-hop neighbors):"]
    for ego in selected_ego_graphs:
        center_card_id = ego.get("center_card_id")
        if center_card_id is None:
            continue
        text_view_lines.append(
            "- Key Card [{card_id}] | cluster_rank={rank} | center_relevance={center_rel:.3f} | cluster_relevance={subgraph_rel:.3f}".format(
                card_id=center_card_id,
                rank=int(ego.get("ego_rank", 0) or 0),
                center_rel=float(ego.get("center_query_similarity", 0.0) or 0.0),
                subgraph_rel=float(ego.get("subgraph_query_similarity", 0.0) or 0.0),
            )
        )
        root_node = node_by_id.get(int(ego.get("center_node_id", -1)))
        if root_node is not None:
            text_view_lines.append(
                "  Root Card [{card_id}] | {source} | {heading}".format(
                    card_id=center_card_id,
                    source=root_node.get("source") or "Unknown",
                    heading=root_node.get("heading") or root_node.get("title") or "Untitled",
                )
            )

        node_depths_by_card = ego.get("node_depths_by_card", {})
        depth_sorted_cards = sorted(
            [(card_id, depth) for card_id, depth in node_depths_by_card.items() if card_id != center_card_id],
            key=lambda item: (item[1], item[0]),
        )
        for card_id, depth in depth_sorted_cards:
            node = next((item for item in retrieved_nodes if int(item.get("card_id", 0) or 0) == int(card_id)), None)
            if node is None:
                continue
            related_edges = [
                edge
                for edge in selected_edges
                if edge["source_card_id"] == card_id or edge["target_card_id"] == card_id
            ]
            strongest_edge = max(
                related_edges,
                key=lambda edge: (edge["query_similarity"], edge["weight"]),
            ) if related_edges else None
            if strongest_edge is not None:
                if strongest_edge["source_card_id"] == card_id:
                    edge_desc = "{relation} -> Card [{other}]".format(
                        relation=strongest_edge["relation"],
                        other=strongest_edge["target_card_id"],
                    )
                else:
                    edge_desc = "{relation} <- Card [{other}]".format(
                        relation=strongest_edge["relation"],
                        other=strongest_edge["source_card_id"],
                    )
                text_view_lines.append(
                    "  Depth {depth}: Card [{card}] | {edge_desc} | q_edge={q_edge:.3f} | {heading}".format(
                        depth=int(depth),
                        card=card_id,
                        edge_desc=edge_desc,
                        q_edge=float(strongest_edge["query_similarity"]),
                        heading=node.get("heading") or node.get("title") or "Untitled",
                    )
                )
            else:
                text_view_lines.append(
                    "  Depth {depth}: Card [{card}] | {heading}".format(
                        depth=int(depth),
                        card=card_id,
                        heading=node.get("heading") or node.get("title") or "Untitled",
                    )
                )

        cross_edges = [
            edge
            for edge in selected_edges
            if edge["source_card_id"] != center_card_id and edge["target_card_id"] != center_card_id
        ]
        if cross_edges:
            text_view_lines.append("  Cross-links:")
            for edge in sorted(cross_edges, key=lambda item: (item["query_similarity"], item["weight"]), reverse=True)[:6]:
                text_view_lines.append(
                    "    - Card [{source}] --{relation}/{weight:.2f}--> Card [{target}] | q_edge={q_edge:.3f}".format(
                        source=edge["source_card_id"],
                        relation=edge["relation"],
                        weight=float(edge["weight"]),
                        target=edge["target_card_id"],
                        q_edge=float(edge["query_similarity"]),
                    )
                )

    graph_view_lines = ["Relational Context (node-edge topology summary):"]
    for edge in sorted(selected_edges, key=lambda item: (item["query_similarity"], item["weight"]), reverse=True)[:18]:
        graph_view_lines.append(
            "- Card [{source}] --{relation}/{weight:.2f}--> Card [{target}] | q_edge={q_edge:.3f} | {reason}".format(
                source=edge["source_card_id"],
                relation=edge["relation"],
                weight=float(edge["weight"]),
                target=edge["target_card_id"],
                q_edge=float(edge["query_similarity"]),
                reason=edge["reason"],
            )
        )

    materialized = dict(method_context)
    materialized["selected_edges"] = selected_edges
    materialized["selected_ego_graphs"] = selected_ego_graphs
    materialized["text_view"] = "\n".join(text_view_lines)
    materialized["graph_view"] = "\n".join(graph_view_lines)
    return materialized


def build_reclaim_sentence_pool(retrieved_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool = []
    sentence_id = 1
    for node in retrieved_nodes:
        candidate_sentences = []
        for sentence in split_text_sentences(node.get("text") or ""):
            if len(sentence.split()) < 7:
                continue
            candidate_sentences.append(sentence)
            if len(candidate_sentences) >= RECLAIM_MAX_SENTENCES_PER_CARD:
                break
        for sentence_index, sentence in enumerate(candidate_sentences, start=1):
            pool.append(
                {
                    "sentence_id": sentence_id,
                    "card_id": int(node.get("card_id", 0) or 0),
                    "source": node.get("source") or "Unknown",
                    "heading": node.get("heading") or node.get("title") or "Untitled",
                    "sentence_index": sentence_index,
                    "text": sentence,
                }
            )
            sentence_id += 1
            if len(pool) >= RECLAIM_MAX_SENTENCE_POOL:
                return pool
    return pool


def format_reclaim_history(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "No previous reference-claim pairs yet."
    lines = ["Previous reference-claim pairs:"]
    for idx, step in enumerate(steps, start=1):
        lines.append(
            "- Pair {idx} | refs={refs}".format(
                idx=idx,
                refs=step.get("reference_ids", []),
            )
        )
        lines.append("  Claim: {}".format(step.get("claim", "")))
    return "\n".join(lines)


def bootstrap_reclaim_reference_plan(question: str, sentence_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    card_to_sentences: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    card_order = []
    for item in sentence_pool:
        card_id = int(item["card_id"])
        if card_id not in card_to_sentences:
            card_order.append(card_id)
        card_to_sentences[card_id].append(item)

    plans = []
    for card_id in card_order:
        plans.append({"references": [card_to_sentences[card_id][0]]})
        if len(plans) >= RECLAIM_MAX_STEPS:
            break
    return plans


def select_reclaim_references(
    client: OpenAI,
    model: str,
    question: str,
    sentence_pool: List[Dict[str, Any]],
    previous_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_lines = ["Candidate reference sentences:"]
    for item in sentence_pool:
        candidate_lines.append(
            "[S{sid}] Card [{card}] | {source} | {text}".format(
                sid=item["sentence_id"],
                card=item["card_id"],
                source=item["source"],
                text=item["text"],
            )
        )

    system_prompt = (
        "You are an interleaving reference-then-claim generator that emits citations before writing each claim.\n"
        "Select the reference sentences (by sentence ID from the candidate pool) that will support the next claim.\n"
        "Use only sentence IDs from the provided pool. Do not reuse sentences already cited in previous steps unless strictly necessary.\n"
        "Continue selecting until the pool is sufficient to answer the question; choose 'stop' only when no remaining sentence adds new useful evidence.\n"
        "Return ONLY JSON with fields:\n"
        "{\n"
        '  "action": "continue" | "stop",\n'
        '  "reference_ids": [1, 2],\n'
        '  "rationale": "..."\n'
        "}"
    )
    user_prompt = "Question:\n{}\n\n{}\n\n{}".format(question, format_reclaim_history(previous_steps), "\n".join(candidate_lines))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
    except Exception as exc:
        return {"action": "stop", "reference_ids": [], "rationale": str(exc)}

    action = str(parsed.get("action", "stop")).strip().lower()
    if action not in {"continue", "stop"}:
        action = "stop"
    reference_ids = []
    for ref in parsed.get("reference_ids", []) or []:
        try:
            ref_int = int(ref)
        except Exception:
            continue
        if any(item["sentence_id"] == ref_int for item in sentence_pool) and ref_int not in reference_ids:
            reference_ids.append(ref_int)
    return {
        "action": action,
        "reference_ids": reference_ids,
        "rationale": parsed.get("rationale", ""),
    }


def generate_reclaim_claim(
    client: OpenAI,
    model: str,
    question: str,
    selected_refs: List[Dict[str, Any]],
    previous_steps: List[Dict[str, Any]],
) -> str:
    reference_text = []
    for item in selected_refs:
        reference_text.append(
            "Reference [S{sid}] from Card [{card}] | {source}\n{sentence}".format(
                sid=item["sentence_id"],
                card=item["card_id"],
                source=item["source"],
                sentence=item["text"],
            )
        )

    system_prompt = (
        "You are an interleaving reference-then-claim generator. The references have already been chosen; write the next claim.\n"
        "Use ONLY the selected references.\n"
        "Write exactly one concise answer sentence.\n"
        "The sentence must end with inline [Card X] citations using the provided card IDs.\n"
        "Do not add facts not supported by the selected references."
    )
    user_prompt = (
        "Question:\n{question}\n\n"
        "{history}\n\n"
        "Selected references:\n{refs}\n"
    ).format(
        question=question,
        history=format_reclaim_history(previous_steps),
        refs="\n\n".join(reference_text),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return "Claim generation failed: {}".format(exc)


def build_plan_prompt(question: str, anchor_block: str, trust_cards: str) -> str:
    return (
        "Question:\n"
        "{}\n\n"
        "{}\n\n"
        "{}\n\n"
        "Produce a compact planning JSON that identifies the strongest direct-answer evidence before writing the final answer."
    ).format(question, anchor_block, trust_cards)


def build_anchor_plan(
    client: OpenAI,
    question: str,
    anchor_block: str,
    trust_cards: str,
    planner_model: str,
) -> Dict[str, Any]:
    system_prompt = (
        "You are an evidence planner for political RAG.\n"
        "Your job is to read community anchors and trust-aware evidence cards, then create a compact evidence-weighted plan.\n\n"
        "Rules:\n"
        "1. Community anchors are candidate backbones, not mandatory backbones.\n"
        "2. Prioritize cards that most directly answer the question, especially when the question asks about empirical effects, causes, or outcomes.\n"
        "3. Then prioritize higher S_trust, stronger query relevance, and stronger anchor support.\n"
        "4. If a card is mainly policy background, public perception, or indirect context, do not treat it as a main answer point unless direct evidence is missing.\n"
        "5. Do not force equal weight across perspectives when evidence quality is asymmetric.\n"
        "6. Preserve major disagreements if they are supported by meaningful evidence.\n"
        "7. CRITICAL: Only populate \"main_disagreements\" when the cards themselves contain explicitly competing positions on the question. If the cards are one-sided, broadly consensual, or only differ in topical focus rather than on the answer, return an empty list for \"main_disagreements\". Do NOT fabricate disagreement, invent competing studies, or infer controversy that is not directly evidenced in the cited cards.\n"
        "8. Every \"card_ids\" entry must point to a card that directly supports the associated claim or issue. Do not attach card IDs to disagreement issues unless those cards actually voice that disagreement.\n"
        "9. Do NOT add issues or evidence_gaps that comment on source bias, ideological leaning, balance of perspectives, or what viewpoints are missing from the retrieval set (e.g. 'limited evidence from right-leaning sources', 'no liberal perspective in the cards', 'this side is underrepresented'). Such absence-of-source claims are out of scope.\n"
        "10. Separate direct answer points, indirect context, and evidence gaps.\n"
        "11. Use only the card IDs that appear in the provided context.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "direct_answer_points": [{"claim": "...", "card_ids": [1,2], "why_direct": "..."}],\n'
        '  "most_supported_points": [{"claim": "...", "card_ids": [1,2]}],\n'
        '  "main_disagreements": [{"issue": "...", "card_ids": [3,4]}],\n'
        '  "context_only_cards": [{"card_id": 5, "reason": "..."}],\n'
        '  "evidence_gaps": ["..."],\n'
        '  "answer_strategy": "..."\n'
        "}"
    )

    try:
        response = client.chat.completions.create(
            model=planner_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_plan_prompt(question, anchor_block, trust_cards)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        return {
            "direct_answer_points": parsed.get("direct_answer_points", []),
            "most_supported_points": parsed.get("most_supported_points", []),
            "main_disagreements": parsed.get("main_disagreements", []),
            "context_only_cards": parsed.get("context_only_cards", []),
            "evidence_gaps": parsed.get("evidence_gaps", []),
            "answer_strategy": parsed.get("answer_strategy", ""),
        }
    except Exception as exc:
        return {
            "direct_answer_points": [],
            "most_supported_points": [],
            "main_disagreements": [],
            "context_only_cards": [],
            "evidence_gaps": ["Planning failed: {}".format(exc)],
            "answer_strategy": "Fallback to direct anchor-aware synthesis using the strongest cards first.",
        }


def format_anchor_plan(plan: Dict[str, Any]) -> str:
    lines = ["Evidence Plan:"]

    direct_points = plan.get("direct_answer_points", []) or []
    if direct_points:
        lines.append("Direct answer points:")
        for item in direct_points:
            lines.append(
                "- {} | cards={} | why_direct={}".format(
                    item.get("claim", "[missing claim]"),
                    item.get("card_ids", []),
                    item.get("why_direct", ""),
                )
            )

    supported = plan.get("most_supported_points", []) or []
    if supported:
        lines.append("Most supported points:")
        for item in supported:
            lines.append("- {} | cards={}".format(item.get("claim", "[missing claim]"), item.get("card_ids", [])))

    disagreements = plan.get("main_disagreements", []) or []
    if disagreements:
        lines.append("Main disagreements:")
        for item in disagreements:
            lines.append("- {} | cards={}".format(item.get("issue", "[missing issue]"), item.get("card_ids", [])))

    gaps = plan.get("evidence_gaps", []) or []
    if gaps:
        lines.append("Evidence gaps:")
        for item in gaps:
            lines.append("- {}".format(item))

    context_only_cards = plan.get("context_only_cards", []) or []
    if context_only_cards:
        lines.append("Context-only cards:")
        for item in context_only_cards:
            lines.append("- Card [{}] | {}".format(item.get("card_id", "?"), item.get("reason", "")))

    strategy = (plan.get("answer_strategy", "") or "").strip()
    if strategy:
        lines.append("Answer strategy: {}".format(strategy))
    return "\n".join(lines)


def extract_card_ids_from_text(text: str) -> List[int]:
    ids = []
    for match in re.findall(r"\[Card\s+(\d+)\]", text or ""):
        try:
            card_id = int(match)
        except Exception:
            continue
        if card_id not in ids:
            ids.append(card_id)
    return ids


def normalize_card_ids(values: Sequence[Any], valid_card_ids: Sequence[int]) -> List[int]:
    valid = set(int(card_id) for card_id in valid_card_ids)
    cleaned = []
    for value in values or []:
        try:
            card_id = int(value)
        except Exception:
            continue
        if card_id in valid and card_id not in cleaned:
            cleaned.append(card_id)
    return cleaned


def ensure_card_citations(text: str, card_ids: Sequence[int]) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return ""
    if extract_card_ids_from_text(cleaned):
        return cleaned
    if not card_ids:
        return cleaned
    citation_block = " ".join("[Card {}]".format(card_id) for card_id in card_ids)
    if cleaned.endswith((".", "!", "?")):
        cleaned = cleaned[:-1].rstrip()
    return "{} {}".format(cleaned, citation_block)


def collect_trustrag_selected_card_ids(plan: Dict[str, Any], retrieved_nodes: List[Dict[str, Any]]) -> List[int]:
    selected = []
    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in retrieved_nodes]
    for section in ("direct_answer_points", "most_supported_points", "main_disagreements"):
        for item in plan.get(section, []) or []:
            for card_id in normalize_card_ids(item.get("card_ids", []), valid_card_ids):
                if card_id not in selected:
                    selected.append(card_id)

    if len(selected) < TRUSTRAG_SELECTED_CARD_LIMIT:
        ranked_nodes = sorted(
            retrieved_nodes,
            key=lambda node: (
                0.55 * float(node.get("direct_answer_score", 0.0) or 0.0)
                + 0.25 * float(node.get("query_relevance", 0.0) or 0.0)
                + 0.20 * float(node.get("S_trust", 0.0) or 0.0)
            ),
            reverse=True,
        )
        for node in ranked_nodes:
            card_id = int(node.get("card_id", 0) or 0)
            if card_id not in selected:
                selected.append(card_id)
            if len(selected) >= TRUSTRAG_SELECTED_CARD_LIMIT:
                break
    return selected[:TRUSTRAG_SELECTED_CARD_LIMIT]


def fallback_trustrag_claim_units(
    plan: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    claim_units = []
    seen_texts = set()
    for section, slot in [
        ("direct_answer_points", "supported"),
        ("most_supported_points", "supported"),
        ("main_disagreements", "disagreement"),
    ]:
        key = "claim" if section != "main_disagreements" else "issue"
        for item in plan.get(section, []) or []:
            text = " ".join(str(item.get(key, "")).split()).strip()
            if not text or text in seen_texts:
                continue
            card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
            if not card_ids:
                continue
            claim_units.append(
                {
                    "slot": slot,
                    "text": ensure_card_citations(text, card_ids),
                    "card_ids": card_ids,
                }
            )
            seen_texts.add(text)
            if len(claim_units) >= TRUSTRAG_MAX_CLAIM_UNITS:
                return claim_units

    return claim_units[:TRUSTRAG_MAX_CLAIM_UNITS]


def draft_trustrag_claim_units(
    client: OpenAI,
    model: str,
    question: str,
    selected_nodes: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    fallback_support_ids = [
        int(node.get("card_id", 0) or 0)
        for node in sorted(
            selected_nodes,
            key=lambda node: (
                float(node.get("direct_answer_score", 0.0) or 0.0),
                float(node.get("query_relevance", 0.0) or 0.0),
                float(node.get("S_trust", 0.0) or 0.0),
            ),
            reverse=True,
        )[:2]
        if int(node.get("card_id", 0) or 0) > 0
    ]
    context = build_trust_evidence_cards(question, selected_nodes)
    system_prompt = (
        "You are drafting compact claim units for trust-aware answer generation.\n\n"
        "Rules:\n"
        "1. Generate 3 to 5 short claim units that directly answer the question.\n"
        "2. Prefer direct empirical evidence over broad policy background or public-perception context.\n"
        "3. Use indirect cards only to explain disagreement, uncertainty, or evidence limits.\n"
        "4. Each claim unit must contain one main claim and end with inline [Card X] citations.\n"
        "5. Keep each claim unit concise and highly grounded. Avoid uncited factual generalizations.\n"
        "6. If direct evidence is limited, include one explicit limitation claim.\n"
        "7. CRITICAL: Only emit a \"disagreement\" slot when at least one cited card explicitly takes a competing position on the question. If the cards do not actually disagree, do NOT emit any disagreement slot. Never fabricate \"some argue\", \"studies disagree\", \"there is debate\", or similar meta-claims that are not directly visible in the cards.\n"
        "8. Do NOT emit any claim about source bias, missing perspectives, ideological balance, or what viewpoints are absent from the retrieved cards (e.g. 'limited evidence from right-leaning sources', 'no liberal viewpoint shown', 'this side is underrepresented'). Stick to claims about the topic that the cards directly support.\n"
        "9. Every [Card X] citation in a claim must point to a card whose text explicitly supports that exact claim. Do not attach citations cosmetically.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "claims": [\n'
        '    {"slot": "supported|disagreement|limitation", "text": "...", "card_ids": [1,2]}\n'
        "  ]\n"
        "}"
    )
    user_prompt = "Question:\n{}\n\n{}\n\nSelected cards:\n{}".format(question, format_anchor_plan(plan), context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        claim_units = []
        for item in parsed.get("claims", []) or []:
            slot = str(item.get("slot", "supported")).strip().lower()
            if slot not in {"supported", "disagreement", "limitation"}:
                slot = "supported"
            text = " ".join(str(item.get("text", "")).split()).strip()
            card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
            if not card_ids:
                card_ids = normalize_card_ids(extract_card_ids_from_text(text), valid_card_ids)
            if not card_ids and slot == "limitation":
                card_ids = fallback_support_ids
            text = ensure_card_citations(text, card_ids)
            if not text:
                continue
            claim_units.append({"slot": slot, "text": text, "card_ids": card_ids})
            if len(claim_units) >= TRUSTRAG_MAX_CLAIM_UNITS:
                break
        if claim_units:
            return claim_units
    except Exception:
        pass
    return fallback_trustrag_claim_units(plan, selected_nodes)


def assemble_trustrag_answer(
    client: OpenAI,
    model: str,
    question: str,
    claim_units: List[Dict[str, Any]],
) -> str:
    system_prompt = (
        "You are assembling a final answer from already grounded claim units.\n"
        "Use ONLY the provided claim units.\n"
        "Do not add new facts.\n"
        "Do not introduce any disagreement, controversy, source-bias commentary, ideological-balance commentary, or competing-view framing that is not present in the supplied claim units. In particular, do not add claims about which perspectives or sources are missing from the retrieval.\n"
        "Start by directly answering the question rather than drifting into general background.\n"
        "Keep the inline [Card X] citations unchanged. Do not invent new [Card X] references.\n"
        "Every factual bullet should retain inline citations when the claim units provide them.\n"
        "If no claim unit describes a disagreement, omit that section entirely — do NOT manufacture disagreement and do NOT write placeholder text such as 'evidence is one-sided' or 'no disagreement appears'.\n"
        "Be concise.\n\n"
        "Output format:\n"
        "- Most supported points\n"
        "- Main disagreement or uncertainty\n"
        "    (Include this section ONLY if at least one disagreement claim unit was supplied. Otherwise omit it.)\n"
        "- Bottom line"
    )
    unit_lines = []
    for idx, item in enumerate(claim_units, start=1):
        unit_lines.append(
            "Claim Unit {idx} | slot={slot} | cards={cards}\n{claim}".format(
                idx=idx,
                slot=item.get("slot", "supported"),
                cards=item.get("card_ids", []),
                claim=item.get("text", ""),
            )
        )
    user_prompt = "Question:\n{}\n\nClaim units:\n{}".format(question, "\n\n".join(unit_lines))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        supported = [item["text"] for item in claim_units if item.get("slot") == "supported"]
        disagreements = [item["text"] for item in claim_units if item.get("slot") == "disagreement"]
        limitations = [item["text"] for item in claim_units if item.get("slot") == "limitation"]
        lines = []
        lines.extend(supported[:3])
        lines.extend(disagreements[:2])
        lines.extend(limitations[:1])
        return "\n".join(lines)


def generate_trustrag_anchor_answer(
    client: OpenAI,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
    generator_model: str,
    planner_model: str,
) -> Dict[str, Any]:
    annotate_question_directness(question, retrieved_nodes, community_anchors)
    anchor_block = build_anchor_block(community_anchors)
    trust_cards = build_trust_evidence_cards(question, retrieved_nodes)
    plan = build_anchor_plan(client, question, anchor_block, trust_cards, planner_model)
    selected_card_ids = collect_trustrag_selected_card_ids(plan, retrieved_nodes)
    selected_nodes = [node for node in retrieved_nodes if int(node.get("card_id", 0) or 0) in selected_card_ids]
    claim_units = draft_trustrag_claim_units(client, generator_model, question, selected_nodes, plan)
    answer = assemble_trustrag_answer(client, generator_model, question, claim_units)

    plan_with_claims = dict(plan)
    plan_with_claims["selected_card_ids"] = selected_card_ids
    plan_with_claims["claim_units"] = claim_units
    avg_directness = 0.0
    if selected_nodes:
        avg_directness = sum(float(node.get("direct_answer_score", 0.0) or 0.0) for node in selected_nodes) / len(selected_nodes)

    return {
        "answer": answer,
        "plan": plan_with_claims,
        "metadata": {
            "prompt_style": "anchor_directness_claim_generation",
            "selected_card_count": len(selected_nodes),
            "claim_unit_count": len(claim_units),
            "avg_selected_direct_answer_score": round(avg_directness, 4),
        },
    }


def build_hybrid_v5_unit(slot: str, text: str, card_ids: Sequence[int]) -> Dict[str, Any]:
    return {
        "slot": slot,
        "text": ensure_card_citations(text, card_ids),
        "card_ids": list(card_ids),
        "verification": {"label": "unverified", "explanation": ""},
    }


def select_hybrid_v5_nodes(
    question: str,
    plan: Dict[str, Any],
    retrieved_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    node_by_card_id = {int(node.get("card_id", 0) or 0): node for node in retrieved_nodes}
    ordered_card_ids: List[int] = []

    for section in ("direct_answer_points", "most_supported_points"):
        for item in plan.get(section, []) or []:
            for card_id in normalize_card_ids(item.get("card_ids", []), node_by_card_id.keys()):
                if card_id not in ordered_card_ids:
                    ordered_card_ids.append(card_id)

    if question_requests_conflict(question):
        for item in plan.get("main_disagreements", []) or []:
            for card_id in normalize_card_ids(item.get("card_ids", []), node_by_card_id.keys()):
                if card_id not in ordered_card_ids:
                    ordered_card_ids.append(card_id)
                if len(ordered_card_ids) >= HYBRID_V5_SELECTED_CARD_LIMIT:
                    break

    ranked_nodes = sorted(
        retrieved_nodes,
        key=lambda node: (
            0.42 * float(node.get("direct_answer_score", 0.0) or 0.0)
            + 0.28 * float(node.get("query_relevance", 0.0) or 0.0)
            + 0.20 * float(node.get("S_trust", 0.0) or 0.0)
            + 0.10 * float(node.get("anchor_score", 0.0) or 0.0)
        ),
        reverse=True,
    )
    for node in ranked_nodes:
        card_id = int(node.get("card_id", 0) or 0)
        if card_id not in ordered_card_ids:
            ordered_card_ids.append(card_id)
        if len(ordered_card_ids) >= HYBRID_V5_SELECTED_CARD_LIMIT:
            break

    return [node_by_card_id[card_id] for card_id in ordered_card_ids if card_id in node_by_card_id][:HYBRID_V5_SELECTED_CARD_LIMIT]


def fallback_hybrid_v5_units(
    question: str,
    plan: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    supported_units = []
    uncertainty_units = []
    seen = set()

    for section in ("direct_answer_points", "most_supported_points"):
        for item in plan.get(section, []) or []:
            text = " ".join(str(item.get("claim", "")).split()).strip()
            card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
            if not text or not card_ids or text in seen:
                continue
            supported_units.append(build_hybrid_v5_unit("supported", text, card_ids))
            seen.add(text)
            if len(supported_units) >= HYBRID_V5_MAX_SUPPORTED:
                break
        if len(supported_units) >= HYBRID_V5_MAX_SUPPORTED:
            break

    for item in plan.get("main_disagreements", []) or []:
        text = " ".join(str(item.get("issue", "")).split()).strip()
        card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
        if not text or not card_ids or text in seen:
            continue
        uncertainty_units.append(build_hybrid_v5_unit("uncertainty", text, card_ids))
        seen.add(text)
        if len(uncertainty_units) >= HYBRID_V5_MAX_UNCERTAINTIES:
            break

    if not uncertainty_units and plan.get("evidence_gaps"):
        top_cards = [int(node.get("card_id", 0) or 0) for node in selected_nodes[:2] if int(node.get("card_id", 0) or 0) > 0]
        uncertainty_units.append(
            build_hybrid_v5_unit(
                "uncertainty",
                "Available evidence is partial on this aspect.",
                top_cards,
            )
        )

    if supported_units:
        bottom_text = strip_inline_citations(supported_units[0]["text"])
        if uncertainty_units:
            bottom_text = "{} The broader conclusion remains contested or incomplete.".format(bottom_text.rstrip("."))
        bottom_unit = build_hybrid_v5_unit("bottom_line", bottom_text, supported_units[0]["card_ids"])
    elif selected_nodes:
        top_card = int(selected_nodes[0].get("card_id", 0) or 0)
        bottom_unit = build_hybrid_v5_unit(
            "bottom_line",
            "The retrieved reporting provides only partial evidence, so the answer should remain cautious.",
            [top_card] if top_card > 0 else [],
        )
    else:
        bottom_unit = build_hybrid_v5_unit("bottom_line", "Available evidence is partial on this question.", [])

    return {
        "supported_points": supported_units,
        "uncertainties": uncertainty_units,
        "bottom_line": bottom_unit,
    }


def draft_hybrid_v5_units(
    client: OpenAI,
    model: str,
    question: str,
    selected_nodes: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    if not selected_nodes:
        return {
            "supported_points": [],
            "uncertainties": [],
            "bottom_line": build_hybrid_v5_unit("bottom_line", "Available evidence is partial on this question.", []),
        }

    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    context = build_trust_evidence_cards(question, selected_nodes)
    system_prompt = (
        "You are drafting a direct, useful answer from trust-aware evidence cards.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided evidence cards.\n"
        "2. Answer the question directly. Do not start with meta commentary about evidence limitations.\n"
        "3. Put the strongest direct evidence first.\n"
        "4. Use broad policy background only when it is necessary to explain uncertainty.\n"
        "5. Keep each sentence concise and substantive.\n"
        "6. Every sentence must end with inline [Card X] citations.\n"
        "7. Limitation language should be brief and appear mainly in uncertainty or bottom-line sentences.\n"
        "8. Avoid repetitive phrases such as 'the provided cards' or 'does not fully settle the question'.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "supported_points": [{"text": "...", "card_ids": [1,2]}],\n'
        '  "uncertainties": [{"text": "...", "card_ids": [3,4]}],\n'
        '  "bottom_line": {"text": "...", "card_ids": [1,3]}\n'
        "}"
    )
    user_prompt = "Question:\n{}\n\n{}\n\nPlanning hints:\n{}".format(question, context, format_anchor_plan(plan))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        supported_units = []
        uncertainty_units = []

        for item in (parsed.get("supported_points", []) or [])[:HYBRID_V5_MAX_SUPPORTED]:
            text = " ".join(str(item.get("text", "")).split()).strip()
            card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
            if not card_ids:
                card_ids = normalize_card_ids(extract_card_ids_from_text(text), valid_card_ids)
            if text and card_ids:
                supported_units.append(build_hybrid_v5_unit("supported", text, card_ids))

        for item in (parsed.get("uncertainties", []) or [])[:HYBRID_V5_MAX_UNCERTAINTIES]:
            text = " ".join(str(item.get("text", "")).split()).strip()
            card_ids = normalize_card_ids(item.get("card_ids", []), valid_card_ids)
            if not card_ids:
                card_ids = normalize_card_ids(extract_card_ids_from_text(text), valid_card_ids)
            if text and card_ids:
                uncertainty_units.append(build_hybrid_v5_unit("uncertainty", text, card_ids))

        bottom = parsed.get("bottom_line", {}) or {}
        bottom_text = " ".join(str(bottom.get("text", "")).split()).strip()
        bottom_card_ids = normalize_card_ids(bottom.get("card_ids", []), valid_card_ids)
        if not bottom_card_ids:
            bottom_card_ids = normalize_card_ids(extract_card_ids_from_text(bottom_text), valid_card_ids)
        if not bottom_text:
            bottom_text = "Available evidence is partial on this question."
        bottom_unit = build_hybrid_v5_unit(
            "bottom_line",
            bottom_text,
            bottom_card_ids or (supported_units[0]["card_ids"] if supported_units else []),
        )

        if supported_units:
            return {
                "supported_points": supported_units,
                "uncertainties": uncertainty_units,
                "bottom_line": bottom_unit,
            }
    except Exception:
        pass

    return fallback_hybrid_v5_units(question, plan, selected_nodes)


def sentence_doc_match_score(sentence: str, node: Dict[str, Any]) -> float:
    sentence_terms = set(extract_focus_terms(strip_inline_citations(sentence)))
    doc_terms = set(
        extract_focus_terms(
            "{} {} {}".format(
                node.get("heading") or node.get("title") or "",
                node.get("summary") or "",
                clean_snippet(node.get("text") or "", limit=125),
            )
        )
    )
    overlap = 0.0
    if sentence_terms and doc_terms:
        overlap = len(sentence_terms & doc_terms) / max(1, len(sentence_terms))
    return (
        0.45 * overlap
        + 0.20 * float(node.get("direct_answer_score", 0.0) or 0.0)
        + 0.20 * float(node.get("query_relevance", 0.0) or 0.0)
        + 0.15 * float(node.get("S_trust", 0.0) or 0.0)
    )


def select_hybrid_v5_evidence_docs(
    sentence: str,
    selected_nodes: List[Dict[str, Any]],
    preferred_card_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    by_card_id = {int(node.get("card_id", 0) or 0): node for node in selected_nodes}
    docs = [by_card_id[card_id] for card_id in preferred_card_ids if card_id in by_card_id]
    refs = extract_doc_refs(sentence)
    for ref in refs:
        if ref in by_card_id and by_card_id[ref] not in docs:
            docs.append(by_card_id[ref])
    if docs:
        return docs[:HYBRID_V5_AUDIT_DOC_LIMIT]

    ranked = sorted(
        selected_nodes,
        key=lambda node: sentence_doc_match_score(sentence, node),
        reverse=True,
    )
    return ranked[:HYBRID_V5_AUDIT_DOC_LIMIT]


def hybrid_v5_sentence_needs_rewrite(sentence: str, slot: str, verdict_label: str) -> bool:
    lowered = strip_inline_citations(sentence).lower()
    has_refs = bool(extract_doc_refs(sentence))
    risky = any(term in lowered for term in POLICY_HIGH_RISK_TERMS) or bool(re.search(r"\b\d+(?:\.\d+)?%?\b", lowered))
    if verdict_label == "unsupported":
        return True
    if slot == "bottom_line" and verdict_label != "supported":
        return True
    if not has_refs:
        return True
    if verdict_label == "unclear" and risky:
        return True
    return False


def rewrite_hybrid_v5_sentence(
    client: OpenAI,
    model: str,
    question: str,
    slot: str,
    sentence: str,
    evidence_docs: List[Dict[str, Any]],
) -> str:
    if not evidence_docs:
        return sentence

    evidence_blocks = []
    for doc in evidence_docs:
        evidence_blocks.append(
            "Doc [{}] (Source: {} | S_trust: {})\n{}".format(
                int(doc.get("card_id", 0) or 0),
                doc.get("source", "N/A"),
                doc.get("S_trust", "N/A"),
                (doc.get("text") or "")[:800],
            )
        )

    system_prompt = (
        "You are revising one answer sentence after a trust audit.\n"
        "Keep the sentence direct and useful, but only claim what the evidence can support.\n"
        "Prefer narrowing the claim over replacing it with generic meta commentary.\n"
        "Return exactly one sentence with inline [Card X] citations.\n"
        "Do not mention 'provided cards' or 'the retrieved evidence' unless absolutely necessary.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "text": "..."\n'
        "}"
    )
    user_prompt = (
        "Question:\n{question}\n\n"
        "Sentence role: {slot}\n\n"
        "Current sentence:\n{sentence}\n\n"
        "Evidence:\n{evidence}\n"
    ).format(
        question=question,
        slot=slot,
        sentence=sentence,
        evidence="\n\n".join(evidence_blocks),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        card_ids = [int(doc.get("card_id", 0) or 0) for doc in evidence_docs if int(doc.get("card_id", 0) or 0) > 0]
        rewritten = ensure_card_citations(str(parsed.get("text", "")).strip(), card_ids)
        return rewritten or sentence
    except Exception:
        return sentence


def fallback_hybrid_v5_sentence(slot: str, evidence_docs: List[Dict[str, Any]]) -> str:
    card_ids = [int(doc.get("card_id", 0) or 0) for doc in evidence_docs if int(doc.get("card_id", 0) or 0) > 0][:2]
    if slot == "bottom_line":
        text = "Overall, the reporting points in a real direction of evidence, but the broader conclusion remains unsettled."
    else:
        text = "The available reporting leaves this point unresolved."
    return ensure_card_citations(text, card_ids)


def audit_hybrid_v5_units(
    client: OpenAI,
    model: str,
    question: str,
    draft_units: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    audited_supported = []
    audited_uncertainties = []
    stats = {"supported": 0, "kept_unclear": 0, "rewritten": 0, "dropped": 0}

    def audit_one(unit: Dict[str, Any]) -> Dict[str, Any]:
        evidence_docs = select_hybrid_v5_evidence_docs(unit.get("text", ""), selected_nodes, unit.get("card_ids", []))
        if not evidence_docs:
            return {}
        verdict = check_sentence_support(client, model, unit.get("text", ""), evidence_docs)
        updated = dict(unit)
        updated["verification"] = verdict

        if verdict["label"] == "supported":
            stats["supported"] += 1
            return updated

        if not hybrid_v5_sentence_needs_rewrite(unit.get("text", ""), unit.get("slot", "supported"), verdict["label"]):
            stats["kept_unclear"] += 1
            return updated

        rewritten_text = rewrite_hybrid_v5_sentence(
            client=client,
            model=model,
            question=question,
            slot=unit.get("slot", "supported"),
            sentence=unit.get("text", ""),
            evidence_docs=evidence_docs,
        )
        rewritten = dict(updated)
        rewritten["text"] = rewritten_text
        rewritten["card_ids"] = extract_doc_refs(rewritten_text) or unit.get("card_ids", [])
        rewritten_verdict = check_sentence_support(client, model, rewritten["text"], evidence_docs)
        rewritten["verification"] = rewritten_verdict
        if rewritten_verdict["label"] == "supported":
            stats["rewritten"] += 1
            return rewritten
        if unit.get("slot") == "uncertainty":
            fallback = dict(updated)
            fallback["text"] = fallback_hybrid_v5_sentence("uncertainty", evidence_docs)
            fallback["card_ids"] = extract_doc_refs(fallback["text"]) or unit.get("card_ids", [])
            fallback["verification"] = {"label": "unclear", "explanation": "Fallback uncertainty sentence used."}
            stats["rewritten"] += 1
            return fallback
        if unit.get("slot") == "bottom_line":
            fallback = dict(updated)
            fallback["text"] = fallback_hybrid_v5_sentence("bottom_line", evidence_docs)
            fallback["card_ids"] = extract_doc_refs(fallback["text"]) or unit.get("card_ids", [])
            fallback["verification"] = {"label": "unclear", "explanation": "Fallback bottom line used."}
            stats["rewritten"] += 1
            return fallback

        stats["dropped"] += 1
        return {}

    for unit in draft_units.get("supported_points", []) or []:
        audited = audit_one(unit)
        if audited:
            audited_supported.append(audited)
    for unit in draft_units.get("uncertainties", []) or []:
        audited = audit_one(unit)
        if audited:
            audited_uncertainties.append(audited)

    bottom_unit = draft_units.get("bottom_line") or build_hybrid_v5_unit("bottom_line", "Available evidence is partial on this question.", [])
    audited_bottom = audit_one(bottom_unit)
    if not audited_bottom:
        fallback_docs = select_hybrid_v5_evidence_docs(
            bottom_unit.get("text", ""),
            selected_nodes,
            bottom_unit.get("card_ids", []),
        )
        fallback_text = fallback_hybrid_v5_sentence("bottom_line", fallback_docs)
        audited_bottom = build_hybrid_v5_unit("bottom_line", fallback_text, extract_doc_refs(fallback_text))

    if not audited_supported and selected_nodes:
        top_node = sorted(
            selected_nodes,
            key=lambda node: (
                float(node.get("direct_answer_score", 0.0) or 0.0),
                float(node.get("query_relevance", 0.0) or 0.0),
                float(node.get("S_trust", 0.0) or 0.0),
            ),
            reverse=True,
        )[0]
        top_card = int(top_node.get("card_id", 0) or 0)
        audited_supported.append(
            build_hybrid_v5_unit(
                "supported",
                "{} is the strongest directly relevant evidence in the retrieval set.".format(
                    top_node.get("heading") or top_node.get("title") or "This article"
                ),
                [top_card] if top_card > 0 else [],
            )
        )

    return {
        "supported_points": audited_supported[:HYBRID_V5_MAX_SUPPORTED],
        "uncertainties": audited_uncertainties[:HYBRID_V5_MAX_UNCERTAINTIES],
        "bottom_line": audited_bottom,
        "stats": stats,
    }


def assemble_hybrid_v5_answer(units: Dict[str, Any]) -> str:
    lines = ["- Most supported points"]
    supported_points = units.get("supported_points", []) or []
    if supported_points:
        for item in supported_points[:HYBRID_V5_MAX_SUPPORTED]:
            lines.append("  - {}".format(item.get("text", "")))
    else:
        lines.append("  - Available evidence is partial on this question.")

    lines.append("- Main disagreement or uncertainty")
    uncertainties = units.get("uncertainties", []) or []
    if uncertainties:
        for item in uncertainties[:HYBRID_V5_MAX_UNCERTAINTIES]:
            lines.append("  - {}".format(item.get("text", "")))
    else:
        bottom = units.get("bottom_line", {}) or {}
        if bottom.get("text"):
            lines.append("  - {}".format(bottom.get("text", "")))

    lines.append("- Bottom line")
    bottom_line = units.get("bottom_line", {}) or {}
    lines.append("  - {}".format(bottom_line.get("text", "Available evidence is partial on this question.")))
    return "\n".join(lines)


def generate_trustrag_hybrid_v5_answer(
    client: OpenAI,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
    generator_model: str,
    planner_model: str,
) -> Dict[str, Any]:
    annotate_question_directness(question, retrieved_nodes, community_anchors)
    anchor_block = build_anchor_block(community_anchors)
    trust_cards = build_trust_evidence_cards(question, retrieved_nodes)
    plan = build_anchor_plan(client, question, anchor_block, trust_cards, planner_model)
    selected_nodes = select_hybrid_v5_nodes(question, plan, retrieved_nodes)
    draft_units = draft_hybrid_v5_units(client, generator_model, question, selected_nodes, plan)
    audited_units = audit_hybrid_v5_units(client, planner_model, question, draft_units, selected_nodes)
    answer = assemble_hybrid_v5_answer(audited_units)

    avg_directness = 0.0
    if selected_nodes:
        avg_directness = sum(float(node.get("direct_answer_score", 0.0) or 0.0) for node in selected_nodes) / len(selected_nodes)

    plan_payload = dict(plan)
    plan_payload["selected_card_ids"] = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    plan_payload["draft_units"] = draft_units
    plan_payload["audited_units"] = audited_units

    return {
        "answer": answer,
        "plan": plan_payload,
        "metadata": {
            "prompt_style": "hybrid_trust_audit_v5",
            "selected_card_count": len(selected_nodes),
            "avg_selected_direct_answer_score": round(avg_directness, 4),
            "audit_stats": audited_units.get("stats", {}),
        },
    }


def normalize_claim_role(role: Any) -> str:
    normalized = str(role or "supporting").strip().lower()
    if normalized in {"support", "pro", "positive"}:
        return "supporting"
    if normalized in {"counter", "against", "negative", "opposing"}:
        return "counter"
    if normalized not in {"supporting", "counter", "mixed", "background"}:
        return "supporting"
    return normalized


def normalize_support_strength(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    if normalized not in {"high", "medium", "low"}:
        return "medium"
    return normalized


def strip_inline_citations(text: str) -> str:
    return re.sub(r"\s*\[Card\s+\d+\]", "", text or "").strip()


def claim_token_jaccard(text_a: str, text_b: str) -> float:
    tokens_a = set(extract_focus_terms(strip_inline_citations(text_a)))
    tokens_b = set(extract_focus_terms(strip_inline_citations(text_b)))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_claim_query_alignment(question: str, claim_text: str, card_directness: float) -> float:
    question_terms = extract_focus_terms(question)
    claim_terms = set(extract_focus_terms(claim_text))
    if not question_terms or not claim_terms:
        return round(0.5 * card_directness, 4)
    overlap = sum(1 for term in question_terms if term in claim_terms) / len(question_terms)
    return round(min(1.0, (0.55 * overlap) + (0.45 * card_directness)), 4)


def score_claim_candidate(candidate: Dict[str, Any]) -> float:
    return round(
        (0.45 * float(candidate.get("claim_query_alignment", 0.0) or 0.0))
        + (0.30 * float(candidate.get("card_s_trust", 0.0) or 0.0))
        + (0.15 * float(candidate.get("support_strength_score", 0.0) or 0.0))
        + (0.10 * float(candidate.get("anchor_bonus", 0.0) or 0.0)),
        4,
    )


def fallback_claim_candidates(
    question: str,
    plan: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    node_by_card_id = {int(node.get("card_id", 0) or 0): node for node in selected_nodes}
    candidates = []
    seq = 1
    for section, key, role in [
        ("direct_answer_points", "claim", "supporting"),
        ("most_supported_points", "claim", "supporting"),
        ("main_disagreements", "issue", "counter"),
    ]:
        for item in plan.get(section, []) or []:
            card_ids = normalize_card_ids(item.get("card_ids", []), node_by_card_id.keys())
            if not card_ids:
                continue
            card = node_by_card_id.get(card_ids[0])
            if card is None:
                continue
            claim_text = " ".join(str(item.get(key, "")).split()).strip()
            if not claim_text:
                continue
            support_strength = "high" if section == "direct_answer_points" else "medium"
            candidate = {
                "claim_id": "fallback-{}".format(seq),
                "claim_text": claim_text,
                "claim_type": "unknown",
                "stance_role": role,
                "support_strength": support_strength,
                "support_strength_score": CLAIM_SUPPORT_STRENGTH[support_strength],
                "evidence_span": clean_snippet(card.get("text") or "", 55),
                "needs_caveat": bool(plan.get("evidence_gaps")),
                "card_ids": card_ids,
                "source_card_id": card_ids[0],
                "source": card.get("source") or "Unknown",
                "community_id": card.get("community", -1),
                "claim_query_alignment": compute_claim_query_alignment(
                    question,
                    claim_text,
                    float(card.get("direct_answer_score", 0.0) or 0.0),
                ),
                "card_s_trust": float(card.get("S_trust", 0.0) or 0.0),
                "anchor_bonus": 1.0 if bool(card.get("is_community_anchor")) else 0.0,
                "is_anchor_backed": bool(card.get("is_community_anchor")),
            }
            candidate["claim_score"] = score_claim_candidate(candidate)
            candidates.append(candidate)
            seq += 1
            if len(candidates) >= CLAIM_GRAPH_MAX_TOTAL_CLAIMS:
                return candidates
    return candidates


def extract_claim_candidates(
    client: OpenAI,
    model: str,
    question: str,
    selected_nodes: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not selected_nodes:
        return []

    valid_card_ids = [int(node.get("card_id", 0) or 0) for node in selected_nodes]
    node_by_card_id = {int(node.get("card_id", 0) or 0): node for node in selected_nodes}
    context = build_trust_evidence_cards(question, selected_nodes)
    system_prompt = (
        "You are extracting claim-level intermediate representations from evidence cards.\n\n"
        "Rules:\n"
        "1. For each card, extract at most 2 answer-relevant claims.\n"
        "2. Prefer descriptive, verifiable claims over broad political framing.\n"
        "3. If a card is mostly background, return an empty claim list for that card.\n"
        "4. Keep claim_text short and faithful to the card.\n"
        "5. evidence_span must be a short snippet from the card that supports the claim.\n"
        "6. stance_role must be one of supporting, counter, mixed, background.\n"
        "7. support_strength must be one of high, medium, low.\n"
        "8. needs_caveat should be true when the claim is conditional, narrow, or only indirectly supported.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "cards": [\n'
        '    {"card_id": 1, "claims": [\n'
        '      {"claim_text": "...", "claim_type": "...", "stance_role": "supporting", "support_strength": "high", "evidence_span": "...", "needs_caveat": false}\n'
        "    ]}\n"
        "  ]\n"
        "}"
    )
    user_prompt = "Question:\n{}\n\n{}\n\nPlanning hints:\n{}".format(question, context, format_anchor_plan(plan))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        candidates = []
        seq = 1
        for card_item in parsed.get("cards", []) or []:
            try:
                card_id = int(card_item.get("card_id"))
            except Exception:
                continue
            if card_id not in valid_card_ids:
                continue
            node = node_by_card_id[card_id]
            for claim in (card_item.get("claims", []) or [])[:CLAIM_GRAPH_MAX_CARD_CLAIMS]:
                claim_text = " ".join(str(claim.get("claim_text", "")).split()).strip()
                if not claim_text:
                    continue
                stance_role = normalize_claim_role(claim.get("stance_role"))
                support_strength = normalize_support_strength(claim.get("support_strength"))
                candidate = {
                    "claim_id": "{}-{}".format(card_id, seq),
                    "claim_text": claim_text,
                    "claim_type": str(claim.get("claim_type", "unknown")).strip() or "unknown",
                    "stance_role": stance_role,
                    "support_strength": support_strength,
                    "support_strength_score": CLAIM_SUPPORT_STRENGTH[support_strength],
                    "evidence_span": clean_snippet(str(claim.get("evidence_span", "")).strip() or (node.get("text") or ""), 55),
                    "needs_caveat": bool(claim.get("needs_caveat", False)),
                    "card_ids": [card_id],
                    "source_card_id": card_id,
                    "source": node.get("source") or "Unknown",
                    "community_id": node.get("community", -1),
                    "claim_query_alignment": compute_claim_query_alignment(
                        question,
                        claim_text,
                        float(node.get("direct_answer_score", 0.0) or 0.0),
                    ),
                    "card_s_trust": float(node.get("S_trust", 0.0) or 0.0),
                    "anchor_bonus": 1.0 if bool(node.get("is_community_anchor")) else 0.0,
                    "is_anchor_backed": bool(node.get("is_community_anchor")),
                }
                candidate["claim_score"] = score_claim_candidate(candidate)
                candidates.append(candidate)
                seq += 1
                if len(candidates) >= CLAIM_GRAPH_MAX_TOTAL_CLAIMS:
                    return candidates
        if candidates:
            return candidates
    except Exception:
        pass
    return fallback_claim_candidates(question, plan, selected_nodes)


def claim_candidate_to_unit(candidate: Dict[str, Any], slot: str) -> Dict[str, Any]:
    base_text = candidate.get("claim_text", "")
    if candidate.get("needs_caveat") and slot == "supported":
        base_text = "{} (with important caveats and limits).".format(strip_inline_citations(base_text).rstrip("."))
    return {
        "slot": slot,
        "text": ensure_card_citations(base_text, candidate.get("card_ids", [])),
        "card_ids": list(candidate.get("card_ids", [])),
        "claim_text": candidate.get("claim_text", ""),
        "claim_score": float(candidate.get("claim_score", 0.0) or 0.0),
        "evidence_span": candidate.get("evidence_span", ""),
        "source_card_id": candidate.get("source_card_id"),
        "source": candidate.get("source", "Unknown"),
        "claim_type": candidate.get("claim_type", "unknown"),
        "verification": {"label": "unverified", "explanation": ""},
    }


def question_requests_conflict(question: str) -> bool:
    question_lower = (question or "").lower()
    return any(term in question_lower for term in POLICY_CONFLICT_TERMS)


def question_needs_direct_evidence(question: str) -> bool:
    question_lower = (question or "").lower()
    return any(term in question_lower for term in POLICY_DIRECT_EVIDENCE_TERMS)


def build_policy_limitation_unit(text: str, card_ids: Sequence[int]) -> Dict[str, Any]:
    cleaned = ensure_card_citations(text, card_ids)
    return {
        "slot": "limitation",
        "text": cleaned,
        "card_ids": list(card_ids),
        "claim_text": strip_inline_citations(cleaned),
        "claim_score": 0.0,
        "evidence_span": "",
        "source_card_id": card_ids[0] if card_ids else None,
        "source": "multiple",
        "claim_type": "policy_limitation",
        "verification": {"label": "unverified", "explanation": ""},
    }


def compute_policy_evidence_status(
    question: str,
    plan: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    direct_nodes = sorted(
        selected_nodes,
        key=lambda node: float(node.get("direct_answer_score", 0.0) or 0.0),
        reverse=True,
    )
    top_direct_nodes = [node for node in direct_nodes[:3] if float(node.get("direct_answer_score", 0.0) or 0.0) > 0.0]
    avg_top_directness = 0.0
    if top_direct_nodes:
        avg_top_directness = sum(float(node.get("direct_answer_score", 0.0) or 0.0) for node in top_direct_nodes) / len(top_direct_nodes)
    direct_card_ids = [
        int(node.get("card_id", 0) or 0)
        for node in direct_nodes
        if float(node.get("direct_answer_score", 0.0) or 0.0) >= POLICY_DIRECTNESS_THRESHOLD
    ]
    gap_count = len(plan.get("evidence_gaps", []) or [])
    direct_point_count = len(plan.get("direct_answer_points", []) or [])
    needs_direct_evidence = question_needs_direct_evidence(question)

    sufficiency = "limited"
    if direct_point_count >= 2 and len(direct_card_ids) >= 2 and avg_top_directness >= POLICY_DIRECTNESS_THRESHOLD:
        sufficiency = "adequate"
    elif direct_point_count >= 1 and avg_top_directness >= 0.10:
        sufficiency = "mixed"

    if needs_direct_evidence and gap_count and sufficiency == "adequate":
        sufficiency = "mixed"
    if needs_direct_evidence and avg_top_directness < 0.12:
        sufficiency = "limited"

    return {
        "sufficiency": sufficiency,
        "needs_direct_evidence": needs_direct_evidence,
        "avg_top_directness": round(avg_top_directness, 4),
        "direct_card_ids": direct_card_ids[:3],
        "gap_count": gap_count,
        "gap_messages": list(plan.get("evidence_gaps", []) or []),
    }


def is_policy_high_risk_candidate(
    question: str,
    candidate: Dict[str, Any],
    source_node: Dict[str, Any],
) -> bool:
    claim_text = strip_inline_citations(candidate.get("claim_text", "")).lower()
    directness = float(source_node.get("direct_answer_score", 0.0) or 0.0)
    alignment = float(candidate.get("claim_query_alignment", 0.0) or 0.0)
    support_score = float(candidate.get("support_strength_score", 0.0) or 0.0)
    risky_language = any(term in claim_text for term in POLICY_HIGH_RISK_TERMS) or bool(re.search(r"\b\d+(?:\.\d+)?%?\b", claim_text))

    if question_needs_direct_evidence(question) and directness < POLICY_DIRECTNESS_THRESHOLD and support_score < CLAIM_SUPPORT_STRENGTH["high"]:
        return True
    if risky_language and (alignment < 0.38 or directness < 0.12):
        return True
    return False


def build_policy_claim_units(
    question: str,
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    selected_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    node_by_card_id = {int(node.get("card_id", 0) or 0): node for node in selected_nodes}
    evidence_status = compute_policy_evidence_status(question, plan, selected_nodes)
    selected_units: List[Dict[str, Any]] = []
    seen_texts: List[str] = []

    def should_skip(text: str) -> bool:
        for existing in seen_texts:
            if claim_token_jaccard(text, existing) >= 0.65:
                return True
        return False

    def add_candidate(candidate: Dict[str, Any], slot: str) -> bool:
        text = candidate.get("claim_text", "")
        if not text or should_skip(text):
            return False
        unit = claim_candidate_to_unit(candidate, slot)
        selected_units.append(unit)
        seen_texts.append(text)
        return True

    def top_support_card_ids() -> List[int]:
        chosen = [unit["card_ids"][0] for unit in selected_units if unit.get("card_ids")]
        if chosen:
            return chosen[:2]
        return evidence_status.get("direct_card_ids", [])[:2]

    supported_candidates = [
        item for item in candidates if item.get("stance_role") in {"supporting", "mixed"} and item.get("claim_type") != "background"
    ]
    supported_candidates.sort(
        key=lambda item: (
            float(item.get("claim_score", 0.0) or 0.0)
            + (0.20 * float(node_by_card_id.get(int(item.get("source_card_id", 0) or 0), {}).get("direct_answer_score", 0.0) or 0.0))
        ),
        reverse=True,
    )
    supported_limit = 2 if evidence_status["sufficiency"] == "limited" else 3
    supported_count = 0
    for candidate in supported_candidates:
        source_node = node_by_card_id.get(int(candidate.get("source_card_id", 0) or 0))
        if source_node is None:
            continue
        if is_policy_high_risk_candidate(question, candidate, source_node):
            continue
        if add_candidate(candidate, "supported"):
            supported_count += 1
        if supported_count >= supported_limit:
            break

    if question_requests_conflict(question):
        disagreement_candidates = [item for item in candidates if item.get("stance_role") in {"counter", "mixed"}]
        disagreement_candidates.sort(
            key=lambda item: (
                float(item.get("claim_score", 0.0) or 0.0)
                + (0.20 * float(node_by_card_id.get(int(item.get("source_card_id", 0) or 0), {}).get("direct_answer_score", 0.0) or 0.0))
            ),
            reverse=True,
        )
        disagreement_added = False
        for candidate in disagreement_candidates:
            source_node = node_by_card_id.get(int(candidate.get("source_card_id", 0) or 0))
            if source_node is None:
                continue
            if is_policy_high_risk_candidate(question, candidate, source_node):
                continue
            if add_candidate(candidate, "disagreement"):
                disagreement_added = True
                break
        if not disagreement_added:
            limitation_text = (
                "The retrieved cards do not provide a clearly direct counterpoint, so the disagreement is only partially evidenced."
            )
            selected_units.append(build_policy_limitation_unit(limitation_text, top_support_card_ids()))

    if evidence_status["sufficiency"] != "adequate":
        if evidence_status["sufficiency"] == "limited":
            limitation_text = "The retrieved cards provide only limited direct evidence for the core question, so the answer should stay cautious and narrow."
        else:
            limitation_text = "The retrieved cards provide some direct evidence, but it remains incomplete and should not be treated as fully settling the question."
        selected_units.append(build_policy_limitation_unit(limitation_text, top_support_card_ids()))
    elif plan.get("evidence_gaps"):
        selected_units.append(
            build_policy_limitation_unit(
                "Some relevant parts of the question are still under-supported in the retrieved cards.",
                top_support_card_ids(),
            )
        )

    if not any(unit.get("slot") == "supported" for unit in selected_units) and selected_nodes:
        top_node = sorted(
            selected_nodes,
            key=lambda node: (
                float(node.get("direct_answer_score", 0.0) or 0.0),
                float(node.get("query_relevance", 0.0) or 0.0),
                float(node.get("S_trust", 0.0) or 0.0),
            ),
            reverse=True,
        )[0]
        card_id = int(top_node.get("card_id", 0) or 0)
        heading = top_node.get("heading") or top_node.get("title") or "This card"
        selected_units.insert(
            0,
            {
                "slot": "supported",
                "text": ensure_card_citations(
                    "{} provides part of the available evidence, but it does not fully settle the question.".format(heading),
                    [card_id],
                ),
                "card_ids": [card_id],
                "claim_text": heading,
                "claim_score": 0.0,
                "evidence_span": clean_snippet(top_node.get("text") or "", 45),
                "source_card_id": card_id,
                "source": top_node.get("source", "Unknown"),
                "claim_type": "fallback_supported",
                "verification": {"label": "unverified", "explanation": ""},
            },
        )

    return {
        "claim_units": selected_units[:POLICY_MAX_CLAIM_UNITS],
        "evidence_status": evidence_status,
    }


def build_claim_graph(
    question: str,
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    selected_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    claim_graph = {
        "supported_claims": [],
        "disagreement_claims": [],
        "background_claims": [],
        "selected_claim_units": [],
    }
    selected_units = []
    seen_texts: List[str] = []

    def should_skip(text: str) -> bool:
        for existing in seen_texts:
            if claim_token_jaccard(text, existing) >= 0.65:
                return True
        return False

    supported_candidates = [
        item for item in candidates if item.get("stance_role") in {"supporting", "mixed"} and item.get("claim_type") != "background"
    ]
    supported_candidates.sort(key=lambda item: float(item.get("claim_score", 0.0) or 0.0), reverse=True)
    for candidate in supported_candidates:
        if should_skip(candidate.get("claim_text", "")):
            continue
        unit = claim_candidate_to_unit(candidate, "supported")
        selected_units.append(unit)
        claim_graph["supported_claims"].append(candidate)
        seen_texts.append(candidate.get("claim_text", ""))
        if len([item for item in selected_units if item["slot"] == "supported"]) >= 3:
            break

    disagreement_candidates = [item for item in candidates if item.get("stance_role") in {"counter", "mixed"}]
    disagreement_candidates.sort(key=lambda item: float(item.get("claim_score", 0.0) or 0.0), reverse=True)
    for candidate in disagreement_candidates:
        if should_skip(candidate.get("claim_text", "")):
            continue
        unit = claim_candidate_to_unit(candidate, "disagreement")
        selected_units.append(unit)
        claim_graph["disagreement_claims"].append(candidate)
        seen_texts.append(candidate.get("claim_text", ""))
        if len([item for item in selected_units if item["slot"] == "disagreement"]) >= 2:
            break

    for candidate in candidates:
        if candidate in claim_graph["supported_claims"] or candidate in claim_graph["disagreement_claims"]:
            continue
        claim_graph["background_claims"].append(candidate)

    if not selected_units:
        selected_units = fallback_trustrag_claim_units(plan, selected_nodes)

    gaps = plan.get("evidence_gaps", []) or []
    if gaps and len(selected_units) < TRUSTRAG_MAX_CLAIM_UNITS:
        top_cards = [unit["card_ids"][0] for unit in selected_units if unit.get("card_ids")] or [int(node.get("card_id", 0) or 0) for node in selected_nodes[:2]]
        gap_sentence = str(gaps[0]).strip().rstrip(".")
        if not gap_sentence:
            gap_sentence = "Available evidence is partial on the core question"
        limitation = {
            "slot": "limitation",
            "text": ensure_card_citations("{}. ".format(gap_sentence).strip(), top_cards[:2]),
            "card_ids": top_cards[:2],
            "claim_text": gap_sentence,
            "claim_score": 0.0,
            "evidence_span": "",
            "source_card_id": top_cards[0] if top_cards else None,
            "source": "multiple",
            "claim_type": "evidence_gap",
            "verification": {"label": "unverified", "explanation": ""},
        }
        selected_units.append(limitation)

    claim_graph["selected_claim_units"] = selected_units[:TRUSTRAG_MAX_CLAIM_UNITS]
    claim_graph["question_focus_terms"] = extract_focus_terms(question)
    return claim_graph


def check_sentence_support(
    client: OpenAI,
    model: str,
    sentence: str,
    evidence_docs: List[Dict[str, Any]],
) -> Dict[str, str]:
    evidence_blocks = []
    for doc in evidence_docs:
        evidence_blocks.append(
            "Doc [{}] (Source: {} | S_trust: {})\n{}".format(
                int(doc.get("card_id", 0) or 0),
                doc.get("source", "N/A"),
                doc.get("S_trust", "N/A"),
                (doc.get("text") or "")[:900],
            )
        )

    user_prompt = "Answer sentence:\n{}\n\nEvidence:\n{}".format(sentence, "\n\n".join(evidence_blocks))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        label = str(parsed.get("label", "unclear")).strip().lower()
        if label not in {"supported", "unsupported", "unclear"}:
            label = "unclear"
        return {"label": label, "explanation": parsed.get("explanation", "")}
    except Exception as exc:
        return {"label": "unclear", "explanation": str(exc)}


def rewrite_claim_unit_with_evidence(
    client: OpenAI,
    model: str,
    question: str,
    claim_unit: Dict[str, Any],
    evidence_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence_blocks = []
    for doc in evidence_docs:
        evidence_blocks.append(
            "Doc [{}] (Source: {} | S_trust: {})\n{}".format(
                int(doc.get("card_id", 0) or 0),
                doc.get("source", "N/A"),
                doc.get("S_trust", "N/A"),
                (doc.get("text") or "")[:800],
            )
        )

    system_prompt = (
        "You are rewriting one unsupported or weakly supported claim unit.\n"
        "Use ONLY the provided evidence.\n"
        "Return a safer one-sentence claim with inline [Card X] citations, or return action=drop if the claim cannot be safely rescued.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "action": "rewrite" | "drop",\n'
        '  "text": "..."\n'
        "}"
    )
    user_prompt = (
        "Question:\n{question}\n\n"
        "Current claim unit:\n{claim}\n\n"
        "Evidence:\n{evidence}\n"
    ).format(
        question=question,
        claim=claim_unit.get("text", ""),
        evidence="\n\n".join(evidence_blocks),
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = extract_json_object(response.choices[0].message.content)
        action = str(parsed.get("action", "drop")).strip().lower()
        text = ensure_card_citations(str(parsed.get("text", "")).strip(), claim_unit.get("card_ids", []))
        if action != "rewrite" or not text:
            return {"action": "drop", "text": ""}
        return {"action": "rewrite", "text": text}
    except Exception:
        return {"action": "drop", "text": ""}


def conservative_limitation_unit(claim_unit: Dict[str, Any]) -> Dict[str, Any]:
    claim_text = strip_inline_citations(claim_unit.get("claim_text") or claim_unit.get("text", "")).rstrip(".")
    if claim_text:
        text = "The retrieved evidence does not clearly establish that {}.".format(claim_text[0].lower() + claim_text[1:] if len(claim_text) > 1 else claim_text.lower())
    else:
        text = "The retrieved evidence does not clearly establish this point."
    updated = dict(claim_unit)
    updated["slot"] = "limitation"
    updated["text"] = ensure_card_citations(text, claim_unit.get("card_ids", []))
    return updated


def verify_claim_units(
    client: OpenAI,
    model: str,
    question: str,
    claim_units: List[Dict[str, Any]],
    selected_nodes: List[Dict[str, Any]],
    strict_policy: bool = False,
) -> Dict[str, Any]:
    by_card_id = {int(node.get("card_id", 0) or 0): node for node in selected_nodes}
    verified_units = []
    stats = {"supported": 0, "rewritten": 0, "downgraded": 0, "dropped": 0}

    for unit in claim_units:
        evidence_docs = [by_card_id[card_id] for card_id in unit.get("card_ids", []) if card_id in by_card_id]
        if not evidence_docs:
            stats["dropped"] += 1
            continue
        verdict = check_sentence_support(client, model, unit.get("text", ""), evidence_docs)
        updated = dict(unit)
        updated["verification"] = verdict
        if verdict["label"] == "supported":
            stats["supported"] += 1
            verified_units.append(updated)
            continue

        rewrite = rewrite_claim_unit_with_evidence(client, model, question, updated, evidence_docs)
        if rewrite["action"] == "rewrite":
            rewritten = dict(updated)
            rewritten["text"] = rewrite["text"]
            rewritten_verdict = check_sentence_support(client, model, rewritten["text"], evidence_docs)
            rewritten["verification"] = rewritten_verdict
            if rewritten_verdict["label"] == "supported":
                stats["rewritten"] += 1
                verified_units.append(rewritten)
                continue

        limitation = conservative_limitation_unit(updated)
        limitation["verification"] = {"label": "supported", "explanation": "Downgraded to an explicit evidence-limit statement."}
        if strict_policy:
            limitation["text"] = ensure_card_citations(
                "The retrieved evidence does not directly establish this claim, so it should be treated as uncertain.",
                limitation.get("card_ids", []),
            )
        stats["downgraded"] += 1
        verified_units.append(limitation)

    if not verified_units:
        verified_units = claim_units[:1]
    return {"claim_units": verified_units[:TRUSTRAG_MAX_CLAIM_UNITS], "stats": stats}


def assemble_policy_answer(
    question: str,
    claim_units: List[Dict[str, Any]],
    evidence_status: Dict[str, Any],
) -> str:
    supported = [unit["text"] for unit in claim_units if unit.get("slot") == "supported"]
    contested = [unit["text"] for unit in claim_units if unit.get("slot") == "disagreement"]
    uncertain = [unit["text"] for unit in claim_units if unit.get("slot") == "limitation"]
    support_cards = []
    for unit in claim_units:
        for card_id in unit.get("card_ids", []):
            if card_id not in support_cards:
                support_cards.append(card_id)

    sufficiency = evidence_status.get("sufficiency", "limited")
    if sufficiency == "adequate":
        status_line = "The retrieved cards include multiple directly relevant pieces of evidence for this question."
    elif sufficiency == "mixed":
        status_line = "The retrieved cards include some direct evidence, but it remains incomplete and should be treated cautiously."
    else:
        status_line = "The retrieved cards provide only limited direct evidence for this question, so the answer should stay narrow."

    if sufficiency == "limited":
        bottom_line = "Bottom line: the available evidence is partial and indirect, so the question cannot be settled confidently from these cards alone."
    elif contested:
        bottom_line = "Bottom line: the retrieved evidence points in competing directions, so the answer is best framed as contested rather than settled."
    else:
        bottom_line = "Bottom line: the retrieved evidence leans in one direction, but remaining uncertainty should still be acknowledged."

    lines = ["- Evidence sufficiency"]
    lines.append("  - {}".format(ensure_card_citations(status_line, support_cards[:2])))
    lines.append("- Directly supported evidence")
    if supported:
        for item in supported[:2]:
            lines.append("  - Supported: {}".format(item))
    else:
        lines.append("  - Supported: {}".format(ensure_card_citations("No strongly supported direct claim could be stated safely from the retrieved cards.", support_cards[:2])))

    lines.append("- Counter-evidence or uncertainty")
    emitted_counter = False
    for item in contested[:1]:
        lines.append("  - Contested: {}".format(item))
        emitted_counter = True
    for item in uncertain[:2]:
        lines.append("  - Uncertain: {}".format(item))
        emitted_counter = True
    if not emitted_counter:
        lines.append("  - Uncertain: {}".format(ensure_card_citations("The retrieved evidence still leaves meaningful room for disagreement or missing context.", support_cards[:2])))

    lines.append("- Bottom line")
    lines.append("  - {}".format(ensure_card_citations(bottom_line, support_cards[:2])))
    return "\n".join(lines)


def generate_claim_graph_answer(
    client: OpenAI,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
    generator_model: str,
    planner_model: str,
    verifier_enabled: bool,
    policy_mode: str = "default",
) -> Dict[str, Any]:
    annotate_question_directness(question, retrieved_nodes, community_anchors)
    anchor_block = build_anchor_block(community_anchors)
    trust_cards = build_trust_evidence_cards(question, retrieved_nodes)
    plan = build_anchor_plan(client, question, anchor_block, trust_cards, planner_model)
    selected_card_ids = collect_trustrag_selected_card_ids(plan, retrieved_nodes)
    selected_nodes = [node for node in retrieved_nodes if int(node.get("card_id", 0) or 0) in selected_card_ids]
    claim_candidates = extract_claim_candidates(client, generator_model, question, selected_nodes, plan)
    claim_graph = build_claim_graph(question, plan, claim_candidates, selected_nodes)
    evidence_status = {}
    if policy_mode == "policy_v1":
        policy_payload = build_policy_claim_units(question, plan, claim_candidates, selected_nodes)
        claim_units = policy_payload["claim_units"]
        evidence_status = policy_payload["evidence_status"]
    else:
        claim_units = claim_graph.get("selected_claim_units", [])

    verification_stats = {}
    final_units = claim_units
    if verifier_enabled:
        verification = verify_claim_units(
            client,
            planner_model,
            question,
            claim_units,
            selected_nodes,
            strict_policy=(policy_mode == "policy_v1"),
        )
        final_units = verification["claim_units"]
        verification_stats = verification["stats"]

    if policy_mode == "policy_v1":
        answer = assemble_policy_answer(question, final_units, evidence_status)
    else:
        answer = assemble_trustrag_answer(client, generator_model, question, final_units)
    avg_directness = 0.0
    if selected_nodes:
        avg_directness = sum(float(node.get("direct_answer_score", 0.0) or 0.0) for node in selected_nodes) / len(selected_nodes)

    plan_payload = dict(plan)
    plan_payload["selected_card_ids"] = selected_card_ids
    plan_payload["claim_candidates"] = claim_candidates
    plan_payload["claim_graph"] = claim_graph
    plan_payload["claim_units"] = claim_units
    plan_payload["final_claim_units"] = final_units
    if evidence_status:
        plan_payload["evidence_status"] = evidence_status

    return {
        "answer": answer,
        "plan": plan_payload,
        "metadata": {
            "prompt_style": "claim_graph_policy" if policy_mode == "policy_v1" else ("claim_graph_verifier" if verifier_enabled else "claim_graph_only"),
            "selected_card_count": len(selected_nodes),
            "claim_candidate_count": len(claim_candidates),
            "claim_unit_count": len(claim_units),
            "final_claim_unit_count": len(final_units),
            "avg_selected_direct_answer_score": round(avg_directness, 4),
            "verification_stats": verification_stats,
            "policy_mode": policy_mode,
            "evidence_status": evidence_status,
        },
    }


def generate_reclaim_answer(
    client: OpenAI,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    generator_model: str,
) -> Dict[str, Any]:
    sentence_pool = build_reclaim_sentence_pool(retrieved_nodes)
    steps = []
    used_reference_ids = set()

    for _ in range(RECLAIM_MAX_STEPS):
        selection = select_reclaim_references(
            client=client,
            model=generator_model,
            question=question,
            sentence_pool=sentence_pool,
            previous_steps=steps,
        )
        if selection.get("action") != "continue":
            break

        reference_ids = [ref_id for ref_id in selection.get("reference_ids", []) if ref_id not in used_reference_ids]
        if not reference_ids:
            break

        selected_refs = [item for item in sentence_pool if item["sentence_id"] in reference_ids]
        if not selected_refs:
            break

        claim = generate_reclaim_claim(
            client=client,
            model=generator_model,
            question=question,
            selected_refs=selected_refs,
            previous_steps=steps,
        )
        if not claim:
            break

        used_reference_ids.update(reference_ids)
        steps.append(
            {
                "reference_ids": reference_ids,
                "references": selected_refs,
                "claim": claim,
                "rationale": selection.get("rationale", ""),
            }
        )

    if not steps:
        for bootstrap in bootstrap_reclaim_reference_plan(question, sentence_pool):
            selected_refs = bootstrap["references"]
            reference_ids = [item["sentence_id"] for item in selected_refs]
            claim = generate_reclaim_claim(
                client=client,
                model=generator_model,
                question=question,
                selected_refs=selected_refs,
                previous_steps=steps,
            )
            if not claim:
                continue
            used_reference_ids.update(reference_ids)
            steps.append(
                {
                    "reference_ids": reference_ids,
                    "references": selected_refs,
                    "claim": claim,
                    "rationale": "bootstrap_fallback",
                }
            )
            if len(steps) >= RECLAIM_MAX_STEPS:
                break

    if steps:
        answer = "\n".join(step["claim"].strip() for step in steps if step.get("claim"))
    else:
        answer = ""

    return {
        "answer": answer,
        "plan": {"reclaim_steps": steps},
        "metadata": {
            "prompt_style": "interleaving_reference_claim_generation",
            "reclaim_sentence_pool_size": len(sentence_pool),
            "reclaim_step_count": len(steps),
        },
    }


def build_generation_payload(
    method: str,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
    client: OpenAI,
    planner_model: str,
    method_context: Dict[str, Any],
) -> Dict[str, Any]:
    if method in {"vanilla_rag", "mmr_rag", "graph_retrieval"}:
        return {
            "context": build_basic_evidence_cards(question, retrieved_nodes),
            "system_prompt": (
                "You are an expert policy analyst specializing in evidence-grounded synthesis.\n\n"
                "Rules:\n"
                "1. Use ONLY the provided evidence cards.\n"
                "2. Do not use outside knowledge.\n"
                "3. Distinguish well-supported points from contested ones.\n"
                "4. Represent important competing views without inventing false balance.\n"
                "5. Cite factual claims explicitly with [Card X].\n"
                "6. If the evidence is thin, conflicting, or incomplete, say so clearly.\n\n"
                "Use whatever sentence- or paragraph-level format best presents the analysis."
            ),
            "plan": {},
            "metadata": {"prompt_style": "basic_grounded_synthesis"},
        }

    if method == "r2ag":
        signal_block = build_r2ag_signal_block(retrieved_nodes)
        return {
            "context": "{}\n\n{}\n\n{}".format(
                signal_block,
                build_retrieval_metadata_cards(question, retrieved_nodes),
                "Treat each <R_i> block as the retrieval-information prefix for Card [i]. Use these retrieval scores to decide which documents deserve more attention, while still grounding every claim in the document text itself.",
            ),
            "system_prompt": (
                "You are an analyst that uses retrieval-side scoring features alongside the document text to answer questions.\n\n"
                "Rules:\n"
                "1. Use ONLY the provided evidence cards and retrieval signals.\n"
                "2. For each document, interpret its <R_i> block as retrieval-side information prepended to that document.\n"
                "3. Prefer documents with stronger retrieval signals when deciding the answer backbone, unless their actual text is clearly weaker or off-topic.\n"
                "4. Use lower-signal documents mainly to explain disagreement, edge cases, or uncertainty.\n"
                "5. Cite factual claims explicitly with [Card X].\n"
                "6. Do not force equal weight across perspectives when retrieval evidence quality is asymmetric.\n\n"
                "Use whatever sentence- or paragraph-level format best presents the analysis."
            ),
            "plan": {},
            "metadata": {
                "prompt_style": "retrieval_info_prefixed_prompting",
                "signal_summary": signal_block,
            },
        }

    if method == "grag":
        text_view = method_context.get("text_view") or build_graph_cluster_block(retrieved_nodes)
        graph_view = method_context.get("graph_view") or "Relational Context (node-edge topology summary):\n[graph view unavailable]"
        return {
            "context": "{}\n\n{}\n\n{}".format(
                text_view,
                graph_view,
                build_basic_evidence_cards(question, retrieved_nodes),
            ),
            "system_prompt": (
                "You are an expert analyst that uses both textual and relational context to answer questions.\n\n"
                "Rules:\n"
                "1. Use ONLY the provided context and evidence cards.\n"
                "2. Use the hierarchical context to identify which evidence clusters are most relevant to the question.\n"
                "3. Use the relational context to reason about how documents reinforce or contradict each other.\n"
                "4. Organize the answer around the most strongly connected and relevant evidence, prioritizing cards that appear central to multiple related documents.\n"
                "5. Explain where documents reinforce each other and where they expose disagreement or uncertainty.\n"
                "6. Cite factual claims explicitly with [Card X], using exactly that bracket format (do not write 'Card [X]' or '[X]' alone).\n"
                "7. Each cited sentence must state a claim that is literally present in the cited card(s); do not write 'this card implies', 'this suggests', or other interpretive paraphrase as a cited factual claim. If you must offer interpretation, mark it with phrases like 'taken together' and do NOT cite a card for it.\n"
                "8. Do not invent relations that are not supported by the provided context and cards."
            ),
            "plan": {},
            "metadata": {
                "prompt_style": "two_view_graph_context_generation",
                "graph_summary": text_view,
                "graph_view": graph_view,
            },
        }

    if method in {"trustrag_anchor", "trustrag_anchor_top2"}:
        anchor_block = build_anchor_block(community_anchors)
        trust_cards = build_trust_evidence_cards(question, retrieved_nodes)
        plan = build_anchor_plan(client, question, anchor_block, trust_cards, planner_model)
        return {
            "context": "{}\n\n{}\n\n{}".format(anchor_block, format_anchor_plan(plan), trust_cards),
            "system_prompt": (
                "You are an expert policy analyst specializing in trust-aware, evidence-grounded synthesis.\n\n"
                "Critical rules:\n"
                "1. Use ONLY the provided community anchors and trust-aware evidence cards.\n"
                "2. Start from community anchors as the answer backbone unless the card evidence clearly contradicts them.\n"
                "3. Prioritize claims supported by higher S_trust and stronger query relevance.\n"
                "4. Use anchor-backed evidence to establish the most supported points before discussing disagreements.\n"
                "5. Evidence-weighted balance is required: represent important competing views, but do NOT give equal weight to every side when the evidence quality is asymmetric.\n"
                "6. Use lower-trust or conflicting evidence to explain disagreement or uncertainty, not as the main basis of the final conclusion unless stronger evidence is absent.\n"
                "7. Cite factual claims explicitly using [Card X].\n"
                "8. If the evidence is insufficient or conflicting, say so clearly."
            ),
            "plan": plan,
            "metadata": {"prompt_style": "anchor_aware_generation"},
        }

    raise ValueError("Unsupported method: {}".format(method))


def generate_answer(
    client: OpenAI,
    method: str,
    question: str,
    retrieved_nodes: List[Dict[str, Any]],
    community_anchors: List[Dict[str, Any]],
    generator_model: str,
    planner_model: str,
    method_context: Dict[str, Any],
) -> Dict[str, Any]:
    if method == "reclaim":
        return generate_reclaim_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            generator_model=generator_model,
        )

    if method in {"trustrag_anchor", "trustrag_anchor_top2"}:
        return generate_trustrag_anchor_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            community_anchors=community_anchors,
            generator_model=generator_model,
            planner_model=planner_model,
        )

    if method == "claim_graph_only":
        return generate_claim_graph_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            community_anchors=community_anchors,
            generator_model=generator_model,
            planner_model=planner_model,
            verifier_enabled=False,
        )

    if method == "claim_graph_verifier":
        return generate_claim_graph_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            community_anchors=community_anchors,
            generator_model=generator_model,
            planner_model=planner_model,
            verifier_enabled=True,
        )

    if method == "claim_graph_policy":
        return generate_claim_graph_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            community_anchors=community_anchors,
            generator_model=generator_model,
            planner_model=planner_model,
            verifier_enabled=True,
            policy_mode="policy_v1",
        )

    if method == "trustrag_hybrid_v5":
        return generate_trustrag_hybrid_v5_answer(
            client=client,
            question=question,
            retrieved_nodes=retrieved_nodes,
            community_anchors=community_anchors,
            generator_model=generator_model,
            planner_model=planner_model,
        )

    payload = build_generation_payload(
        method,
        question,
        retrieved_nodes,
        community_anchors,
        client,
        planner_model,
        method_context,
    )
    try:
        response = client.chat.completions.create(
            model=generator_model,
            messages=[
                {"role": "system", "content": payload["system_prompt"]},
                {"role": "user", "content": "Context:\n{}\n\nQuestion: {}".format(payload["context"], question)},
            ],
            temperature=0.0,
        )
        answer_text = response.choices[0].message.content
    except Exception as exc:
        answer_text = "Error: {}".format(exc)

    return {
        "answer": answer_text,
        "plan": payload.get("plan", {}),
        "metadata": payload.get("metadata", {}),
    }


def serialize_retrieval_nodes(retrieved_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for node in retrieved_nodes:
        serialized.append(
            {
                "id": node.get("id"),
                "title": node.get("title"),
                "heading": node.get("heading"),
                "source": node.get("source"),
                "stance_label": node.get("stance_label"),
                "S_trust": node.get("S_trust"),
                "P_polar": node.get("P_polar"),
                "community": node.get("community"),
                "community_purity": node.get("community_purity"),
                "text": clean_snippet(node.get("text") or "", RETRIEVAL_TEXT_LIMIT),
                "query_relevance": node.get("query_relevance"),
                "retrieval_rank": node.get("retrieval_rank"),
                "card_id": node.get("card_id"),
                "is_community_anchor": node.get("is_community_anchor"),
                "anchor_score": node.get("anchor_score"),
                "anchor_internal_degree_norm": node.get("anchor_internal_degree_norm"),
                "anchor_query_norm": node.get("anchor_query_norm"),
                "anchor_high_trust_support_ratio": node.get("anchor_high_trust_support_ratio"),
                "r2ag_rel_norm": node.get("r2ag_rel_norm"),
                "r2ag_rank_percentile": node.get("r2ag_rank_percentile"),
                "r2ag_precedent_similarity": node.get("r2ag_precedent_similarity"),
                "r2ag_neighbor_similarity": node.get("r2ag_neighbor_similarity"),
                "r2ag_listwise_coherence": node.get("r2ag_listwise_coherence"),
                "grag_membership_count": node.get("grag_membership_count"),
                "grag_min_depth": node.get("grag_min_depth"),
                "grag_soft_prune_score": node.get("grag_soft_prune_score"),
                "grag_is_key_node": node.get("grag_is_key_node"),
            }
        )
    return serialized


def prepare_topic_payload(
    client: OpenAI,
    topic_key: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    topic_cfg = TOPIC_CONFIGS[topic_key]
    graph_payload = load_graph_payload(topic_cfg["graph_path"])
    nodes = graph_payload["nodes"]
    links = graph_payload["links"]
    cache_dir = Path(args.cache_dir)
    docs_text = [(node.get("text") or "")[:1200] for node in nodes]
    queries = topic_cfg["queries"][: args.limit_queries] if args.limit_queries else topic_cfg["queries"]
    query_texts = [item["q"] for item in queries]

    node_embs = embed_texts(
        client=client,
        items=docs_text,
        model=args.embedding_model,
        cache_dir=cache_dir,
        prefix="docs_{}".format(topic_key),
        refresh_cache=args.refresh_cache,
    )
    query_embs = embed_texts(
        client=client,
        items=query_texts,
        model=args.embedding_model,
        cache_dir=cache_dir,
        prefix="queries_{}".format(topic_key),
        refresh_cache=args.refresh_cache,
    )
    edge_items = []
    edge_key_to_index = {}
    for link in links:
        key = edge_signature(link)
        if key in edge_key_to_index:
            continue
        edge_key_to_index[key] = len(edge_items)
        edge_items.append(edge_text(link))
    edge_embs = embed_texts(
        client=client,
        items=edge_items or ["no edges"],
        model=args.embedding_model,
        cache_dir=cache_dir,
        prefix="edges_{}".format(topic_key),
        refresh_cache=args.refresh_cache,
    )
    anchor_feature_map = build_anchor_feature_map(nodes, links)
    node_id_to_index = {int(node["id"]): idx for idx, node in enumerate(nodes)}
    links_by_node_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for link in links:
        src = int(link.get("source"))
        tgt = int(link.get("target"))
        links_by_node_id[src].append(link)
        links_by_node_id[tgt].append(link)

    return {
        "topic_key": topic_key,
        "topic_display": topic_cfg["display_name"],
        "nodes": nodes,
        "links": links,
        "node_id_to_index": node_id_to_index,
        "links_by_node_id": links_by_node_id,
        "edge_key_to_index": edge_key_to_index,
        "edge_embs": edge_embs,
        "queries": queries,
        "node_embs": node_embs,
        "query_embs": query_embs,
        "anchor_feature_map": anchor_feature_map,
    }


def retrieve_for_method(
    method: str,
    query_emb: np.ndarray,
    topic_payload: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    nodes = topic_payload["nodes"]
    node_embs = topic_payload["node_embs"]

    if method in {"vanilla_rag", "reclaim"}:
        return {
            "retrieved_nodes": semantic_topk_retrieval(query_emb, nodes, node_embs, args.top_k),
            "method_context": {},
        }
    if method == "r2ag":
        retrieved_nodes = semantic_topk_retrieval(query_emb, nodes, node_embs, args.top_k)
        annotate_r2ag_features(retrieved_nodes, node_embs)
        return {"retrieved_nodes": retrieved_nodes, "method_context": {}}
    if method == "mmr_rag":
        return {
            "retrieved_nodes": mmr_retrieval(query_emb, nodes, node_embs, args.top_k, args.mmr_lambda),
            "method_context": {},
        }
    if method == "graph_retrieval":
        return {
            "retrieved_nodes": graph_retrieval(
                query_emb,
                nodes,
                node_embs,
                top_k=args.top_k,
                candidate_pool=args.candidate_pool,
                alpha=args.graph_alpha,
                gamma=args.graph_gamma,
                delta=args.graph_delta,
            ),
            "method_context": {},
        }
    if method == "grag":
        return grag_subgraph_retrieval(
            query_emb=query_emb,
            topic_payload=topic_payload,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
        )
    if method in {
        "trustrag_anchor",
        "trustrag_anchor_top2",
        "claim_graph_only",
        "claim_graph_verifier",
        "claim_graph_policy",
        "trustrag_hybrid_v5",
    }:
        return {
            "retrieved_nodes": trustrag_retrieval(
                query_emb,
                nodes,
                node_embs,
                top_k=args.top_k,
                candidate_pool=args.candidate_pool,
                alpha=args.trust_alpha,
                beta=args.trust_beta,
                use_diversity=True,
            ),
            "method_context": {},
        }
    raise ValueError("Unsupported method: {}".format(method))


def anchors_per_community_for_method(method: str) -> int:
    if method == "trustrag_anchor_top2":
        return 2
    return 1


def apply_sensitivity_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    global ANCHOR_HIGH_TRUST_THRESHOLD, TRUST_DIV_RHO, TRUST_DIV_MODE, CARD_GAMMA
    global ANCHOR_LAMBDA_STRUCTURE, ANCHOR_LAMBDA_RELEVANCE, ANCHOR_LAMBDA_SUPPORT
    ANCHOR_HIGH_TRUST_THRESHOLD = float(args.anchor_high_trust_threshold)
    TRUST_DIV_RHO = float(args.trust_div_rho)
    TRUST_DIV_MODE = str(args.trust_div_mode)
    CARD_GAMMA = float(args.card_gamma)
    ANCHOR_LAMBDA_STRUCTURE = float(args.anchor_lambda_structure)
    ANCHOR_LAMBDA_RELEVANCE = float(args.anchor_lambda_relevance)
    ANCHOR_LAMBDA_SUPPORT = float(args.anchor_lambda_support)
    return {
        "trust_alpha": float(args.trust_alpha),
        "trust_beta": float(args.trust_beta),
        "trust_div_rho": TRUST_DIV_RHO,
        "trust_div_mode": TRUST_DIV_MODE,
        "card_gamma": CARD_GAMMA,
        "anchor_lambda_structure": ANCHOR_LAMBDA_STRUCTURE,
        "anchor_lambda_relevance": ANCHOR_LAMBDA_RELEVANCE,
        "anchor_lambda_support": ANCHOR_LAMBDA_SUPPORT,
        "anchor_high_trust_threshold": ANCHOR_HIGH_TRUST_THRESHOLD,
    }


def main() -> None:
    args = parse_args()
    sensitivity_config = apply_sensitivity_overrides(args)
    output_path = Path(args.output_jsonl)
    ensure_parent(output_path)

    # Resume: load already-completed (topic, query_id) pairs from existing output.
    completed: set = set()
    file_mode = "w"
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as _f:
            for _line in _f:
                try:
                    _row = json.loads(_line)
                    completed.add((_row.get("topic"), _row.get("query_id")))
                except json.JSONDecodeError:
                    pass
        if completed:
            print("[resume] {} rows already done — appending.".format(len(completed)))
            file_mode = "a"

    client = create_openai_client()

    topic_payloads = {}
    for topic_key in args.topics:
        print("Preparing topic:", topic_key)
        topic_payloads[topic_key] = prepare_topic_payload(client, topic_key, args)

    with output_path.open(file_mode, encoding="utf-8") as out_f:
        for topic_key in args.topics:
            payload = topic_payloads[topic_key]
            print("Running topic:", topic_key)
            for query_item, query_emb in tqdm(
                list(zip(payload["queries"], payload["query_embs"])),
                desc="Exp4 {}".format(topic_key),
            ):
                if (topic_key, query_item["id"]) in completed:
                    continue

                row = {
                    "query_id": query_item["id"],
                    "topic": topic_key,
                    "topic_display": payload["topic_display"],
                    "question": query_item["q"],
                    "top_k": args.top_k,
                    "candidate_pool": args.candidate_pool,
                    "answers": {},
                    "answer_plans": {},
                    "generation_metadata": {},
                    "community_anchors": {},
                    "retrievals": {},
                    "method_specs": {method: METHOD_SPECS[method] for method in args.methods},
                    "sensitivity_config": sensitivity_config,
                }

                for method in args.methods:
                    retrieval_result = retrieve_for_method(method, query_emb, payload, args)
                    retrieved_nodes = retrieval_result["retrieved_nodes"]
                    method_context = retrieval_result.get("method_context", {})
                    community_anchors = finalize_retrieval(
                        retrieved_nodes,
                        payload["anchor_feature_map"],
                        anchors_per_community=anchors_per_community_for_method(method),
                    )
                    if method == "grag":
                        method_context = materialize_grag_context(method_context, retrieved_nodes)
                    generated = generate_answer(
                        client=client,
                        method=method,
                        question=query_item["q"],
                        retrieved_nodes=retrieved_nodes,
                        community_anchors=community_anchors,
                        generator_model=args.generator_model,
                        planner_model=args.planner_model,
                        method_context=method_context,
                    )

                    row["answers"][method] = generated["answer"]
                    row["answer_plans"][method] = generated["plan"]
                    generation_metadata = dict(generated["metadata"])
                    if method_context:
                        generation_metadata["method_context"] = method_context
                    row["generation_metadata"][method] = generation_metadata
                    row["community_anchors"][method] = community_anchors
                    row["retrievals"][method] = serialize_retrieval_nodes(retrieved_nodes)

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    print("Saved Exp4 outputs to {}".format(output_path))


if __name__ == "__main__":
    main()
