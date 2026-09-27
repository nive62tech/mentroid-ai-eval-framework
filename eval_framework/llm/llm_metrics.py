"""
LLM / AI system evaluation metrics:
    - Accuracy (exact/normalized match against expected answer)
    - Relevance (lexical overlap / semantic-hook proxy against the query)
    - Groundedness (does the answer's content appear supported by provided context)
    - Hallucination rate (inverse of groundedness, at the response level)
    - Token usage / cost

These default implementations are dependency-free, deterministic,
lexical-overlap based proxies so the framework runs out of the box
without an API key or embedding model. Each function is written so it
can be swapped for an LLM-as-judge or embedding-similarity
implementation (see `judge.py`) without changing the runner's
interface — same input/output shape.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    """Fraction of predictions that normalized-match their reference exactly."""
    if not predictions:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references) if _normalize(p) == _normalize(r)
    )
    return round(matches / len(predictions), 4)


def token_overlap_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between a prediction and a reference answer.
    Useful as an accuracy proxy when exact match is too strict (e.g. free-form answers).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in common.items():
        overlap += min(c, ref_counts.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def relevance_score(response: str, query: str) -> float:
    """Lexical-overlap proxy for how relevant a response is to the query.
    Score in [0, 1]: fraction of query's meaningful tokens reflected in the response.
    """
    query_tokens = set(_tokenize(query))
    response_tokens = set(_tokenize(response))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & response_tokens
    return round(len(overlap) / len(query_tokens), 4)


def groundedness_score(response: str, context: str) -> float:
    """Proxy for whether a response's claims are supported by the provided context.
    Measures the fraction of the response's content tokens that also appear in
    the context. Low score suggests unsupported / hallucinated content.
    """
    response_tokens = set(_tokenize(response))
    context_tokens = set(_tokenize(context))
    if not response_tokens:
        return 1.0  # empty response has nothing ungrounded
    supported = response_tokens & context_tokens
    return round(len(supported) / len(response_tokens), 4)


def hallucination_rate(responses: List[str], contexts: List[str], threshold: float = 0.5) -> float:
    """Fraction of responses whose groundedness score falls below `threshold`,
    i.e. flagged as likely containing unsupported / hallucinated content.
    """
    if not responses:
        return 0.0
    flagged = sum(
        1 for r, c in zip(responses, contexts) if groundedness_score(r, c) < threshold
    )
    return round(flagged / len(responses), 4)


# ---- Token usage / cost -----------------------------------------------------

# Illustrative per-1K-token pricing; override via config for the actual model used.
DEFAULT_PRICING_PER_1K = {
    "input": 0.003,
    "output": 0.015,
}


def estimate_tokens(text: str) -> int:
    """Very rough token estimate (whitespace-based) used only when an exact
    token count from the API/tokenizer isn't available. Prefer passing real
    token counts from the API response when you have them.
    """
    return max(1, len(text.split()))


def token_usage_and_cost(
    input_tokens: int,
    output_tokens: int,
    pricing_per_1k: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    pricing = pricing_per_1k or DEFAULT_PRICING_PER_1K
    cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }
