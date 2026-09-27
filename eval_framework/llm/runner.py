"""
LLM / AI system evaluation runner.

Wraps a user-supplied `generate_fn(prompt) -> dict` around the LLM
metrics and timing utilities. `generate_fn` should return:
    {
        "text": "<model response>",
        "input_tokens": <int, optional>,
        "output_tokens": <int, optional>,
    }
If token counts aren't supplied, they are estimated from word counts
(clearly marked as estimated in the result).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from eval_framework.core.timing import LatencyTracker, Timer
from eval_framework.core.results import EvalResult
from eval_framework.llm.llm_metrics import (
    exact_match_accuracy,
    token_overlap_f1,
    relevance_score,
    groundedness_score,
    hallucination_rate,
    estimate_tokens,
    token_usage_and_cost,
)


def run_llm_benchmark(
    generate_fn: Callable[[str], dict],
    test_cases: List[dict],
    model_name: str,
    dataset_name: str,
    hallucination_threshold: float = 0.5,
    pricing_per_1k: Optional[dict] = None,
    config: Optional[dict] = None,
) -> EvalResult:
    """Run a full LLM benchmark: accuracy/relevance/groundedness + latency + cost.

    Args:
        generate_fn: function taking a prompt string, returning
            {"text": str, "input_tokens": int?, "output_tokens": int?}
        test_cases: list of dicts, each with:
            {"query": str, "context": str (optional), "reference": str (optional)}
        model_name: identifier for the model/version under test
        dataset_name: identifier for the test set used
        hallucination_threshold: groundedness cutoff below which a response is flagged
        pricing_per_1k: optional {"input": $/1K tokens, "output": $/1K tokens}
        config: arbitrary config dict stored alongside results for reproducibility

    Returns:
        EvalResult with accuracy/relevance/groundedness/hallucination + latency + cost.
    """
    if not test_cases:
        raise ValueError("No test cases provided for LLM benchmark.")

    tracker = LatencyTracker()
    predictions, references, responses, contexts = [], [], [], []
    f1_scores, relevance_scores, groundedness_scores = [], [], []
    total_input_tokens = total_output_tokens = 0
    tokens_estimated = False

    for case in test_cases:
        query = case["query"]
        context = case.get("context", "")
        reference = case.get("reference", "")

        with Timer() as t:
            result = generate_fn(query)
        tracker.record_from(t)

        text = result.get("text", "")
        in_tok = result.get("input_tokens")
        out_tok = result.get("output_tokens")
        if in_tok is None:
            in_tok = estimate_tokens(query + context)
            tokens_estimated = True
        if out_tok is None:
            out_tok = estimate_tokens(text)
            tokens_estimated = True

        total_input_tokens += in_tok
        total_output_tokens += out_tok

        predictions.append(text)
        references.append(reference)
        responses.append(text)
        contexts.append(context)

        if reference:
            f1_scores.append(token_overlap_f1(text, reference))
        relevance_scores.append(relevance_score(text, query))
        if context:
            groundedness_scores.append(groundedness_score(text, context))

    latency_stats = tracker.summary()
    usage = token_usage_and_cost(total_input_tokens, total_output_tokens, pricing_per_1k)

    metrics = {
        "num_cases": len(test_cases),
        "latency_ms_mean": latency_stats["mean_ms"],
        "latency_ms_p95": latency_stats["p95_ms"],
        "latency_ms_p99": latency_stats["p99_ms"],
        "token_usage": usage,
        "tokens_estimated": tokens_estimated,
    }

    not_applicable = {}

    if any(references):
        metrics["accuracy_exact_match"] = exact_match_accuracy(
            [p for p, r in zip(predictions, references) if r],
            [r for r in references if r],
        )
        metrics["accuracy_token_f1"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    else:
        not_applicable["accuracy"] = "No reference answers provided in test cases."

    metrics["relevance"] = round(sum(relevance_scores) / len(relevance_scores), 4)

    if any(contexts):
        metrics["groundedness"] = (
            round(sum(groundedness_scores) / len(groundedness_scores), 4) if groundedness_scores else 0.0
        )
        metrics["hallucination_rate"] = hallucination_rate(
            [r for r, c in zip(responses, contexts) if c],
            [c for c in contexts if c],
            threshold=hallucination_threshold,
        )
    else:
        not_applicable["groundedness"] = "No context provided in test cases."
        not_applicable["hallucination_rate"] = "No context provided in test cases."

    return EvalResult(
        task_type="llm",
        model_name=model_name,
        dataset_name=dataset_name,
        metrics=metrics,
        not_applicable=not_applicable,
        config=config or {},
    )
