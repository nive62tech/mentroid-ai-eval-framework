import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.llm.llm_metrics import (
    exact_match_accuracy,
    token_overlap_f1,
    relevance_score,
    groundedness_score,
    hallucination_rate,
    token_usage_and_cost,
)


def test_exact_match_accuracy_all_correct():
    preds = ["Paris", "London"]
    refs = ["paris", "London"]
    assert exact_match_accuracy(preds, refs) == 1.0


def test_exact_match_accuracy_partial():
    preds = ["Paris", "Berlin"]
    refs = ["Paris", "London"]
    assert exact_match_accuracy(preds, refs) == 0.5


def test_token_overlap_f1_identical():
    assert token_overlap_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_token_overlap_f1_no_overlap():
    assert token_overlap_f1("apple banana", "car truck") == 0.0


def test_relevance_score_high_overlap():
    score = relevance_score("The capital of France is Paris", "What is the capital of France")
    assert score > 0.5


def test_relevance_score_no_overlap():
    score = relevance_score("Bananas are yellow", "What is quantum physics")
    assert score == 0.0


def test_groundedness_fully_supported():
    context = "Paris is the capital of France and its largest city"
    response = "Paris is the capital of France"
    assert groundedness_score(response, context) == 1.0


def test_groundedness_unsupported():
    context = "Paris stands as the capital city of France"
    response = "Bananas grow abundantly in tropical jungles"
    score = groundedness_score(response, context)
    assert score < 0.5


def test_hallucination_rate_flags_low_groundedness():
    responses = ["Bananas grow abundantly in tropical jungles", "Paris stands as the capital city of France"]
    contexts = ["Paris stands as the capital city of France", "Paris stands as the capital city of France"]
    rate = hallucination_rate(responses, contexts, threshold=0.5)
    assert rate == 0.5


def test_token_usage_and_cost():
    usage = token_usage_and_cost(1000, 500, {"input": 0.003, "output": 0.015})
    assert usage["total_tokens"] == 1500
    assert usage["estimated_cost_usd"] > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
