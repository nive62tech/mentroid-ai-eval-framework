#!/usr/bin/env python3
"""
Run an LLM / AI system evaluation/benchmark.

Example (using the bundled synthetic sample + a mock model):
    python scripts/run_llm_eval.py \\
        --test-cases data/llm_sample/test_cases.json \\
        --model-name mock-llm-v1 \\
        --dataset-name llm_sample

To benchmark a REAL model (e.g. via the Anthropic API), replace
`load_mock_model()` below with a real `generate_fn`, e.g.:

    import anthropic
    client = anthropic.Anthropic()
    def generate_fn(prompt):
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return {
            "text": text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.llm.runner import run_llm_benchmark
from eval_framework.core.results import ResultStore


def load_mock_model(noise_seed: int = 42):
    """A tiny synthetic 'model' standing in for a real LLM call.

    Echoes back a context-grounded answer most of the time, and
    occasionally hallucinates an ungrounded answer, so accuracy /
    groundedness / hallucination-rate come out realistically imperfect.
    Replace with a real API call for actual benchmarking.
    """
    rng = random.Random(noise_seed)

    def generate_fn(query: str) -> dict:
        time.sleep(rng.uniform(0.05, 0.15))  # simulate response latency
        if rng.random() < 0.15:
            text = "I'm not certain, but it might be related to something else entirely."
        else:
            # naive stand-in "answer": in a real model this would be an actual generation
            text = f"Based on the provided context, the answer to '{query}' is as described."
        return {
            "text": text,
            "input_tokens": len(query.split()) + 20,
            "output_tokens": len(text.split()),
        }

    return generate_fn


def main():
    parser = argparse.ArgumentParser(description="Run LLM / AI system evaluation/benchmark.")
    parser.add_argument("--test-cases", required=True, help="Path to test_cases.json")
    parser.add_argument("--model-name", default="mock-llm-v1", help="Model/version identifier")
    parser.add_argument("--dataset-name", default="llm_sample", help="Dataset identifier")
    parser.add_argument("--hallucination-threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42, help="Mock-model noise seed (reproducibility)")
    args = parser.parse_args()

    with open(args.test_cases) as f:
        test_cases = json.load(f)

    generate_fn = load_mock_model(noise_seed=args.seed)

    result = run_llm_benchmark(
        generate_fn=generate_fn,
        test_cases=test_cases,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        hallucination_threshold=args.hallucination_threshold,
        config={
            "hallucination_threshold": args.hallucination_threshold,
            "seed": args.seed,
            "test_cases_file": args.test_cases,
        },
    )

    store = ResultStore(results_dir=args.results_dir)
    path = store.save(result)

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nSaved result to: {path}")


if __name__ == "__main__":
    main()
