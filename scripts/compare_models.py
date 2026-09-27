#!/usr/bin/env python3
"""
Compare the latest saved evaluation results of two or more model
versions/configurations, side by side.

Example:
    python scripts/compare_models.py --models mock-detector-v1 mock-detector-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.core.results import ResultStore


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across model versions.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to compare")
    parser.add_argument("--metrics", nargs="*", default=None, help="Specific metric keys to compare (default: all shared)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    store = ResultStore(results_dir=args.results_dir)
    comparison = store.compare(args.models, metric_keys=args.metrics)

    if comparison["missing_models"]:
        print(f"Warning: no saved results found for: {comparison['missing_models']}")

    print("\n=== Model Comparison ===\n")
    header = f"{'Metric':<25}" + "".join(f"{m:<25}" for m in args.models)
    print(header)
    print("-" * len(header))
    for metric, values in comparison["metrics"].items():
        row = f"{metric:<25}" + "".join(f"{str(values.get(m, 'N/A')):<25}" for m in args.models)
        print(row)

    print("\nFull JSON:\n")
    print(json.dumps(comparison["metrics"], indent=2))


if __name__ == "__main__":
    main()
