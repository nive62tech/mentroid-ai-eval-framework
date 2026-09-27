#!/usr/bin/env python3
"""
Run a Computer Vision evaluation/benchmark.

Example (using the bundled synthetic sample + a mock model):
    python scripts/run_cv_eval.py \\
        --ground-truth data/cv_sample/ground_truth.json \\
        --model-name mock-detector-v1 \\
        --dataset-name cv_sample

To benchmark a REAL model, replace `load_mock_model()` below with your
own model-loading + `predict_fn`, or import `run_cv_benchmark` directly
in your own script (see docs/METHODOLOGY.md for the interface).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.cv.runner import run_cv_benchmark
from eval_framework.core.results import ResultStore


def load_mock_model(noise_seed: int = 42):
    """A tiny synthetic 'model' standing in for a real detector.

    It perturbs ground-truth boxes slightly and occasionally drops or adds
    a box, so precision/recall/mAP come out realistically imperfect. This
    is ONLY for demonstrating the pipeline; replace with your real model's
    inference call for actual benchmarking.
    """
    rng = random.Random(noise_seed)

    def predict_fn(gt_boxes_for_image):
        # simulate inference cost
        time.sleep(rng.uniform(0.01, 0.04))
        preds = []
        for box in gt_boxes_for_image:
            if rng.random() < 0.12:  # simulate a missed detection
                continue
            x1, y1, x2, y2 = box["bbox"]
            jitter = lambda v: v + rng.uniform(-4, 4)
            preds.append({
                "class": box["class"],
                "bbox": [jitter(x1), jitter(y1), jitter(x2), jitter(y2)],
                "score": round(rng.uniform(0.55, 0.99), 3),
            })
        if rng.random() < 0.15:  # simulate a false positive
            preds.append({
                "class": rng.choice(["car", "person"]),
                "bbox": [rng.uniform(0, 300), rng.uniform(0, 200), rng.uniform(300, 400), rng.uniform(200, 260)],
                "score": round(rng.uniform(0.3, 0.6), 3),
            })
        return preds

    return predict_fn


def main():
    parser = argparse.ArgumentParser(description="Run CV model evaluation/benchmark.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("--model-name", default="mock-detector-v1", help="Model/version identifier")
    parser.add_argument("--dataset-name", default="cv_sample", help="Dataset identifier")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42, help="Mock-model noise seed (reproducibility)")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # "images" here is just the ground-truth boxes themselves, since the mock
    # model uses them to synthesize predictions. A real integration would
    # load actual image files/tensors here instead.
    images = {image_id: boxes for image_id, boxes in ground_truth.items()}

    predict_fn = load_mock_model(noise_seed=args.seed)

    result = run_cv_benchmark(
        predict_fn=predict_fn,
        images=images,
        ground_truth=ground_truth,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        iou_threshold=args.iou_threshold,
        config={
            "iou_threshold": args.iou_threshold,
            "seed": args.seed,
            "ground_truth_file": args.ground_truth,
        },
    )

    store = ResultStore(results_dir=args.results_dir)
    path = store.save(result)

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nSaved result to: {path}")


if __name__ == "__main__":
    main()
