"""
CV evaluation runner.

Wraps a user-supplied `predict_fn(image) -> List[{"class", "bbox", "score"}]`
around the detection metrics and timing utilities, so a full benchmark
(accuracy metrics + latency + FPS) is one function call.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from eval_framework.core.timing import LatencyTracker, Timer
from eval_framework.core.results import EvalResult
from eval_framework.cv.detection_metrics import precision_recall_f1, mean_average_precision


def run_cv_benchmark(
    predict_fn: Callable[[object], List[dict]],
    images: Dict[str, object],
    ground_truth: Dict[str, List[dict]],
    model_name: str,
    dataset_name: str,
    iou_threshold: float = 0.5,
    warmup_runs: int = 3,
    config: Optional[dict] = None,
) -> EvalResult:
    """Run a full CV benchmark: accuracy metrics + latency + FPS.

    Args:
        predict_fn: function taking one loaded image and returning a list of
            detections: [{"class": str, "bbox": [x1,y1,x2,y2], "score": float}, ...]
        images: dict mapping image_id -> loaded image object (whatever predict_fn expects)
        ground_truth: dict mapping image_id -> list of {"class", "bbox"} ground-truth boxes
        model_name: identifier for the model/version under test
        dataset_name: identifier for the dataset used
        iou_threshold: IoU threshold for counting a detection as a match
        warmup_runs: number of un-timed warmup calls (avoids cold-start skew)
        config: arbitrary config dict to store alongside results for reproducibility

    Returns:
        EvalResult with precision/recall/f1/mAP and latency/FPS stats.
    """
    image_items = list(images.items())
    if not image_items:
        raise ValueError("No images provided for CV benchmark.")

    # Warmup (not timed) — avoids first-call overhead skewing latency stats
    for image_id, img in image_items[:warmup_runs]:
        predict_fn(img)

    predictions: Dict[str, List[dict]] = {}
    tracker = LatencyTracker()

    for image_id, img in image_items:
        with Timer() as t:
            preds = predict_fn(img)
        tracker.record_from(t)
        predictions[image_id] = preds

    accuracy_metrics = precision_recall_f1(ground_truth, predictions, iou_threshold)
    map_metrics = mean_average_precision(ground_truth, predictions, iou_threshold)
    latency_stats = tracker.summary()

    metrics = {
        "precision": accuracy_metrics["precision"],
        "recall": accuracy_metrics["recall"],
        "f1_score": accuracy_metrics["f1_score"],
        "mAP": map_metrics["mAP"],
        "per_class_ap": map_metrics["per_class_ap"],
        "fps": latency_stats["fps"],
        "latency_ms_mean": latency_stats["mean_ms"],
        "latency_ms_p95": latency_stats["p95_ms"],
        "latency_ms_p99": latency_stats["p99_ms"],
        "num_images": len(image_items),
        "iou_threshold": iou_threshold,
    }

    return EvalResult(
        task_type="cv",
        model_name=model_name,
        dataset_name=dataset_name,
        metrics=metrics,
        config=config or {},
    )
