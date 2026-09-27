import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.cv.detection_metrics import (
    iou,
    precision_recall_f1,
    mean_average_precision,
)


def test_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert iou(box, box) == 1.0


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    result = iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < result < 1.0


def test_precision_recall_perfect_match():
    gt = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10]}]}
    pred = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10], "score": 0.9}]}
    result = precision_recall_f1(gt, pred, iou_threshold=0.5)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_precision_recall_missed_detection():
    gt = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10]}]}
    pred = {"img1": []}
    result = precision_recall_f1(gt, pred, iou_threshold=0.5)
    assert result["recall"] == 0.0
    assert result["false_negatives"] == 1


def test_precision_recall_false_positive():
    gt = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10]}]}
    pred = {"img1": [
        {"class": "car", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"class": "car", "bbox": [50, 50, 60, 60], "score": 0.8},
    ]}
    result = precision_recall_f1(gt, pred, iou_threshold=0.5)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1


def test_mean_average_precision_perfect():
    gt = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10]}]}
    pred = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10], "score": 0.99}]}
    result = mean_average_precision(gt, pred, iou_threshold=0.5)
    assert result["mAP"] == 1.0


def test_mean_average_precision_no_predictions():
    gt = {"img1": [{"class": "car", "bbox": [0, 0, 10, 10]}]}
    pred = {"img1": []}
    result = mean_average_precision(gt, pred, iou_threshold=0.5)
    assert result["mAP"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
