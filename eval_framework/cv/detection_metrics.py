"""
Computer Vision detection metrics: Precision, Recall, F1, and mAP.

Works on standard bounding-box predictions/ground truth in the form:

    ground_truth = {
        "img_001.jpg": [{"class": "car", "bbox": [x1, y1, x2, y2]}, ...],
        ...
    }
    predictions = {
        "img_001.jpg": [{"class": "car", "bbox": [x1, y1, x2, y2], "score": 0.93}, ...],
        ...
    }

bbox format is [x1, y1, x2, y2] in absolute pixel coordinates.

This is a self-contained reference implementation (no external CV
dependency required) so the framework runs anywhere. For large-scale
production benchmarking, swap in pycocotools; the public functions
here (`precision_recall_f1`, `mean_average_precision`) can be
re-pointed at that implementation without touching the runner.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _match_detections(
    gt_boxes: List[dict], pred_boxes: List[dict], iou_threshold: float
) -> Tuple[int, int, int]:
    """Greedy matching by descending confidence score for one image, one class.

    Returns (true_positives, false_positives, false_negatives).
    """
    preds_sorted = sorted(pred_boxes, key=lambda p: p.get("score", 0.0), reverse=True)
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            i = iou(pred["bbox"], gt["bbox"])
            if i > best_iou:
                best_iou = i
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(
    ground_truth: Dict[str, List[dict]],
    predictions: Dict[str, List[dict]],
    iou_threshold: float = 0.5,
    class_filter: str = None,
) -> Dict[str, float]:
    """Compute overall Precision, Recall, F1 across all images (optionally one class)."""
    total_tp = total_fp = total_fn = 0

    for image_id, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(image_id, [])

        if class_filter:
            gt_boxes = [b for b in gt_boxes if b["class"] == class_filter]
            pred_boxes = [b for b in pred_boxes if b["class"] == class_filter]

        tp, fp, fn = _match_detections(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def average_precision_for_class(
    ground_truth: Dict[str, List[dict]],
    predictions: Dict[str, List[dict]],
    class_name: str,
    iou_threshold: float = 0.5,
) -> float:
    """11-point interpolated Average Precision for a single class."""
    all_preds = []  # (score, image_id, bbox)
    total_gt = 0

    for image_id, gt_boxes in ground_truth.items():
        class_gts = [b for b in gt_boxes if b["class"] == class_name]
        total_gt += len(class_gts)
        for p in predictions.get(image_id, []):
            if p["class"] == class_name:
                all_preds.append((p.get("score", 0.0), image_id, p["bbox"]))

    if total_gt == 0:
        return 0.0
    if not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: x[0], reverse=True)
    matched_gt_by_image = {img: set() for img in ground_truth}

    tps = []
    fps = []
    for score, image_id, bbox in all_preds:
        gt_boxes = [b for b in ground_truth.get(image_id, []) if b["class"] == class_name]
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt_by_image[image_id]:
                continue
            i = iou(bbox, gt["bbox"])
            if i > best_iou:
                best_iou, best_idx = i, idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt_by_image[image_id].add(best_idx)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp = _cumsum(tps)
    cum_fp = _cumsum(fps)
    recalls = [t / total_gt for t in cum_tp]
    precisions = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(cum_tp, cum_fp)]

    # 11-point interpolation (standard PASCAL VOC style)
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += (max(precisions_at_recall) if precisions_at_recall else 0.0) / 11.0

    return round(ap, 4)


def _cumsum(values: List[int]) -> List[int]:
    out, total = [], 0
    for v in values:
        total += v
        out.append(total)
    return out


def mean_average_precision(
    ground_truth: Dict[str, List[dict]],
    predictions: Dict[str, List[dict]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """mAP across all classes present in the ground truth."""
    classes = sorted({b["class"] for boxes in ground_truth.values() for b in boxes})
    if not classes:
        return {"mAP": 0.0, "per_class_ap": {}}

    per_class_ap = {
        cls: average_precision_for_class(ground_truth, predictions, cls, iou_threshold)
        for cls in classes
    }
    mAP = sum(per_class_ap.values()) / len(per_class_ap)
    return {"mAP": round(mAP, 4), "per_class_ap": per_class_ap}
