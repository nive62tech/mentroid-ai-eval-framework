import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_framework.core.timing import Timer, LatencyTracker
from eval_framework.core.results import EvalResult, ResultStore


def test_timer_measures_elapsed():
    with Timer() as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 10.0


def test_latency_tracker_summary():
    tracker = LatencyTracker()
    for ms in [10, 20, 30, 40, 50]:
        tracker.record(ms)
    summary = tracker.summary()
    assert summary["count"] == 5
    assert summary["mean_ms"] == 30.0
    assert summary["fps"] > 0


def test_latency_tracker_empty():
    tracker = LatencyTracker()
    summary = tracker.summary()
    assert summary["count"] == 0
    assert summary["mean_ms"] is None


def test_eval_result_serializes():
    result = EvalResult(
        task_type="cv",
        model_name="test-model",
        dataset_name="test-dataset",
        metrics={"precision": 0.9},
    )
    d = result.to_dict()
    assert d["model_name"] == "test-model"
    assert d["metrics"]["precision"] == 0.9
    assert "timestamp" in d


def test_result_store_save_and_load():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ResultStore(results_dir=tmp_dir)
        result = EvalResult(
            task_type="cv",
            model_name="model-a",
            dataset_name="ds",
            metrics={"precision": 0.8},
        )
        path = store.save(result)
        assert os.path.exists(path)

        history = store.load_history()
        assert len(history) == 1
        assert history[0]["model_name"] == "model-a"
    finally:
        shutil.rmtree(tmp_dir)


def test_result_store_compare():
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ResultStore(results_dir=tmp_dir)
        store.save(EvalResult(task_type="cv", model_name="model-a", dataset_name="ds", metrics={"precision": 0.8}))
        store.save(EvalResult(task_type="cv", model_name="model-b", dataset_name="ds", metrics={"precision": 0.9}))

        comparison = store.compare(["model-a", "model-b"])
        assert comparison["metrics"]["precision"]["model-a"] == 0.8
        assert comparison["metrics"]["precision"]["model-b"] == 0.9
        assert comparison["missing_models"] == []
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
