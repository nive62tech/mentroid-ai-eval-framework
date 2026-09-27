"""
Result storage, persistence, and model-version comparison utilities.

Every evaluation run (CV or LLM) produces an EvalResult, which is a
uniform, serializable record. Results are saved as JSON under
`results/` (one file per run) and also appended to a running
`results/history.jsonl` so multiple model versions can be compared
over time.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class EvalResult:
    """Uniform record for a single evaluation run."""

    task_type: str                    # "cv" or "llm"
    model_name: str                   # e.g. "yolov8n-v2" or "gpt-4o-mini"
    dataset_name: str
    metrics: Dict[str, Any]           # e.g. {"precision": 0.89, "recall": 0.84, ...}
    not_applicable: Dict[str, str] = field(default_factory=dict)  # metric -> reason
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def __post_init__(self):
        if not self.environment:
            self.environment = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": _git_commit(),
            }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultStore:
    """Saves EvalResult objects to disk and supports version comparison."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.history_path = os.path.join(self.results_dir, "history.jsonl")

    def save(self, result: EvalResult) -> str:
        """Save a single result as its own timestamped JSON file, and append
        it to the history log used for comparisons. Returns the file path."""
        safe_model = result.model_name.replace("/", "_").replace(" ", "_")
        ts = result.timestamp.replace(":", "-")
        filename = f"{result.task_type}_{safe_model}_{ts}.json"
        path = os.path.join(self.results_dir, filename)

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        with open(self.history_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        return path

    def load_history(self) -> list:
        if not os.path.exists(self.history_path):
            return []
        records = []
        with open(self.history_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def compare(self, model_names: list, metric_keys: Optional[list] = None) -> Dict[str, Any]:
        """Compare the latest run of each given model name across shared metrics.

        Returns a dict: {metric: {model_name: value}} plus a "runs" section
        with the full latest record used for each model.
        """
        history = self.load_history()
        latest_by_model = {}
        for rec in history:
            name = rec["model_name"]
            if name in model_names:
                # keep the most recent (history.jsonl is append-only, so last wins)
                latest_by_model[name] = rec

        missing = [m for m in model_names if m not in latest_by_model]

        if metric_keys is None:
            # union of all metric keys across selected runs
            keys = set()
            for rec in latest_by_model.values():
                keys.update(rec["metrics"].keys())
            metric_keys = sorted(keys)

        comparison: Dict[str, Any] = {"metrics": {}, "runs": latest_by_model, "missing_models": missing}
        for key in metric_keys:
            comparison["metrics"][key] = {
                name: rec["metrics"].get(key, "N/A")
                for name, rec in latest_by_model.items()
            }
        return comparison
