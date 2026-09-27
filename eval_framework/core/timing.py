"""
Timing utilities shared by CV and LLM evaluators.

Provides:
    Timer          - context manager / manual stopwatch for a single call
    LatencyTracker - accumulates many latency samples and derives statistics
                     (mean, p50, p95, p99, min, max) plus FPS for CV models
"""

from __future__ import annotations

import statistics
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import List


class Timer(ContextDecorator):
    """Simple stopwatch usable as a context manager.

    Example:
        with Timer() as t:
            model.predict(x)
        print(t.elapsed_ms)
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self._end = time.perf_counter()
        return False

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000.0


@dataclass
class LatencyTracker:
    """Collects per-call latency samples (in milliseconds) and computes stats."""

    samples_ms: List[float] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.samples_ms.append(elapsed_ms)

    def record_from(self, timer: Timer) -> None:
        self.record(timer.elapsed_ms)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    def _percentile(self, p: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        k = (len(ordered) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(ordered) - 1)
        if f == c:
            return ordered[f]
        return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {
                "count": 0,
                "mean_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
                "fps": None,
            }
        mean_ms = statistics.mean(self.samples_ms)
        return {
            "count": self.count,
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(self._percentile(50), 3),
            "p95_ms": round(self._percentile(95), 3),
            "p99_ms": round(self._percentile(99), 3),
            "min_ms": round(min(self.samples_ms), 3),
            "max_ms": round(max(self.samples_ms), 3),
            # FPS is only meaningful for per-frame CV latency; harmless for LLM too.
            "fps": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
        }
