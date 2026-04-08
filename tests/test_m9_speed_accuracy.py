from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from src.surrogate.run_m9 import _abs_summary, _latency_summary
except ModuleNotFoundError:
    from surrogate.run_m9 import _abs_summary, _latency_summary


def test_latency_summary_monotone_percentiles() -> None:
    stats = _latency_summary([0.001, 0.002, 0.003, 0.004, 0.005])
    assert stats["n"] == 5
    assert stats["mean_ms"] > 0.0
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]


def test_abs_summary_basic() -> None:
    stats = _abs_summary([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert np.isclose(stats["mean"], 2.5)
    assert stats["p50"] <= stats["p95"] <= stats["max"]


def test_m9_report_exists_after_run() -> None:
    # This test validates artifact schema only if the runner has been executed.
    repo_root = Path(__file__).resolve().parents[1]
    report = repo_root / "artifacts" / "surrogate" / "m9_speed_accuracy.json"
    if not report.exists():
        return

    import json

    payload = json.loads(report.read_text())
    assert payload["module"] == "M9"
    assert "accuracy" in payload
    assert "frontier" in payload
    assert "pricing" in payload["frontier"]
    assert "greeks" in payload["frontier"]
