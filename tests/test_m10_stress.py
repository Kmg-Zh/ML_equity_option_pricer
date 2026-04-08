"""Tests for M10 stress-test runner.

Unit tests verify internal helper logic (no trained model required).
The final integration test validates the JSON artifact schema if the runner
has already been executed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

try:
    from src.surrogate.run_m10 import (
        StressPoint,
        _DELTA_HI,
        _DELTA_LO,
        _make_stress_scenarios,
    )
    from src.surrogate.data_gen import SurrogateTrainSample
except ModuleNotFoundError:
    from surrogate.run_m10 import (
        StressPoint,
        _DELTA_HI,
        _DELTA_LO,
        _make_stress_scenarios,
    )
    from surrogate.data_gen import SurrogateTrainSample

_REPORT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "surrogate"
    / "m10_stress_report.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake(
    log_moneyness: float = 0.0,
    T: float = 0.5,
    sigma: float = 0.20,
) -> SurrogateTrainSample:
    return SurrogateTrainSample(
        log_moneyness=log_moneyness,
        T=T,
        r=0.03,
        sigma=sigma,
        q=0.02,
        price_normalized=0.0,
    )


# ---------------------------------------------------------------------------
# Unit tests — no model artifacts required
# ---------------------------------------------------------------------------

def test_stress_scenario_keys() -> None:
    """_make_stress_scenarios returns exactly the 6 expected scenario names."""
    scenarios = _make_stress_scenarios([_fake()])
    expected = {
        "base",
        "spot_jump_up_30pct",
        "spot_jump_down_30pct",
        "vol_spike_80pct",
        "near_expiry",
        "extreme_long_maturity",
    }
    assert set(scenarios.keys()) == expected


def test_stress_scenario_lengths() -> None:
    """Each scenario contains one StressPoint per base sample."""
    n = 7
    samples = [_fake(math.log(0.9 + 0.03 * i)) for i in range(n)]
    scenarios = _make_stress_scenarios(samples)
    for name, pts in scenarios.items():
        assert len(pts) == n, f"Scenario '{name}' length mismatch"


def test_vol_spike_sigma_fixed() -> None:
    """vol_spike_80pct must set sigma = 0.80 for every base sample."""
    samples = [_fake(sigma=0.20), _fake(sigma=0.40), _fake(sigma=0.55)]
    for pt in _make_stress_scenarios(samples)["vol_spike_80pct"]:
        assert pt.sigma == pytest.approx(0.80)


def test_near_expiry_T() -> None:
    """near_expiry must set T = 0.02 for every base sample."""
    samples = [_fake(T=1.0), _fake(T=0.5)]
    for pt in _make_stress_scenarios(samples)["near_expiry"]:
        assert pt.T == pytest.approx(0.02)


def test_extreme_long_T() -> None:
    """extreme_long_maturity must set T = 3.5 for every base sample."""
    for pt in _make_stress_scenarios([_fake(T=1.0)])["extreme_long_maturity"]:
        assert pt.T == pytest.approx(3.5)


def test_spot_jump_up_log_moneyness() -> None:
    """Spot +30 % must shift log-moneyness by exactly log(1.3)."""
    base = _fake(log_moneyness=0.0)
    pt = _make_stress_scenarios([base])["spot_jump_up_30pct"][0]
    assert pt.log_moneyness == pytest.approx(math.log(1.3), abs=1e-12)


def test_spot_jump_down_log_moneyness() -> None:
    """Spot −30 % must shift log-moneyness by exactly log(0.7)."""
    base = _fake(log_moneyness=0.0)
    pt = _make_stress_scenarios([base])["spot_jump_down_30pct"][0]
    assert pt.log_moneyness == pytest.approx(math.log(0.7), abs=1e-12)


def test_base_scenario_unchanged() -> None:
    """'base' scenario must be a faithful copy of the input sample."""
    s = _fake(log_moneyness=0.05, T=0.75, sigma=0.30)
    pt = _make_stress_scenarios([s])["base"][0]
    assert isinstance(pt, StressPoint)
    assert pt.log_moneyness == pytest.approx(s.log_moneyness)
    assert pt.T == pytest.approx(s.T)
    assert pt.sigma == pytest.approx(s.sigma)
    assert pt.r == pytest.approx(s.r)
    assert pt.q == pytest.approx(s.q)


def test_delta_validity_window_constants() -> None:
    """Delta validity window constants must bracket the theoretical put range."""
    assert _DELTA_LO < -1.0  # allows slight undershoot
    assert _DELTA_HI > 0.0   # allows slight overshoot


# ---------------------------------------------------------------------------
# Integration test — requires pre-generated artifact
# ---------------------------------------------------------------------------

def test_m10_report_schema() -> None:
    """Validate JSON artifact schema; skipped if file not yet generated."""
    if not _REPORT.exists():
        pytest.skip("m10_stress_report.json not yet generated — run run_m10.py first")

    payload = json.loads(_REPORT.read_text())

    assert payload["module"] == "M10"
    assert "scenarios" in payload
    assert "monotonicity_base" in payload
    assert "monotonicity_vol_spike" in payload
    assert payload["n_base_samples"] > 0

    required_scenarios = {
        "base",
        "spot_jump_up_30pct",
        "spot_jump_down_30pct",
        "vol_spike_80pct",
        "near_expiry",
        "extreme_long_maturity",
    }
    assert required_scenarios.issubset(payload["scenarios"].keys())

    for name in required_scenarios:
        res = payload["scenarios"][name]
        assert "price" in res, f"Scenario '{name}' missing 'price'"
        assert "delta" in res, f"Scenario '{name}' missing 'delta'"
        assert "non_negative_pct" in res["price"]
        assert "above_intrinsic_pct" in res["price"]
        assert "sign_pass_pct" in res["delta"]

    mono = payload["monotonicity_base"]
    assert "spot_monotonicity_pass_pct" in mono
    assert "vol_monotonicity_pass_pct" in mono
    assert 0.0 <= mono["spot_monotonicity_pass_pct"] <= 100.0
    assert 0.0 <= mono["vol_monotonicity_pass_pct"] <= 100.0
