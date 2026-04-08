"""M10: Stress tests — surrogate robustness under extreme market conditions.

Tests three categories of stress against the M6-trained surrogate:

1. **Spot jumps**: ±30 % from base spot (some combinations push log-moneyness
   outside the training domain [-0.5, 0.5]).
2. **Volatility spike**: sigma = 0.80 (training max was 0.60; 33 % OOD).
3. **Maturity extremes**: T = 0.02 (near-expiry, below training min of 0.08)
   and T = 3.5 (beyond training max of 3.0).

For each scenario the surrogate is evaluated with ``surrogate_ad_greeks`` and
checked for:
  * price finiteness and non-negativity
  * price ≥ intrinsic value (K − S)⁺  (with 0.50 tolerance)
  * delta ∈ [−1.05, 0.05]  (American put, with tolerance for ReLU kinks)

Monotonicity sanity checks (local finite-difference):
  * dP/dS  < 0   (higher spot  → lower put price)
  * dP/dσ  > 0   (higher vol   → higher put price)
Run on both the baseline sample pool and the vol-spike scenario.

Outputs
-------
Writes ``artifacts/surrogate/m10_stress_report.json`` with per-scenario
violation counts, price/delta summary statistics, and monotonicity pass rates.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_gen import _K, SurrogateTrainSample, generate_training_data_stratified
from .greeks import load_surrogate_artifacts, surrogate_ad_greeks
from .mlp import AmericanPriceMLP
from .normalizer import FeatureNormalizer

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Intrinsic lower-bound tolerance (surrogate is approximate).
_INTRINSIC_TOL = 0.50
# Delta validity window for American put: theoretical range is [-1, 0].
_DELTA_LO = -1.05
_DELTA_HI = 0.05


# ---------------------------------------------------------------------------
# Stress scenario definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressPoint:
    """A single parameter set for out-of-sample stress evaluation."""
    log_moneyness: float
    T: float
    r: float
    sigma: float
    q: float


def _make_stress_scenarios(
    base_samples: list[SurrogateTrainSample],
) -> dict[str, list[StressPoint]]:
    """Apply each stress transformation to every base sample.

    Returns a mapping of scenario name → list of StressPoint (same length
    as *base_samples*).  The ``'base'`` key holds the unmodified baseline.
    """

    def _to(
        s: SurrogateTrainSample,
        *,
        lm: float | None = None,
        T: float | None = None,
        sigma: float | None = None,
    ) -> StressPoint:
        return StressPoint(
            log_moneyness=lm if lm is not None else s.log_moneyness,
            T=T if T is not None else s.T,
            r=s.r,
            sigma=sigma if sigma is not None else s.sigma,
            q=s.q,
        )

    return {
        "base": [
            _to(s) for s in base_samples
        ],
        "spot_jump_up_30pct": [
            _to(s, lm=s.log_moneyness + math.log(1.3)) for s in base_samples
        ],
        "spot_jump_down_30pct": [
            _to(s, lm=s.log_moneyness + math.log(0.7)) for s in base_samples
        ],
        "vol_spike_80pct": [
            _to(s, sigma=0.80) for s in base_samples
        ],
        "near_expiry": [
            _to(s, T=0.02) for s in base_samples
        ],
        "extreme_long_maturity": [
            _to(s, T=3.5) for s in base_samples
        ],
    }


# ---------------------------------------------------------------------------
# Scenario evaluator
# ---------------------------------------------------------------------------

def _evaluate_scenario(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    points: list[StressPoint],
    K: float = _K,
) -> dict[str, Any]:
    """Evaluate surrogate on all stress points; return validity statistics."""
    prices: list[float] = []
    deltas: list[float] = []
    n_below_zero = 0
    n_below_intrinsic = 0
    n_delta_oor = 0  # out-of-range

    for pt in points:
        S = K * math.exp(pt.log_moneyness)
        g = surrogate_ad_greeks(model, normalizer, S, K, pt.T, pt.r, pt.sigma, pt.q)
        prices.append(g.price)
        deltas.append(g.delta)

        intrinsic = max(0.0, K - S)
        if g.price < -0.01:
            n_below_zero += 1
        if g.price < intrinsic - _INTRINSIC_TOL:
            n_below_intrinsic += 1
        if not (_DELTA_LO <= g.delta <= _DELTA_HI):
            n_delta_oor += 1

    n = len(points)
    arr_p = np.array(prices, dtype=np.float64)
    arr_d = np.array(deltas, dtype=np.float64)

    def _pass_pct(violations: int) -> float:
        return 100.0 * (n - violations) / n

    return {
        "n": n,
        "price": {
            "non_negative_pct": _pass_pct(n_below_zero),
            "above_intrinsic_pct": _pass_pct(n_below_intrinsic),
            "mean": float(arr_p.mean()),
            "p5": float(np.quantile(arr_p, 0.05)),
            "p95": float(np.quantile(arr_p, 0.95)),
            "min": float(arr_p.min()),
        },
        "delta": {
            "sign_pass_pct": _pass_pct(n_delta_oor),
            "mean": float(arr_d.mean()),
            "p5": float(np.quantile(arr_d, 0.05)),
            "p95": float(np.quantile(arr_d, 0.95)),
        },
    }


# ---------------------------------------------------------------------------
# Monotonicity checker
# ---------------------------------------------------------------------------

def _monotonicity_check(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    samples: list[SurrogateTrainSample],
    K: float = _K,
) -> dict[str, Any]:
    """Check local monotonicity via 1 % finite-difference nudges.

    * dP/dS  < 0  tested by comparing P(S) vs P(1.01 · S).
    * dP/dσ  > 0  tested by comparing P(σ) vs P(min(1.10 · σ, 0.95)).
    """
    n = len(samples)
    spot_pass = 0
    vol_pass = 0

    for s in samples:
        S = K * math.exp(s.log_moneyness)
        g0 = surrogate_ad_greeks(model, normalizer, S, K, s.T, s.r, s.sigma, s.q)

        # Spot monotonicity — higher S → lower put price
        S_up = S * 1.01
        g1 = surrogate_ad_greeks(model, normalizer, S_up, K, s.T, s.r, s.sigma, s.q)
        if g1.price < g0.price + 1e-6:
            spot_pass += 1

        # Vol monotonicity — higher sigma → higher put price
        sigma_up = min(s.sigma * 1.10, 0.95)
        g2 = surrogate_ad_greeks(model, normalizer, S, K, s.T, s.r, sigma_up, s.q)
        if g2.price > g0.price - 1e-6:
            vol_pass += 1

    return {
        "n": n,
        "spot_monotonicity_pass_pct": 100.0 * spot_pass / n,
        "vol_monotonicity_pass_pct": 100.0 * vol_pass / n,
        "spot_mono_violations": n - spot_pass,
        "vol_mono_violations": n - vol_pass,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> dict[str, Any]:
    out_dir = _REPO_ROOT / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer = load_surrogate_artifacts(
        out_dir / "m6_mlp_weights.npz",
        out_dir / "m6_normalizer.npz",
    )

    # 90 stratified samples → 10 per bucket, all 9 cells represented.
    base_samples = generate_training_data_stratified(
        n_samples=90,
        seed=1010,
        binomial_steps=80,
        option_type="put",
    )

    # ------------------------------------------------------------------
    # Scenario-level validity checks
    # ------------------------------------------------------------------
    stress_scenarios = _make_stress_scenarios(base_samples)
    scenario_results: dict[str, Any] = {
        name: _evaluate_scenario(model, normalizer, points)
        for name, points in stress_scenarios.items()
    }

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    mono_base = _monotonicity_check(model, normalizer, base_samples)

    # Repeat monotonicity at sigma = 0.80 (OOD vol spike).
    vol_spike_samples = [
        SurrogateTrainSample(
            log_moneyness=s.log_moneyness,
            T=s.T,
            r=s.r,
            sigma=0.80,
            q=s.q,
            price_normalized=0.0,  # no label used
        )
        for s in base_samples
    ]
    mono_vol_spike = _monotonicity_check(model, normalizer, vol_spike_samples)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    payload: dict[str, Any] = {
        "module": "M10",
        "description": "Stress tests: surrogate robustness under extreme market conditions",
        "source_model": "m6",
        "n_base_samples": len(base_samples),
        "intrinsic_tolerance": _INTRINSIC_TOL,
        "delta_validity_window": [_DELTA_LO, _DELTA_HI],
        "scenarios": scenario_results,
        "monotonicity_base": mono_base,
        "monotonicity_vol_spike": mono_vol_spike,
    }

    out_path = out_dir / "m10_stress_report.json"
    out_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Summary print
    # ------------------------------------------------------------------
    print(f"[M10] report  → {out_path}")
    print(
        f"[M10] base      spot-mono {mono_base['spot_monotonicity_pass_pct']:.1f}%"
        f"  vol-mono {mono_base['vol_monotonicity_pass_pct']:.1f}%"
    )
    print(
        f"[M10] vol-spike spot-mono {mono_vol_spike['spot_monotonicity_pass_pct']:.1f}%"
        f"  vol-mono {mono_vol_spike['vol_monotonicity_pass_pct']:.1f}%"
    )
    for name, res in scenario_results.items():
        p = res["price"]
        d = res["delta"]
        print(
            f"[M10] {name:28s}  "
            f"price_nn={p['non_negative_pct']:5.1f}%  "
            f"above_intrinsic={p['above_intrinsic_pct']:5.1f}%  "
            f"delta_sign={d['sign_pass_pct']:5.1f}%"
        )

    return payload


if __name__ == "__main__":
    main()
