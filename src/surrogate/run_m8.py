"""M8: Bucketed FD-Greeks comparison.

Three-way comparison per moneyness × maturity bucket:
    1. Surrogate AD Greeks       (M7 analytical chain rule)
    2. Surrogate FD Greeks       (central difference on surrogate)
    3. Teacher FD Greeks         (central difference on Binomial pricer)

Key questions answered
----------------------
- Where do surrogate AD Greeks deviate most from the Binomial teacher?
- Do Gamma/Vega tail errors (flagged in M7) concentrate in specific buckets?
- Is the surrogate FD a faithful proxy for teacher FD (sanity check)?

Theory note
-----------
For American puts the Binomial teacher provides FD Delta/Gamma/Vega by
revaluing at ±ε perturbations.  Surrogate AD obtains the same quantities via
the chain rule through the network, which aggregates all piecewise-ReLU
kinks.  Discrepancies indicate either (a) surrogate pricing errors propagated
into Greeks or (b) the ReLU approximation of curvature (Gamma) losing accuracy
near kink boundaries.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price
except ModuleNotFoundError:
    from src.pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price

from .data_gen import (
    _K,
    generate_training_data_stratified,
    maturity_bucket_from_T,
    moneyness_bucket_from_log_moneyness,
)
from .greeks import load_surrogate_artifacts, surrogate_ad_greeks, surrogate_fd_greeks


# ---------------------------------------------------------------------------
# Teacher FD Greeks (Binomial pricer as ground truth)
# ---------------------------------------------------------------------------

_TEACHER_STEPS = 100  # Fewer than training to keep M8 tractable; still accurate


def _teacher_price(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Price one American put via Binomial tree."""
    inp = AmericanOptionInput(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        dividend_model=FlatContinuousDividendYield(rate=q),
    )
    return float(american_binomial_price(inp, steps=_TEACHER_STEPS))


def teacher_fd_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    ds_rel: float = 5e-3,
    dvol: float = 5e-3,
) -> dict[str, float]:
    """Central-difference Greeks from the Binomial teacher.

    Returns a dict with keys ``price``, ``delta``, ``gamma``, ``vega``.

    We use ds_rel=0.5 % and dvol=0.5 % — large enough to avoid numerical
    noise from the discrete tree but small enough for accurate derivatives.
    """
    ds = max(1e-3, ds_rel * S)

    p0 = _teacher_price(S, K, T, r, sigma, q)
    p_up = _teacher_price(S + ds, K, T, r, sigma, q)
    p_dn = _teacher_price(S - ds, K, T, r, sigma, q)

    delta = (p_up - p_dn) / (2.0 * ds)
    gamma = (p_up - 2.0 * p0 + p_dn) / (ds * ds)

    v_up = _teacher_price(S, K, T, r, sigma + dvol, q)
    v_dn = _teacher_price(S, K, T, r, max(1e-4, sigma - dvol), q)
    vega = (v_up - v_dn) / (2.0 * dvol)

    return {"price": p0, "delta": delta, "gamma": gamma, "vega": vega}


# ---------------------------------------------------------------------------
# Bucketed summary helpers
# ---------------------------------------------------------------------------

MONEYNESS_BUCKETS = ["OTM", "ATM", "ITM"]
MATURITY_BUCKETS = ["short", "medium", "long"]
ALL_BUCKET_KEYS = [f"{m}_{t}" for m in MONEYNESS_BUCKETS for t in MATURITY_BUCKETS]


def _bucket_key(log_moneyness: float, T: float) -> str:
    return f"{moneyness_bucket_from_log_moneyness(log_moneyness)}_{maturity_bucket_from_T(T)}"


def _abs_errors_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "n":    len(arr),
        "mean": float(arr.mean()),
        "p50":  float(np.quantile(arr, 0.50)),
        "p95":  float(np.quantile(arr, 0.95)),
        "max":  float(arr.max()),
    }


def _bucket_table(
    bucket_errors: dict[str, list[float]],
) -> dict[str, Any]:
    return {k: _abs_errors_summary(bucket_errors.get(k, [])) for k in ALL_BUCKET_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "m6_mlp_weights.npz"
    normalizer_path = out_dir / "m6_normalizer.npz"

    model, normalizer = load_surrogate_artifacts(weights_path, normalizer_path)

    # Stratified hold-out: guaranteed coverage of all 9 buckets.
    eval_samples = generate_training_data_stratified(
        n_samples=180,
        seed=404,
        binomial_steps=100,
        option_type="put",
    )

    # Accumulators  — three comparisons × three Greeks
    # Keys: "delta", "gamma", "vega"
    ad_vs_sfd: dict[str, dict[str, list[float]]] = {
        g: defaultdict(list) for g in ("delta", "gamma", "vega")
    }
    ad_vs_tfd: dict[str, dict[str, list[float]]] = {
        g: defaultdict(list) for g in ("delta", "gamma", "vega")
    }
    sfd_vs_tfd: dict[str, dict[str, list[float]]] = {
        g: defaultdict(list) for g in ("delta", "gamma", "vega")
    }

    # Per-comparison overall lists (across all buckets)
    ad_vs_sfd_all: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}
    ad_vs_tfd_all: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}
    sfd_vs_tfd_all: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}

    # Relative error trackers: |err| / max(|teacher|, floor)
    # Useful for Vega which varies hugely with T.
    _REL_FLOOR = {"delta": 0.01, "gamma": 0.001, "vega": 1.0}
    ad_vs_tfd_rel: dict[str, dict[str, list[float]]] = {
        g: defaultdict(list) for g in ("delta", "gamma", "vega")
    }
    ad_vs_tfd_rel_all: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}

    for sample in eval_samples:
        S = float(_K * math.exp(sample.log_moneyness))
        bkey = _bucket_key(sample.log_moneyness, sample.T)

        ad = surrogate_ad_greeks(model, normalizer, S, _K, sample.T, sample.r, sample.sigma, sample.q)
        sfd = surrogate_fd_greeks(model, normalizer, S, _K, sample.T, sample.r, sample.sigma, sample.q)
        tfd = teacher_fd_greeks(S, _K, sample.T, sample.r, sample.sigma, sample.q)

        for greek in ("delta", "gamma", "vega"):
            a = getattr(ad, greek)
            s = getattr(sfd, greek)
            t = tfd[greek]

            err_as = abs(a - s)
            err_at = abs(a - t)
            err_st = abs(s - t)

            ad_vs_sfd[greek][bkey].append(err_as)
            ad_vs_tfd[greek][bkey].append(err_at)
            sfd_vs_tfd[greek][bkey].append(err_st)

            ad_vs_sfd_all[greek].append(err_as)
            ad_vs_tfd_all[greek].append(err_at)
            sfd_vs_tfd_all[greek].append(err_st)

            # Relative error: normalise by magnitude of teacher Greek
            ref_mag = max(abs(t), _REL_FLOOR[greek])
            rel_err = err_at / ref_mag
            ad_vs_tfd_rel[greek][bkey].append(rel_err)
            ad_vs_tfd_rel_all[greek].append(rel_err)

    def _comparison_section(
        overall: dict[str, list[float]],
        bucketed: dict[str, dict[str, list[float]]],
    ) -> dict[str, Any]:
        return {
            g: {
                "overall": _abs_errors_summary(overall[g]),
                "by_bucket": _bucket_table(bucketed[g]),
            }
            for g in ("delta", "gamma", "vega")
        }

    payload: dict[str, Any] = {
        "module": "M8",
        "description": "Bucketed FD Greeks comparison: AD vs surrogate-FD vs teacher-FD",
        "source_model": "m6",
        "teacher_fd_steps": _TEACHER_STEPS,
        "n_eval": len(eval_samples),
        # 1. Sanity: should be very small — AD and surrogate FD should agree.
        "ad_vs_surrogate_fd": _comparison_section(ad_vs_sfd_all, ad_vs_sfd),
        # 2. Key accuracy: how well does surrogate AD match the Binomial teacher?
        "surrogate_ad_vs_teacher_fd": _comparison_section(ad_vs_tfd_all, ad_vs_tfd),
        # 3. Consistency: surrogate FD vs teacher FD (pricing-level errors only).
        "surrogate_fd_vs_teacher_fd": _comparison_section(sfd_vs_tfd_all, sfd_vs_tfd),
        # 4. Relative accuracy: |err| / max(|teacher|, floor) — normalises Vega by tenor.
        #    floor: delta=0.01, gamma=0.001, vega=1.0
        "surrogate_ad_vs_teacher_fd_relative": _comparison_section(
            ad_vs_tfd_rel_all, ad_vs_tfd_rel
        ),
    }

    report_path = out_dir / "m8_fd_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"M8 report saved -> {report_path}")
    print(f"\n{'='*60}")
    print("surrogate AD vs surrogate FD (internal sanity)")
    for g in ("delta", "gamma", "vega"):
        s = payload["ad_vs_surrogate_fd"][g]["overall"]
        print(f"  {g:6s}: mean={s['mean']:.6f}  p95={s['p95']:.6f}  max={s['max']:.6f}")

    print(f"\nsurrogate AD vs teacher FD (key accuracy check)")
    for g in ("delta", "gamma", "vega"):
        s = payload["surrogate_ad_vs_teacher_fd"][g]["overall"]
        print(f"  {g:6s}: mean={s['mean']:.6f}  p95={s['p95']:.6f}  max={s['max']:.6f}")

    print(f"\nsurrogate FD vs teacher FD (consistency check)")
    for g in ("delta", "gamma", "vega"):
        s = payload["surrogate_fd_vs_teacher_fd"][g]["overall"]
        print(f"  {g:6s}: mean={s['mean']:.6f}  p95={s['p95']:.6f}  max={s['max']:.6f}")

    # Highlight worst buckets for surrogate AD vs teacher FD (Gamma, Vega).
    print(f"\n{'='*60}")
    print("Top-3 worst buckets: surrogate AD vs teacher FD — GAMMA")
    gamma_buckets = {
        bk: payload["surrogate_ad_vs_teacher_fd"]["gamma"]["by_bucket"][bk]["mean"]
        for bk in ALL_BUCKET_KEYS
        if payload["surrogate_ad_vs_teacher_fd"]["gamma"]["by_bucket"][bk]["n"] > 0
    }
    for bk, val in sorted(gamma_buckets.items(), key=lambda x: -x[1])[:3]:
        print(f"  {bk}: mean={val:.6f}")

    print("Top-3 worst buckets: surrogate AD vs teacher FD — VEGA")
    vega_buckets = {
        bk: payload["surrogate_ad_vs_teacher_fd"]["vega"]["by_bucket"][bk]["mean"]
        for bk in ALL_BUCKET_KEYS
        if payload["surrogate_ad_vs_teacher_fd"]["vega"]["by_bucket"][bk]["n"] > 0
    }
    for bk, val in sorted(vega_buckets.items(), key=lambda x: -x[1])[:3]:
        print(f"  {bk}: mean={val:.6f}")

    print(f"\nsurrogate AD vs teacher FD — RELATIVE errors (|err|/|teacher|)")
    for g in ("delta", "gamma", "vega"):
        s = payload["surrogate_ad_vs_teacher_fd_relative"][g]["overall"]
        print(f"  {g:6s}: mean={s['mean']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}")

    print("\nTop-3 worst buckets: relative VEGA error")
    vega_rel_buckets = {
        bk: payload["surrogate_ad_vs_teacher_fd_relative"]["vega"]["by_bucket"][bk]["mean"]
        for bk in ALL_BUCKET_KEYS
        if payload["surrogate_ad_vs_teacher_fd_relative"]["vega"]["by_bucket"][bk]["n"] > 0
    }
    for bk, val in sorted(vega_rel_buckets.items(), key=lambda x: -x[1])[:3]:
        print(f"  {bk}: mean={val:.4f}")


if __name__ == "__main__":
    main()
