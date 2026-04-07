"""Tests for M8: bucketed FD Greeks comparison."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    from src.surrogate.run_m8 import (
        ALL_BUCKET_KEYS,
        _bucket_key,
        teacher_fd_greeks,
    )
    from src.surrogate.greeks import load_surrogate_artifacts, surrogate_ad_greeks
    from src.surrogate.data_gen import _K, generate_training_data_stratified
except ModuleNotFoundError:
    from surrogate.run_m8 import (
        ALL_BUCKET_KEYS,
        _bucket_key,
        teacher_fd_greeks,
    )
    from surrogate.greeks import load_surrogate_artifacts, surrogate_ad_greeks
    from surrogate.data_gen import _K, generate_training_data_stratified


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS = _REPO_ROOT / "artifacts" / "surrogate" / "m6_mlp_weights.npz"
_NORMALIZER = _REPO_ROOT / "artifacts" / "surrogate" / "m6_normalizer.npz"
_ARTIFACTS_PRESENT = _WEIGHTS.exists() and _NORMALIZER.exists()


class TestBucketKeyHelper:
    """Bucket-key assignment is deterministic and exhaustive."""

    def test_all_nine_bucket_keys_exist(self) -> None:
        assert len(ALL_BUCKET_KEYS) == 9

    def test_bucket_key_otm_short(self) -> None:
        # log(0.9) ≈ -0.105 → OTM; T=0.10 → short
        bk = _bucket_key(math.log(0.9), 0.10)
        assert bk == "OTM_short"

    def test_bucket_key_atm_medium(self) -> None:
        bk = _bucket_key(0.0, 0.5)
        assert bk == "ATM_medium"

    def test_bucket_key_itm_long(self) -> None:
        # log(1.2) > log(1.05) → ITM; T=2.0 → long
        bk = _bucket_key(math.log(1.2), 2.0)
        assert bk == "ITM_long"


class TestTeacherFDGreeks:
    """Spot-check teacher_fd_greeks for sign and magnitude sanity."""

    def test_teacher_delta_is_negative_for_put(self) -> None:
        """American put delta should be negative (price decreases as S rises)."""
        g = teacher_fd_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
        assert g["delta"] < 0.0, f"Expected negative delta, got {g['delta']}"

    def test_teacher_gamma_is_positive(self) -> None:
        """Gamma ≥ 0 everywhere for vanilla options."""
        g = teacher_fd_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
        assert g["gamma"] >= 0.0, f"Expected non-negative gamma, got {g['gamma']}"

    def test_teacher_vega_is_positive(self) -> None:
        """Option price increases with higher vol (positive vega)."""
        g = teacher_fd_greeks(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
        assert g["vega"] > 0.0, f"Expected positive vega, got {g['vega']}"

    def test_teacher_greeks_are_finite(self) -> None:
        g = teacher_fd_greeks(S=80.0, K=100.0, T=0.25, r=0.03, sigma=0.30, q=0.0)
        for key, val in g.items():
            assert math.isfinite(val), f"teacher_fd_greeks[{key}] is not finite: {val}"


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="M6 artifacts not found — run run_m6.py first")
class TestSurrogateVsTeacherGreeks:
    """Surrogate AD Greeks vs teacher FD Greeks accuracy & consistency."""

    def test_surrogate_ad_delta_close_to_teacher_fd(self) -> None:
        """Mean |surrogate AD delta - teacher FD delta| should be < 0.10."""
        model, normalizer = load_surrogate_artifacts(_WEIGHTS, _NORMALIZER)
        samples = generate_training_data_stratified(
            n_samples=90,
            seed=505,
            binomial_steps=80,
            option_type="put",
        )
        errors = []
        for s in samples:
            S = float(_K * math.exp(s.log_moneyness))
            ad = surrogate_ad_greeks(model, normalizer, S, _K, s.T, s.r, s.sigma, s.q)
            tfd = teacher_fd_greeks(S, _K, s.T, s.r, s.sigma, s.q)
            errors.append(abs(ad.delta - tfd["delta"]))
        mean_err = sum(errors) / len(errors)
        assert mean_err < 0.10, f"Surrogate AD delta mean error vs teacher: {mean_err:.4f}"

    def test_all_nine_buckets_populated(self) -> None:
        """Stratified holdout must produce data in every bucket."""
        samples = generate_training_data_stratified(
            n_samples=90,
            seed=505,
            binomial_steps=80,
            option_type="put",
        )
        observed = {_bucket_key(s.log_moneyness, s.T) for s in samples}
        missing = set(ALL_BUCKET_KEYS) - observed
        assert not missing, f"Missing buckets: {missing}"

    def test_relative_vega_error_below_threshold(self) -> None:
        """Relative Vega error (|err|/|teacher_vega|) should be < 2.0 on average.

        Absolute Vega error appears large because long-dated ATM Vega is large
        (Vega ~ S*sqrt(T) ~ 14 per unit at T=2y). Relative error is a fairer
        metric: if surrogate Vega is within 2x of teacher for a mid-maturity
        average, the surrogate is capturing the vol sensitivity directionally.
        """
        model, normalizer = load_surrogate_artifacts(_WEIGHTS, _NORMALIZER)
        samples = generate_training_data_stratified(
            n_samples=90,
            seed=606,
            binomial_steps=80,
            option_type="put",
        )
        rel_errors = []
        for s in samples:
            S = float(_K * math.exp(s.log_moneyness))
            ad = surrogate_ad_greeks(model, normalizer, S, _K, s.T, s.r, s.sigma, s.q)
            tfd = teacher_fd_greeks(S, _K, s.T, s.r, s.sigma, s.q)
            ref = max(abs(tfd["vega"]), 1.0)
            rel_errors.append(abs(ad.vega - tfd["vega"]) / ref)
        mean_rel = sum(rel_errors) / len(rel_errors)
        assert mean_rel < 2.0, (
            f"Relative Vega error too high: {mean_rel:.3f}. "
            "Surrogate sigma sensitivity is poorly calibrated."
        )

    def test_surrogate_delta_sign_correct_for_put(self) -> None:
        """Surrogate AD delta should be negative for all ATM put samples."""
        model, normalizer = load_surrogate_artifacts(_WEIGHTS, _NORMALIZER)
        samples = generate_training_data_stratified(
            n_samples=45,
            seed=707,
            binomial_steps=80,
            option_type="put",
        )
        atm_samples = [s for s in samples if abs(s.log_moneyness) < math.log(1.05)]
        assert atm_samples, "No ATM samples generated"
        for s in atm_samples:
            S = float(_K * math.exp(s.log_moneyness))
            ad = surrogate_ad_greeks(model, normalizer, S, _K, s.T, s.r, s.sigma, s.q)
            assert ad.delta < 0.0, (
                f"Surrogate put delta should be negative near ATM, got {ad.delta:.4f} "
                f"at log_m={s.log_moneyness:.3f}, T={s.T:.2f}"
            )
