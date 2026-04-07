"""Bucketed accuracy evaluation for the M4 MLP surrogate.

Buckets
-------
Same moneyness × maturity bucketing as the M3 benchmark harness:

    Moneyness S/K : OTM (< 0.95), ATM (0.95–1.05), ITM (> 1.05)
    Maturity T    : short (≤ 0.25 y), medium (0.25–1.0 y), long (> 1.0 y)

Evaluation uses the Binomial-teacher labels already stored in each sample's
``price_normalized`` field — no re-pricing.  This ensures the comparison is
fair: surrogate error is measured against the *same* teacher used for training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data_gen import SurrogateTrainSample, _K
from .mlp import AmericanPriceMLP
from .normalizer import FeatureNormalizer


@dataclass(frozen=True)
class BucketEvalResult:
    """Surrogate accuracy statistics for one moneyness × maturity cell.

    All error metrics are in *dollar* units (not normalised), benchmarked
    against the Binomial teacher.
    """
    moneyness_bucket: str
    maturity_bucket:  str
    count:    int
    mae:      float   # mean absolute error  (price units)
    rmse:     float   # root mean squared error
    max_ae:   float   # maximum absolute error


@dataclass(frozen=True)
class FinancialConstraintMetrics:
    """Global M6 constraint diagnostics on a sample set."""

    monotonicity_violation_rate: float
    monotonicity_violation_mean: float
    convexity_violation_rate: float
    convexity_violation_mean: float


def _moneyness_bucket(log_moneyness: float) -> str:
    m = math.exp(log_moneyness)
    if m < 0.95:
        return "OTM"
    if m <= 1.05:
        return "ATM"
    return "ITM"


def _maturity_bucket(T: float) -> str:
    if T <= 0.25:
        return "short"
    if T <= 1.0:
        return "medium"
    return "long"


def predict_price(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    sample: SurrogateTrainSample,
    K: float = _K,
) -> float:
    """Run one forward pass and return a price in dollar terms."""
    X = normalizer.transform_features([sample])   # (1, 5) float64
    z = model.predict(X).item()                   # standardised label (scalar)
    price_over_K = float(normalizer.inverse_transform_label(z))
    return price_over_K * K


def evaluate_surrogate(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    eval_samples: list[SurrogateTrainSample],
    K: float = _K,
) -> list[BucketEvalResult]:
    """Evaluate surrogate accuracy across moneyness × maturity buckets.

    Uses the Binomial teacher label already embedded in each sample
    (``price_normalized``), so no re-pricing is required.

    Returns a sorted list of :class:`BucketEvalResult` objects.
    """
    # Batch prediction for speed
    X = normalizer.transform_features(eval_samples)        # (N, 5)
    z_pred = model.predict(X).ravel()                      # (N,)
    prices_pred = normalizer.inverse_transform_label(z_pred) * K   # (N,)
    prices_true = np.array([s.price_normalized for s in eval_samples]) * K

    errors = prices_pred - prices_true

    buckets: dict[tuple[str, str], list[float]] = {}
    for i, sample in enumerate(eval_samples):
        key = (
            _moneyness_bucket(sample.log_moneyness),
            _maturity_bucket(sample.T),
        )
        buckets.setdefault(key, []).append(float(errors[i]))

    results: list[BucketEvalResult] = []
    for key, errs in sorted(buckets.items()):
        m_b, t_b = key
        abs_errs = [abs(e) for e in errs]
        mae  = sum(abs_errs) / len(abs_errs)
        rmse = math.sqrt(sum(e ** 2 for e in errs) / len(errs))
        results.append(BucketEvalResult(
            moneyness_bucket=m_b,
            maturity_bucket=t_b,
            count=len(errs),
            mae=mae,
            rmse=rmse,
            max_ae=max(abs_errs),
        ))

    return results


def evaluate_financial_constraints(
        model: AmericanPriceMLP,
        normalizer: FeatureNormalizer,
        eval_samples: list[SurrogateTrainSample],
        eps_lm: float = 0.02,
) -> FinancialConstraintMetrics:
        """Evaluate monotonicity and convexity violations in log-moneyness.

        Monotonicity target for put:
            V(lm + eps) - V(lm - eps) <= 0

        Convexity target proxy:
            V(lm + eps) - 2V(lm) + V(lm - eps) >= 0
        """
        if not eval_samples:
                return FinancialConstraintMetrics(0.0, 0.0, 0.0, 0.0)

        X = normalizer.transform_features(eval_samples)
        eps_z = eps_lm / float(normalizer.feature_stds[0])

        X_minus = X.copy()
        X_plus = X.copy()
        X_minus[:, 0] -= eps_z
        X_plus[:, 0] += eps_z

        y_minus = model.predict(X_minus).ravel()
        y_mid = model.predict(X).ravel()
        y_plus = model.predict(X_plus).ravel()

        mono_violation = np.maximum(y_plus - y_minus, 0.0)
        second_diff = y_plus - 2.0 * y_mid + y_minus
        convex_violation = np.maximum(-second_diff, 0.0)

        return FinancialConstraintMetrics(
                monotonicity_violation_rate=float((mono_violation > 0.0).mean()),
                monotonicity_violation_mean=float(mono_violation.mean()),
                convexity_violation_rate=float((convex_violation > 0.0).mean()),
                convexity_violation_mean=float(convex_violation.mean()),
        )
