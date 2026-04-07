"""M7 AD Greeks for the surrogate model.

The surrogate predicts normalized price z = f(x_norm), with:

    x = [log(S/K), T, r, sigma, q]
    x_norm = (x - mu_x) / std_x
    price = K * (mu_y + std_y * z)

AD provides d z / d x_norm; chain rule gives Greeks in price units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mlp import AmericanPriceMLP
from .normalizer import FeatureNormalizer


@dataclass(frozen=True)
class SurrogateGreeks:
    price: float
    delta: float
    gamma: float
    vega: float


def load_surrogate_artifacts(
    weights_path: str | Path,
    normalizer_path: str | Path,
) -> tuple[AmericanPriceMLP, FeatureNormalizer]:
    """Load model weights and normalizer arrays saved by run_m4/m5/m6."""
    model = AmericanPriceMLP()

    with np.load(weights_path) as w:
        for name in model.params:
            if name not in w:
                raise ValueError(f"Missing parameter {name} in weights file.")
            model.params[name] = np.array(w[name], dtype=np.float64)

    with np.load(normalizer_path) as n:
        normalizer = FeatureNormalizer(
            feature_means=np.array(n["feature_means"], dtype=np.float64),
            feature_stds=np.array(n["feature_stds"], dtype=np.float64),
            label_mean=float(np.array(n["label_mean"]).reshape(-1)[0]),
            label_std=float(np.array(n["label_std"]).reshape(-1)[0]),
        )

    return model, normalizer


def _normalized_input(S: float, K: float, T: float, r: float, sigma: float, q: float) -> np.ndarray:
    if S <= 0.0 or K <= 0.0:
        raise ValueError("S and K must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    x = np.array([[math.log(S / K), T, r, sigma, q]], dtype=np.float64)
    return x


def surrogate_ad_greeks(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> SurrogateGreeks:
    """Return AD Greeks from surrogate.

    Gamma uses the piecewise-linear ReLU property: d²z/d(log-moneyness)² = 0
    almost everywhere, so only chain-rule curvature from log(S/K) is retained.
    """
    x_raw = _normalized_input(S, K, T, r, sigma, q)
    x_norm = (x_raw - normalizer.feature_means) / normalizer.feature_stds

    z_pred, dz_dxnorm = model.predict_and_input_grads(x_norm)
    z = float(z_pred[0, 0])
    dz = dz_dxnorm[0]

    price = K * float(normalizer.inverse_transform_label(z))

    scale = K * normalizer.label_std

    dz_dlm = dz[0] / normalizer.feature_stds[0]
    dz_dsigma = dz[3] / normalizer.feature_stds[3]

    delta = scale * dz_dlm * (1.0 / S)
    gamma = -scale * dz_dlm * (1.0 / (S * S))
    vega = scale * dz_dsigma

    return SurrogateGreeks(price=price, delta=float(delta), gamma=float(gamma), vega=float(vega))


def surrogate_fd_greeks(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    ds_rel: float = 1e-3,
    dvol: float = 1e-3,
) -> SurrogateGreeks:
    """Finite-difference Greeks for sanity checks (used for M7 validation only)."""
    ds = max(1e-4, ds_rel * S)

    p0 = surrogate_ad_greeks(model, normalizer, S, K, T, r, sigma, q).price
    p_up = surrogate_ad_greeks(model, normalizer, S + ds, K, T, r, sigma, q).price
    p_dn = surrogate_ad_greeks(model, normalizer, S - ds, K, T, r, sigma, q).price

    delta = (p_up - p_dn) / (2.0 * ds)
    gamma = (p_up - 2.0 * p0 + p_dn) / (ds * ds)

    v_up = surrogate_ad_greeks(model, normalizer, S, K, T, r, sigma + dvol, q).price
    v_dn = surrogate_ad_greeks(model, normalizer, S, K, T, r, max(1e-4, sigma - dvol), q).price
    vega = (v_up - v_dn) / (2.0 * dvol)

    return SurrogateGreeks(price=p0, delta=float(delta), gamma=float(gamma), vega=float(vega))
