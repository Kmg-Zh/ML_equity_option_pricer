from __future__ import annotations

import math

import numpy as np

try:
    from surrogate.data_gen import _K, generate_training_data_stratified
    from surrogate.greeks import surrogate_ad_greeks, surrogate_fd_greeks
    from surrogate.train import TrainConfig, train_surrogate
except ModuleNotFoundError:
    from src.surrogate.data_gen import _K, generate_training_data_stratified
    from src.surrogate.greeks import surrogate_ad_greeks, surrogate_fd_greeks
    from src.surrogate.train import TrainConfig, train_surrogate


def test_ad_greeks_are_finite() -> None:
    samples = generate_training_data_stratified(n_samples=180, seed=12, binomial_steps=40)
    cfg = TrainConfig(lr=1e-2, epochs=14, batch_size=24, seed=0)
    model, normalizer, _ = train_surrogate(samples, cfg)

    s0 = samples[0]
    S = _K * math.exp(s0.log_moneyness)
    g = surrogate_ad_greeks(model, normalizer, S=S, K=_K, T=s0.T, r=s0.r, sigma=s0.sigma, q=s0.q)

    assert np.isfinite(g.price)
    assert np.isfinite(g.delta)
    assert np.isfinite(g.gamma)
    assert np.isfinite(g.vega)


def test_ad_vs_fd_consistency_delta_and_vega() -> None:
    samples = generate_training_data_stratified(n_samples=220, seed=14, binomial_steps=40)
    cfg = TrainConfig(lr=1e-2, epochs=16, batch_size=24, seed=0)
    model, normalizer, _ = train_surrogate(samples, cfg)

    s0 = samples[10]
    S = _K * math.exp(s0.log_moneyness)

    ad = surrogate_ad_greeks(model, normalizer, S=S, K=_K, T=s0.T, r=s0.r, sigma=s0.sigma, q=s0.q)
    fd = surrogate_fd_greeks(model, normalizer, S=S, K=_K, T=s0.T, r=s0.r, sigma=s0.sigma, q=s0.q)

    assert abs(ad.delta - fd.delta) < 0.2
    assert abs(ad.vega - fd.vega) < 0.5
