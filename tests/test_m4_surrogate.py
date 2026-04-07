"""Tests for M4 MLP surrogate (data generation, normalizer, MLP, training, evaluation).

All tests are fast smoke-tests (< 5 s total):
  - data_gen: 30 samples, binomial_steps=50
  - train: 60 samples, 20 epochs
No full convergence check here — accuracy is validated by run_m4.py artifact.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

try:
    from surrogate.data_gen import FEATURE_BOUNDS, SurrogateTrainSample, _K, generate_training_data
    from surrogate.evaluate import BucketEvalResult, evaluate_surrogate
    from surrogate.mlp import AmericanPriceMLP
    from surrogate.normalizer import FeatureNormalizer
    from surrogate.train import TrainConfig, TrainHistory, train_surrogate
except ModuleNotFoundError:
    from src.surrogate.data_gen import (
        FEATURE_BOUNDS,
        SurrogateTrainSample,
        _K,
        generate_training_data,
    )
    from src.surrogate.evaluate import BucketEvalResult, evaluate_surrogate
    from src.surrogate.mlp import AmericanPriceMLP
    from src.surrogate.normalizer import FeatureNormalizer
    from src.surrogate.train import TrainConfig, TrainHistory, train_surrogate


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def test_data_gen_shapes_and_financial_validity():
    """Samples must lie within declared bounds and satisfy American put lower bound."""
    samples = generate_training_data(n_samples=30, seed=7, binomial_steps=50)
    assert len(samples) == 30

    for s in samples:
        assert FEATURE_BOUNDS["log_moneyness"][0] <= s.log_moneyness <= FEATURE_BOUNDS["log_moneyness"][1]
        assert FEATURE_BOUNDS["T"][0] <= s.T <= FEATURE_BOUNDS["T"][1]
        assert s.price_normalized >= 0.0, "Price must be non-negative."

        # American put price ≥ intrinsic value (K - S)₊ / K = max(0, 1 - exp(lm))
        intrinsic_norm = max(0.0, 1.0 - math.exp(s.log_moneyness))
        assert s.price_normalized >= intrinsic_norm - 1e-5, (
            f"price/K={s.price_normalized:.5f} < intrinsic/K={intrinsic_norm:.5f} "
            f"(lm={s.log_moneyness:.3f})"
        )


def test_data_gen_reproducibility():
    """Same seed must produce identical samples."""
    s1 = generate_training_data(n_samples=10, seed=42, binomial_steps=50)
    s2 = generate_training_data(n_samples=10, seed=42, binomial_steps=50)
    for a, b in zip(s1, s2):
        assert a == b


# ---------------------------------------------------------------------------
# 2. Normalizer
# ---------------------------------------------------------------------------

def test_normalizer_fit_and_roundtrip():
    """Inverse transform must recover original price_normalized to 1e-10."""
    samples = generate_training_data(n_samples=40, seed=1, binomial_steps=50)
    norm = FeatureNormalizer.fit(samples)

    # Verify that label round-trip is exact
    for s in samples[:6]:
        z = (s.price_normalized - norm.label_mean) / norm.label_std
        recovered = float(norm.inverse_transform_label(z))
        assert abs(recovered - s.price_normalized) < 1e-10

    # Feature matrix shape
    X = norm.transform_features(samples)
    assert X.shape == (40, 5)
    # After normalisation, mean ≈ 0, std ≈ 1 (on training set)
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(X.std(axis=0),  1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. MLP architecture
# ---------------------------------------------------------------------------

def test_mlp_forward_shape_and_no_nan():
    """Forward pass must return (N, 1) with no NaN values."""
    model = AmericanPriceMLP(input_dim=5, hidden_dims=(64, 128, 64), seed=0)
    X = np.random.default_rng(0).standard_normal((16, 5))
    y = model.forward(X, training=False)
    assert y.shape == (16, 1), f"Expected (16, 1), got {y.shape}"
    assert not np.isnan(y).any(), "NaN detected in forward pass."


def test_mlp_backward_fills_grads():
    """Backward pass must write non-zero gradients to all parameter arrays."""
    model = AmericanPriceMLP(seed=0)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 5))
    y = model.forward(X, training=True)
    dL = 2.0 * y / 8          # dummy gradient
    model.backward(dL)
    for name, grad in model.grads.items():
        assert np.any(grad != 0.0), f"Gradient for {name} is all zeros after backward."


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def test_train_reduces_loss():
    """Training loss must decrease from first to last epoch (even on tiny data)."""
    samples = generate_training_data(n_samples=60, seed=5, binomial_steps=50)
    config  = TrainConfig(lr=1e-2, epochs=20, batch_size=16, seed=0)
    _, _, history = train_surrogate(samples, config)

    assert isinstance(history, TrainHistory)
    assert len(history.train_loss_by_epoch) == 20
    first, last = history.train_loss_by_epoch[0], history.train_loss_by_epoch[-1]
    assert last < first, (
        f"Training loss did not decrease: epoch-1={first:.4f}, epoch-20={last:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_returns_non_empty_buckets():
    """evaluate_surrogate must return at least one bucket with valid statistics."""
    samples = generate_training_data(n_samples=80, seed=3, binomial_steps=50)
    config  = TrainConfig(lr=1e-2, epochs=10, batch_size=16, seed=0)
    model, normalizer, _ = train_surrogate(samples, config)

    eval_samples = generate_training_data(n_samples=100, seed=77, binomial_steps=50)
    results = evaluate_surrogate(model, normalizer, eval_samples)

    assert len(results) >= 1, "Expected at least one evaluation bucket."
    for r in results:
        assert isinstance(r, BucketEvalResult)
        assert r.count >= 1
        assert r.mae >= 0.0
        assert r.rmse >= 0.0
        assert r.max_ae >= r.mae
