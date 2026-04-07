"""Training loop for the M4 MLP surrogate.

Optimiser: Adam (Kingma & Ba, 2014)
-------------------------------------
Adam maintains per-parameter running estimates of the first moment (mean
gradient) and second moment (uncentred variance):

    m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
    v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²

Bias-corrected estimates are used for the update:

    m̂_t = m_t / (1 − β₁ᵗ)
    v̂_t = v_t / (1 − β₂ᵗ)

    θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)

Default hyper-parameters (α=1e-3, β₁=0.9, β₂=0.999, ε=1e-8) follow the
paper's recommendations and are widely validated for MLP regression tasks.

L2 regularisation (weight_decay) is applied by adding λ·W to the gradient
of each weight matrix before the Adam update, equivalent to a Gaussian weight
prior in the Bayesian interpretation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .data_gen import SurrogateTrainSample
from .data_gen import maturity_bucket_from_T, moneyness_bucket_from_log_moneyness
from .mlp import AmericanPriceMLP
from .normalizer import FeatureNormalizer


@dataclass
class TrainConfig:
    """Hyper-parameters for the training loop."""
    lr:            float = 1e-3
    epochs:        int   = 200
    batch_size:    int   = 64
    weight_decay:  float = 1e-5   # L2 coefficient
    val_fraction:  float = 0.15
    beta1:         float = 0.9
    beta2:         float = 0.999
    adam_eps:      float = 1e-8
    seed:          int   = 42
    use_bucket_weighting: bool = False
    short_maturity_weight: float = 2.0
    wing_moneyness_weight: float = 1.8
    use_financial_constraints: bool = False
    monotonicity_penalty_lambda: float = 0.1
    convexity_penalty_lambda: float = 0.1
    constraint_eps_lm: float = 0.02


@dataclass
class TrainHistory:
    """Per-epoch loss trajectory (MSE on standardised labels)."""
    train_loss_by_epoch: list[float] = field(default_factory=list)
    val_loss_by_epoch:   list[float] = field(default_factory=list)
    mono_penalty_by_epoch: list[float] = field(default_factory=list)
    convex_penalty_by_epoch: list[float] = field(default_factory=list)


def _sample_weight(sample: SurrogateTrainSample, config: TrainConfig) -> float:
    """Return per-sample M5 robustness weight based on bucket membership."""
    if not config.use_bucket_weighting:
        return 1.0

    m_bucket = moneyness_bucket_from_log_moneyness(sample.log_moneyness)
    t_bucket = maturity_bucket_from_T(sample.T)

    w = 1.0
    if t_bucket == "short":
        w *= config.short_maturity_weight
    if m_bucket in ("OTM", "ITM"):
        w *= config.wing_moneyness_weight
    return w


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _train_val_split(
    samples: list[SurrogateTrainSample],
    val_frac: float,
    seed: int,
) -> tuple[list[SurrogateTrainSample], list[SurrogateTrainSample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


class _AdamState:
    """Lightweight Adam state container (moment vectors per parameter)."""

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0  # global step counter

    def step(
        self,
        params: dict[str, np.ndarray],
        grads:  dict[str, np.ndarray],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        self.t += 1
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t
        for k in params:
            g = grads[k]
            if k.startswith("W"):
                # L2 regularisation applied only to weight matrices, not biases
                g = g + weight_decay * params[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * g * g
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_surrogate(
    samples: list[SurrogateTrainSample],
    config: TrainConfig | None = None,
) -> tuple[AmericanPriceMLP, FeatureNormalizer, TrainHistory]:
    """Train an MLP surrogate on Binomial-labelled samples.

    Parameters
    ----------
    samples:  Labelled training samples from :func:`~data_gen.generate_training_data`.
    config:   Hyper-parameter bundle; defaults to :class:`TrainConfig`.

    Returns
    -------
    model:       Trained :class:`~mlp.AmericanPriceMLP` (ready for inference).
    normalizer:  Fitted :class:`~normalizer.FeatureNormalizer` (required at inference).
    history:     Per-epoch MSE trajectory.
    """
    if config is None:
        config = TrainConfig()

    np.random.seed(config.seed)

    train_samples, val_samples = _train_val_split(
        samples, config.val_fraction, config.seed
    )

    normalizer = FeatureNormalizer.fit(train_samples)

    X_train = normalizer.transform_features(train_samples)
    y_train = normalizer.transform_labels(train_samples).reshape(-1, 1)
    X_val   = normalizer.transform_features(val_samples)
    y_val   = normalizer.transform_labels(val_samples).reshape(-1, 1)
    w_train = np.array([_sample_weight(s, config) for s in train_samples], dtype=np.float64).reshape(-1, 1)

    model = AmericanPriceMLP(seed=config.seed)
    adam  = _AdamState(model.params)
    history = TrainHistory()

    rng = np.random.default_rng(config.seed)
    n_train = X_train.shape[0]
    eps_z = config.constraint_eps_lm / float(normalizer.feature_stds[0])

    for _epoch in range(config.epochs):
        # --- training pass (mini-batch SGD with Adam) ---
        idx = rng.permutation(n_train)
        epoch_loss = 0.0
        epoch_mono_penalty = 0.0
        epoch_convex_penalty = 0.0
        n_batches  = 0

        for start in range(0, n_train, config.batch_size):
            batch_idx = idx[start: start + config.batch_size]
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            wb = w_train[batch_idx]

            pred = model.forward(Xb, training=True)    # (batch, 1)
            residual = pred - yb                           # (batch, 1)
            weighted_sq = wb * (residual ** 2)
            loss = float(weighted_sq.mean())
            dL_dy = 2.0 * wb * residual / Xb.shape[0]

            model.zero_grad()
            model.backward(dL_dy)

            mono_penalty = 0.0
            convex_penalty = 0.0
            if config.use_financial_constraints:
                X_minus = Xb.copy()
                X_plus = Xb.copy()
                X_minus[:, 0] -= eps_z
                X_plus[:, 0] += eps_z

                # Monotonicity in moneyness for put: price should be non-increasing
                # as log-moneyness increases (i.e., as S rises).
                if config.monotonicity_penalty_lambda > 0.0:
                    y_minus = model.forward(X_minus, training=True)
                    y_plus = model.forward(X_plus, training=True)
                    mono_violation = np.maximum(y_plus - y_minus, 0.0)
                    mono_penalty = float((mono_violation ** 2).mean())
                    d_mono = 2.0 * mono_violation / Xb.shape[0]

                    model.backward(config.monotonicity_penalty_lambda * d_mono)
                    _ = model.forward(X_minus, training=True)
                    model.backward(config.monotonicity_penalty_lambda * (-d_mono))

                # Convexity in moneyness proxy: second finite difference >= 0.
                if config.convexity_penalty_lambda > 0.0:
                    y_minus = model.forward(X_minus, training=True)
                    y_mid = model.forward(Xb, training=True)
                    y_plus = model.forward(X_plus, training=True)

                    second_diff = y_plus - 2.0 * y_mid + y_minus
                    convex_violation = np.maximum(-second_diff, 0.0)
                    convex_penalty = float((convex_violation ** 2).mean())
                    d_conv_d2 = -2.0 * convex_violation / Xb.shape[0]

                    model.backward(config.convexity_penalty_lambda * d_conv_d2)
                    _ = model.forward(Xb, training=True)
                    model.backward(config.convexity_penalty_lambda * (-2.0 * d_conv_d2))
                    _ = model.forward(X_minus, training=True)
                    model.backward(config.convexity_penalty_lambda * d_conv_d2)

            adam.step(
                model.params, model.grads,
                lr=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
            total_loss = (
                loss
                + config.monotonicity_penalty_lambda * mono_penalty
                + config.convexity_penalty_lambda * convex_penalty
            )
            epoch_loss += total_loss
            epoch_mono_penalty += mono_penalty
            epoch_convex_penalty += convex_penalty
            n_batches  += 1

        history.train_loss_by_epoch.append(epoch_loss / max(n_batches, 1))
        history.mono_penalty_by_epoch.append(epoch_mono_penalty / max(n_batches, 1))
        history.convex_penalty_by_epoch.append(epoch_convex_penalty / max(n_batches, 1))

        # --- validation pass (no gradient) ---
        val_pred = model.predict(X_val)
        val_loss = float(((val_pred - y_val) ** 2).mean())
        history.val_loss_by_epoch.append(val_loss)

    return model, normalizer, history
