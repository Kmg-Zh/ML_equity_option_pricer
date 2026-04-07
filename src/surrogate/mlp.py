"""Numpy-native MLP for American option price surrogate (M4).

Architecture theory
-------------------
By the universal approximation theorem (Hornik et al., 1989), a two-layer
MLP with a non-polynomial activation and enough hidden units can represent
any continuous function on a compact domain.  We use three hidden layers to
give the model capacity for the non-smooth features of the American option
price function (early-exercise kink near the optimal boundary).

Layer layout (M4 base)
~~~~~~~~~~~~~~~~~~~~~~
    Input  (5)  → Linear(5→64) → ReLU
                → Linear(64→128) → ReLU
                → Linear(128→64) → ReLU
                → Linear(64→1)

Output is *unbounded*.  Non-negativity and monotonicity enforcement are
deferred to M6 (financial constraint layer).

Weight initialisation
~~~~~~~~~~~~~~~~~~~~~
Kaiming / He uniform initialisation is used for layers followed by ReLU:
    std = sqrt(2 / fan_in)
This preserves activation variance through deep ReLU networks and is the
standard choice for ReLU MLPs since He et al. (2015).

The final linear layer uses Xavier uniform (fan_in + fan_out denominator)
since it has no ReLU following it.

Forward / backward pass
~~~~~~~~~~~~~~~~~~~~~~~
All operations are numpy matrix products.  Gradients are accumulated
through standard backpropagation (chain rule) and stored in ``self.grads``,
a dict mirroring ``self.params``.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class LayerCache(NamedTuple):
    """Cached activations needed for the backward pass of one linear+ReLU layer."""
    z: np.ndarray   # pre-activation  (batch, out_features)
    a: np.ndarray   # post-activation (batch, out_features) — same as z for last layer
    a_in: np.ndarray  # input to this layer (batch, in_features)


class AmericanPriceMLP:
    """Feedforward MLP with manual numpy forward/backward pass.

    Parameters
    ----------
    input_dim:    Number of input features (default 5).
    hidden_dims:  Tuple of hidden layer widths.
    seed:         RNG seed for weight initialisation.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: tuple[int, ...] = (64, 128, 64),
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_dims, 1]

        self.params: dict[str, np.ndarray] = {}
        self.grads:  dict[str, np.ndarray] = {}

        for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = (i == len(dims) - 2)
            if is_last:
                # Xavier uniform: no ReLU follows
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                W = rng.uniform(-bound, bound, (fan_in, fan_out))
            else:
                # Kaiming / He uniform: ReLU follows
                bound = math.sqrt(2.0 / fan_in)
                W = rng.uniform(-bound, bound, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.params[f"W{i}"] = W
            self.params[f"b{i}"] = b
            self.grads[f"W{i}"]  = np.zeros_like(W)
            self.grads[f"b{i}"]  = np.zeros_like(b)

        self._n_layers = len(dims) - 1
        self._cache: list[LayerCache] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute predictions.

        Parameters
        ----------
        X:        (batch, input_dim) float64 array.
        training: If True, cache activations for the subsequent backward pass.

        Returns
        -------
        (batch, 1) prediction array.
        """
        cache: list[LayerCache] = []
        a = X
        for i in range(self._n_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            a_in = a
            z = a_in @ W + b           # (batch, out)
            is_last = (i == self._n_layers - 1)
            a = z if is_last else np.maximum(0.0, z)   # ReLU except at output
            if training:
                cache.append(LayerCache(z=z, a=a, a_in=a_in))
        if training:
            self._cache = cache
        return a  # (batch, 1)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dL_dy: np.ndarray) -> None:
        """Backpropagate mean-squared-error gradient through all layers.

        After calling this, ``self.grads`` holds the mean gradient over the
        batch.

        Parameters
        ----------
        dL_dy:  (batch, 1) gradient of loss w.r.t. network output.
        """
        batch = dL_dy.shape[0]
        delta = dL_dy  # gradient of loss w.r.t. layer output

        for i in reversed(range(self._n_layers)):
            lc = self._cache[i]
            is_last = (i == self._n_layers - 1)

            if not is_last:
                # ReLU backward: zero out gradient where pre-activation ≤ 0
                delta = delta * (lc.z > 0).astype(np.float64)

            # Accumulate so multiple loss terms (M6 constraints) can backprop
            # within one optimizer step.
            self.grads[f"W{i}"] += lc.a_in.T @ delta / batch
            self.grads[f"b{i}"] += delta.mean(axis=0)

            # Propagate gradient to previous layer's output
            W = self.params[f"W{i}"]
            delta = delta @ W.T

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        for k in self.grads:
            self.grads[k][:] = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference-mode forward pass (no cache). Returns (N, 1)."""
        return self.forward(X, training=False)

    def predict_and_input_grads(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions and AD gradients wrt normalized inputs.

        Parameters
        ----------
        X: (N, input_dim) normalized feature matrix.

        Returns
        -------
        y: (N, 1) predictions.
        dy_dX: (N, input_dim) jacobian rows where each row is
               d y_i / d X_i (single-output model).
        """
        # Forward pass with local caches (inference-safe).
        a = X
        z_list: list[np.ndarray] = []
        for i in range(self._n_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            z = a @ W + b
            z_list.append(z)
            is_last = (i == self._n_layers - 1)
            a = z if is_last else np.maximum(0.0, z)
        y = a

        n, input_dim = X.shape
        dy_dX = np.zeros((n, input_dim), dtype=np.float64)

        # Row-wise Jacobian for each sample (output dim = 1).
        for row in range(n):
            g = np.array([1.0], dtype=np.float64)  # d y / d a_last
            for i in reversed(range(self._n_layers)):
                is_last = (i == self._n_layers - 1)
                if not is_last:
                    mask = (z_list[i][row] > 0.0).astype(np.float64)
                    g = g * mask
                W = self.params[f"W{i}"]
                g = g @ W.T
            dy_dX[row] = g

        return y, dy_dX
