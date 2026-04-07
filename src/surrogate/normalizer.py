"""Z-score normalizer for MLP surrogate features and labels.

Theory
------
Standardisation maps each feature to approximately zero mean and unit
variance, which ensures the gradient signals are similarly scaled across
dimensions and prevents one feature (e.g. T ∈ [0.08, 3.0]) from dominating
another (e.g. r ∈ [0.0, 0.08]) purely due to magnitude differences.

The label (price/K) is also standardised so the MSE loss operates in a
well-conditioned space.  Predictions are un-standardised after the forward
pass when reporting prices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data_gen import SurrogateTrainSample


@dataclass
class FeatureNormalizer:
    """Fitted z-score normaliser for the 5-feature / 1-label problem.

    Attributes
    ----------
    feature_means:  Per-feature training mean  (length-5 array)
    feature_stds:   Per-feature training std   (length-5 array)
    label_mean:     Training mean of price/K
    label_std:      Training std  of price/K
    eps:            Minimum std to prevent division by zero
    """

    feature_means: np.ndarray = field(default_factory=lambda: np.zeros(5))
    feature_stds:  np.ndarray = field(default_factory=lambda: np.ones(5))
    label_mean:    float = 0.0
    label_std:     float = 1.0
    eps:           float = 1e-8

    @classmethod
    def fit(cls, samples: list[SurrogateTrainSample]) -> "FeatureNormalizer":
        """Compute mean/std from the training split."""
        if not samples:
            raise ValueError("Cannot fit on empty sample list.")
        X = _to_feature_array(samples)
        y = np.array([s.price_normalized for s in samples], dtype=np.float64)
        return cls(
            feature_means=X.mean(axis=0),
            feature_stds=np.maximum(X.std(axis=0), 1e-8),
            label_mean=float(y.mean()),
            label_std=float(max(y.std(), 1e-8)),
        )

    def transform_features(self, samples: list[SurrogateTrainSample]) -> np.ndarray:
        """Return (N, 5) float64 normalised feature matrix."""
        X = _to_feature_array(samples)
        return (X - self.feature_means) / self.feature_stds

    def transform_labels(self, samples: list[SurrogateTrainSample]) -> np.ndarray:
        """Return (N,) float64 normalised label vector."""
        y = np.array([s.price_normalized for s in samples], dtype=np.float64)
        return (y - self.label_mean) / self.label_std

    def inverse_transform_label(self, z: float | np.ndarray) -> float | np.ndarray:
        """Map standardised label(s) back to price/K space."""
        return z * self.label_std + self.label_mean


def _to_feature_array(samples: list[SurrogateTrainSample]) -> np.ndarray:
    """Convert samples to a (N, 5) float64 matrix [lm, T, r, sigma, q]."""
    return np.array(
        [[s.log_moneyness, s.T, s.r, s.sigma, s.q] for s in samples],
        dtype=np.float64,
    )
