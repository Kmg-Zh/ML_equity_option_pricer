"""Training-data generator for M4 MLP surrogate.

Sampling strategy
-----------------
Uniform random draws over a 5-dimensional input space:

    log_moneyness = log(S/K) ∈ [-0.5, 0.5]
        Covers OTM/ATM/ITM for typical equity options (~2.5 σ range at 20 % vol).
    T ∈ [0.08, 3.0]   (1 month to 3 years)
    r ∈ [0.00, 0.08]  (zero to 8 % risk-free)
    sigma ∈ [0.10, 0.60]  (low-vol blue chip to high-vol growth/event)
    q ∈ [0.00, 0.06]  (no dividend to 6 % yield)

Strike K = 100 is fixed; S = K · exp(log_moneyness).  The target label is
price / K (scale-normalised by degree-1 homogeneity of vanilla option prices in
(S, K)), which reduces the effective dimensionality of the learning problem.

Only ``option_type = "put"`` is generated in M4.  American puts are the canonical
hard case: the exercise boundary is non-trivial and no closed-form solution
exists.  Calls can be added trivially in M5.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

try:
    from pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price
except ModuleNotFoundError:
    from src.pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price


@dataclass(frozen=True)
class SurrogateTrainSample:
    """One labeled training point.

    All fields are *raw* (pre-normalisation) except ``price_normalized``.

    Attributes
    ----------
    log_moneyness:   log(S/K)
    T:               time to maturity (years)
    r:               risk-free rate
    sigma:           annualised volatility
    q:               continuous dividend yield
    price_normalized: price / K   (scale-invariant target)
    """

    log_moneyness: float
    T: float
    r: float
    sigma: float
    q: float
    price_normalized: float  # price / K


# Input space bounds (all theory-grounded, see module docstring).
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "log_moneyness": (-0.5, 0.5),
    "T":             (0.08, 3.0),
    "r":             (0.00, 0.08),
    "sigma":         (0.10, 0.60),
    "q":             (0.00, 0.06),
}

_K = 100.0  # Reference strike; S = K * exp(log_moneyness)

_LOG_MONEYNESS_SPLITS = (math.log(0.95), math.log(1.05))


def moneyness_bucket_from_log_moneyness(log_m: float) -> str:
    """Map log-moneyness to OTM/ATM/ITM bucket labels."""
    m = math.exp(log_m)
    if m < 0.95:
        return "OTM"
    if m <= 1.05:
        return "ATM"
    return "ITM"


def maturity_bucket_from_T(T: float) -> str:
    """Map maturity in years to short/medium/long bucket labels."""
    if T <= 0.25:
        return "short"
    if T <= 1.0:
        return "medium"
    return "long"


def _bucket_ranges() -> dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]:
    """Return per-bucket sampling ranges for (log_moneyness, T).

    Buckets follow M3/M4 conventions and partition the M4 domain.
    """
    lm_min, lm_max = FEATURE_BOUNDS["log_moneyness"]
    t_min, t_max = FEATURE_BOUNDS["T"]
    lm_atm_l, lm_atm_r = _LOG_MONEYNESS_SPLITS

    return {
        ("OTM", "short"): ((lm_min, lm_atm_l), (t_min, 0.25)),
        ("OTM", "medium"): ((lm_min, lm_atm_l), (0.25, 1.0)),
        ("OTM", "long"): ((lm_min, lm_atm_l), (1.0, t_max)),
        ("ATM", "short"): ((lm_atm_l, lm_atm_r), (t_min, 0.25)),
        ("ATM", "medium"): ((lm_atm_l, lm_atm_r), (0.25, 1.0)),
        ("ATM", "long"): ((lm_atm_l, lm_atm_r), (1.0, t_max)),
        ("ITM", "short"): ((lm_atm_r, lm_max), (t_min, 0.25)),
        ("ITM", "medium"): ((lm_atm_r, lm_max), (0.25, 1.0)),
        ("ITM", "long"): ((lm_atm_r, lm_max), (1.0, t_max)),
    }


def _sample_one(
    rng: random.Random,
    log_moneyness_range: tuple[float, float] | None,
    maturity_range: tuple[float, float] | None,
    binomial_steps: int,
    option_type: str,
) -> SurrogateTrainSample:
    """Sample one labeled point; optional range constraints for M5 buckets."""
    if log_moneyness_range is None:
        lm = rng.uniform(*FEATURE_BOUNDS["log_moneyness"])
    else:
        lm = rng.uniform(*log_moneyness_range)

    if maturity_range is None:
        T = rng.uniform(*FEATURE_BOUNDS["T"])
    else:
        T = rng.uniform(*maturity_range)

    r = rng.uniform(*FEATURE_BOUNDS["r"])
    sigma = rng.uniform(*FEATURE_BOUNDS["sigma"])
    q = rng.uniform(*FEATURE_BOUNDS["q"])

    S = _K * math.exp(lm)
    opt = AmericanOptionInput(
        S=S,
        K=_K,
        T=T,
        r=r,
        sigma=sigma,
        dividend_model=FlatContinuousDividendYield(rate=q),
        option_type=option_type,
    )
    price = american_binomial_price(opt, steps=binomial_steps)
    return SurrogateTrainSample(
        log_moneyness=lm,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        price_normalized=price / _K,
    )


def generate_training_data(
    n_samples: int = 2000,
    seed: int = 0,
    binomial_steps: int = 200,
    option_type: str = "put",
) -> list[SurrogateTrainSample]:
    """Generate labeled training data using the Binomial (CRR) teacher.

    Parameters
    ----------
    n_samples:       Number of Monte Carlo random draws.
    seed:            RNG seed for reproducibility.
    binomial_steps:  CRR tree depth — higher = more accurate labels, slower.
    option_type:     ``"put"`` (default) or ``"call"``.

    Returns
    -------
    List of ``SurrogateTrainSample`` with Binomial-priced labels.
    """
    rng = random.Random(seed)
    samples: list[SurrogateTrainSample] = []

    for _ in range(n_samples):
        samples.append(
            _sample_one(
                rng=rng,
                log_moneyness_range=None,
                maturity_range=None,
                binomial_steps=binomial_steps,
                option_type=option_type,
            )
        )

    return samples


def generate_training_data_stratified(
    n_samples: int = 1800,
    seed: int = 0,
    binomial_steps: int = 120,
    option_type: str = "put",
) -> list[SurrogateTrainSample]:
    """Generate M5 stratified samples with explicit bucket coverage.

    The dataset is balanced across 9 buckets = 3 moneyness x 3 maturity.
    Remainder samples are distributed deterministically from sorted bucket order.
    """
    rng = random.Random(seed)
    ranges = _bucket_ranges()
    bucket_keys = sorted(ranges.keys())
    n_buckets = len(bucket_keys)
    base = n_samples // n_buckets
    rem = n_samples % n_buckets

    samples: list[SurrogateTrainSample] = []
    for i, bucket_key in enumerate(bucket_keys):
        lm_range, t_range = ranges[bucket_key]
        n_this = base + (1 if i < rem else 0)
        for _ in range(n_this):
            samples.append(
                _sample_one(
                    rng=rng,
                    log_moneyness_range=lm_range,
                    maturity_range=t_range,
                    binomial_steps=binomial_steps,
                    option_type=option_type,
                )
            )

    return samples
