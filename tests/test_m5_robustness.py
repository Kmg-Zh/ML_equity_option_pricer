from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from surrogate.data_gen import (
        generate_training_data,
        generate_training_data_stratified,
        maturity_bucket_from_T,
        moneyness_bucket_from_log_moneyness,
    )
    from surrogate.train import TrainConfig, train_surrogate
except ModuleNotFoundError:
    from src.surrogate.data_gen import (
        generate_training_data,
        generate_training_data_stratified,
        maturity_bucket_from_T,
        moneyness_bucket_from_log_moneyness,
    )
    from src.surrogate.train import TrainConfig, train_surrogate


def _bucket_key(sample) -> tuple[str, str]:
    return (
        moneyness_bucket_from_log_moneyness(sample.log_moneyness),
        maturity_bucket_from_T(sample.T),
    )


def test_stratified_generator_covers_all_9_buckets() -> None:
    samples = generate_training_data_stratified(n_samples=180, seed=3, binomial_steps=40)
    counts = Counter(_bucket_key(s) for s in samples)

    assert len(counts) == 9
    for c in counts.values():
        assert c >= 20


def test_weighted_training_executes_and_has_history() -> None:
    samples = generate_training_data_stratified(n_samples=180, seed=5, binomial_steps=40)
    config = TrainConfig(
        lr=1e-2,
        epochs=15,
        batch_size=24,
        seed=0,
        use_bucket_weighting=True,
        short_maturity_weight=2.2,
        wing_moneyness_weight=1.8,
    )
    _, _, history = train_surrogate(samples, config)

    assert len(history.train_loss_by_epoch) == config.epochs
    assert len(history.val_loss_by_epoch) == config.epochs
    assert history.train_loss_by_epoch[-1] < history.train_loss_by_epoch[0]


def test_stratified_vs_uniform_short_bucket_count() -> None:
    uniform = generate_training_data(n_samples=180, seed=7, binomial_steps=40)
    strat = generate_training_data_stratified(n_samples=180, seed=7, binomial_steps=40)

    short_uniform = sum(1 for s in uniform if maturity_bucket_from_T(s.T) == "short")
    short_strat = sum(1 for s in strat if maturity_bucket_from_T(s.T) == "short")

    assert short_strat > short_uniform
