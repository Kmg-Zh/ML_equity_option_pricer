from __future__ import annotations

try:
    from surrogate.data_gen import generate_training_data, generate_training_data_stratified
    from surrogate.evaluate import evaluate_financial_constraints
    from surrogate.train import TrainConfig, train_surrogate
except ModuleNotFoundError:
    from src.surrogate.data_gen import generate_training_data, generate_training_data_stratified
    from src.surrogate.evaluate import evaluate_financial_constraints
    from src.surrogate.train import TrainConfig, train_surrogate


def test_financial_constraint_metrics_bounds() -> None:
    samples = generate_training_data_stratified(n_samples=180, seed=1, binomial_steps=40)
    cfg = TrainConfig(lr=1e-2, epochs=12, batch_size=24, seed=0)
    model, normalizer, _ = train_surrogate(samples, cfg)

    eval_samples = generate_training_data(n_samples=80, seed=2, binomial_steps=40)
    metrics = evaluate_financial_constraints(model, normalizer, eval_samples, eps_lm=0.02)

    assert 0.0 <= metrics.monotonicity_violation_rate <= 1.0
    assert 0.0 <= metrics.convexity_violation_rate <= 1.0
    assert metrics.monotonicity_violation_mean >= 0.0
    assert metrics.convexity_violation_mean >= 0.0


def test_constraint_training_populates_penalty_history() -> None:
    samples = generate_training_data_stratified(n_samples=180, seed=4, binomial_steps=40)
    cfg = TrainConfig(
        lr=1e-2,
        epochs=10,
        batch_size=24,
        seed=0,
        use_financial_constraints=True,
        monotonicity_penalty_lambda=0.15,
        convexity_penalty_lambda=0.10,
        constraint_eps_lm=0.02,
    )
    _, _, history = train_surrogate(samples, cfg)

    assert len(history.mono_penalty_by_epoch) == cfg.epochs
    assert len(history.convex_penalty_by_epoch) == cfg.epochs
    assert all(v >= 0.0 for v in history.mono_penalty_by_epoch)
    assert all(v >= 0.0 for v in history.convex_penalty_by_epoch)
