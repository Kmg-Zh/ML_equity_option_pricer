"""M6 training entry-point: financial constraints (monotonicity + convexity)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data_gen import generate_training_data, generate_training_data_stratified
from .evaluate import evaluate_financial_constraints, evaluate_surrogate
from .train import TrainConfig, train_surrogate


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating M6 training data (stratified, 1000 samples, teacher 120 steps)...")
    train_samples = generate_training_data_stratified(
        n_samples=1000,
        seed=21,
        binomial_steps=120,
        option_type="put",
    )
    print(f"  {len(train_samples)} training samples generated.")

    print("Generating held-out evaluation data (uniform, 320 samples)...")
    eval_samples = generate_training_data(
        n_samples=320,
        seed=121,
        binomial_steps=120,
        option_type="put",
    )
    print(f"  {len(eval_samples)} eval samples generated.")

    config = TrainConfig(
        lr=1e-3,
        epochs=150,
        batch_size=64,
        seed=42,
        use_bucket_weighting=True,
        short_maturity_weight=2.5,
        wing_moneyness_weight=2.0,
        use_financial_constraints=True,
        monotonicity_penalty_lambda=0.20,
        convexity_penalty_lambda=0.15,
        constraint_eps_lm=0.02,
    )

    print("Training M6 model (robust + financial constraints)...")
    model, normalizer, history = train_surrogate(train_samples, config)
    print(f"  Final train objective: {history.train_loss_by_epoch[-1]:.6f}")
    print(f"  Final val MSE:         {history.val_loss_by_epoch[-1]:.6f}")
    print(f"  Final mono penalty:    {history.mono_penalty_by_epoch[-1]:.6f}")
    print(f"  Final convex penalty:  {history.convex_penalty_by_epoch[-1]:.6f}")

    bucket_results = evaluate_surrogate(model, normalizer, eval_samples)
    constraint_metrics = evaluate_financial_constraints(
        model,
        normalizer,
        eval_samples,
        eps_lm=config.constraint_eps_lm,
    )

    weights_path = out_dir / "m6_mlp_weights.npz"
    np.savez(weights_path, **model.params)

    normalizer_path = out_dir / "m6_normalizer.npz"
    np.savez(
        normalizer_path,
        feature_means=normalizer.feature_means,
        feature_stds=normalizer.feature_stds,
        label_mean=np.array([normalizer.label_mean]),
        label_std=np.array([normalizer.label_std]),
    )

    payload = {
        "model": "AmericanPriceMLP",
        "training_mode": "M6 robust + financial constraints",
        "teacher": "Binomial-CRR (120 steps)",
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "config": {
            "use_bucket_weighting": config.use_bucket_weighting,
            "short_maturity_weight": config.short_maturity_weight,
            "wing_moneyness_weight": config.wing_moneyness_weight,
            "use_financial_constraints": config.use_financial_constraints,
            "monotonicity_penalty_lambda": config.monotonicity_penalty_lambda,
            "convexity_penalty_lambda": config.convexity_penalty_lambda,
            "constraint_eps_lm": config.constraint_eps_lm,
        },
        "final_train_objective": history.train_loss_by_epoch[-1],
        "final_val_mse": history.val_loss_by_epoch[-1],
        "final_mono_penalty": history.mono_penalty_by_epoch[-1],
        "final_convex_penalty": history.convex_penalty_by_epoch[-1],
        "constraint_metrics": asdict(constraint_metrics),
        "bucket_eval": [asdict(r) for r in bucket_results],
    }

    report_path = out_dir / "m6_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  Weights saved -> {weights_path}")
    print(f"  Normalizer saved -> {normalizer_path}")
    print(f"  Report saved -> {report_path}")


if __name__ == "__main__":
    main()
