"""M5 training entry-point: tail robustness via stratified sampling + weighted loss."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data_gen import generate_training_data, generate_training_data_stratified
from .evaluate import evaluate_surrogate
from .train import TrainConfig, train_surrogate


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating stratified M5 training data (900 samples, Binomial 120 steps)...")
    train_samples = generate_training_data_stratified(
        n_samples=900,
        seed=11,
        binomial_steps=120,
        option_type="put",
    )
    print(f"  {len(train_samples)} training samples generated.")

    print("Generating held-out evaluation data (300 uniform samples, seed=101)...")
    eval_samples = generate_training_data(
        n_samples=300,
        seed=101,
        binomial_steps=120,
        option_type="put",
    )
    print(f"  {len(eval_samples)} eval samples generated.")

    config = TrainConfig(
        lr=1e-3,
        epochs=140,
        batch_size=64,
        seed=42,
        use_bucket_weighting=True,
        short_maturity_weight=2.5,
        wing_moneyness_weight=2.0,
    )
    print("Training M5 model (weighted loss enabled)...")
    model, normalizer, history = train_surrogate(train_samples, config)
    print(f"  Final train weighted-MSE: {history.train_loss_by_epoch[-1]:.6f}")
    print(f"  Final val weighted-MSE:   {history.val_loss_by_epoch[-1]:.6f}")

    results = evaluate_surrogate(model, normalizer, eval_samples)

    weights_path = out_dir / "m5_mlp_weights.npz"
    np.savez(weights_path, **model.params)

    normalizer_path = out_dir / "m5_normalizer.npz"
    np.savez(
        normalizer_path,
        feature_means=normalizer.feature_means,
        feature_stds=normalizer.feature_stds,
        label_mean=np.array([normalizer.label_mean]),
        label_std=np.array([normalizer.label_std]),
    )

    payload = {
        "model": "AmericanPriceMLP",
        "training_mode": "M5 stratified + weighted loss",
        "teacher": "Binomial-CRR (120 steps)",
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "final_train_weighted_mse": history.train_loss_by_epoch[-1],
        "final_val_weighted_mse": history.val_loss_by_epoch[-1],
        "weights": {
            "short_maturity_weight": config.short_maturity_weight,
            "wing_moneyness_weight": config.wing_moneyness_weight,
        },
        "bucket_eval": [asdict(r) for r in results],
    }

    report_path = out_dir / "m5_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  Weights saved -> {weights_path}")
    print(f"  Normalizer saved -> {normalizer_path}")
    print(f"  Report saved -> {report_path}")


if __name__ == "__main__":
    main()
