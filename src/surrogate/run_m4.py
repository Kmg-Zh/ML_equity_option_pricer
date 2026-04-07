"""M4 training entry-point.

Generates training + evaluation data, trains the MLP surrogate, and
writes two artifacts under ``artifacts/surrogate/``:

    m4_mlp_weights.npz  — model parameter arrays (NumPy .npz archive)
    m4_eval_report.json — bucketed MAE/RMSE evaluation vs Binomial teacher

Run from the repo root:
    .venv/bin/python -m src.surrogate.run_m4
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data_gen import generate_training_data
from .evaluate import evaluate_surrogate
from .train import TrainConfig, train_surrogate


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir   = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating training data (800 samples, Binomial teacher, 120 steps)…")
    train_samples = generate_training_data(n_samples=800, seed=0, binomial_steps=120)
    print(f"  {len(train_samples)} samples generated.")

    print("Generating held-out evaluation data (200 samples, seed=99)…")
    eval_samples = generate_training_data(n_samples=200, seed=99, binomial_steps=120)
    print(f"  {len(eval_samples)} samples generated.")

    print("Training MLP surrogate (120 epochs, Adam lr=1e-3)…")
    config = TrainConfig(lr=1e-3, epochs=120, batch_size=64, seed=42)
    model, normalizer, history = train_surrogate(train_samples, config)
    print(f"  Final train MSE: {history.train_loss_by_epoch[-1]:.6f}")
    print(f"  Final val   MSE: {history.val_loss_by_epoch[-1]:.6f}")

    print("Evaluating on held-out set…")
    results = evaluate_surrogate(model, normalizer, eval_samples)

    # --- persist model weights ---
    weights_path = out_dir / "m4_mlp_weights.npz"
    np.savez(weights_path, **model.params)
    print(f"  Weights saved → {weights_path}")

    # --- persist normalizer stats (for inference) ---
    norm_path = out_dir / "m4_normalizer.npz"
    np.savez(
        norm_path,
        feature_means=normalizer.feature_means,
        feature_stds=normalizer.feature_stds,
        label_mean=np.array([normalizer.label_mean]),
        label_std=np.array([normalizer.label_std]),
    )
    print(f"  Normalizer saved → {norm_path}")

    # --- persist evaluation report ---
    payload = {
        "model": "AmericanPriceMLP",
        "architecture": "5→64→128→64→1 (ReLU, Adam, L2=1e-5)",
        "teacher": "Binomial-CRR (120 steps)",
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "final_train_mse": history.train_loss_by_epoch[-1],
        "final_val_mse":   history.val_loss_by_epoch[-1],
        "bucket_eval": [asdict(r) for r in results],
    }
    report_path = out_dir / "m4_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Report saved → {report_path}")

    # --- print summary table ---
    print()
    print("Bucket MAE summary (dollar units, vs Binomial teacher):")
    print(f"{'Moneyness':<10} {'Maturity':<10} {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'MaxAE':>8}")
    print("-" * 58)
    for r in results:
        print(
            f"{r.moneyness_bucket:<10} {r.maturity_bucket:<10} {r.count:>5}"
            f"  {r.mae:>8.4f}  {r.rmse:>8.4f}  {r.max_ae:>8.4f}"
        )


if __name__ == "__main__":
    main()
