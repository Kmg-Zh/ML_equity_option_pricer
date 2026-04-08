"""M9: Speed-accuracy frontier for pricing and Greeks.

This module compares runtime and accuracy of:
    1) Binomial teacher pricing
    2) Surrogate MLP pricing
    3) Teacher FD Greeks (reference)
    4) Surrogate AD Greeks
    5) Surrogate FD Greeks

Outputs
-------
Writes ``artifacts/surrogate/m9_speed_accuracy.json`` with:
- latency distributions (mean/p50/p95/p99)
- throughput estimates (samples/sec)
- speedup ratios versus teacher references
- accuracy metrics versus teacher labels (price) and teacher FD (Greeks)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price
except ModuleNotFoundError:
    from src.pricing import AmericanOptionInput, FlatContinuousDividendYield, american_binomial_price

from .data_gen import _K, SurrogateTrainSample, generate_training_data_stratified
from .greeks import load_surrogate_artifacts, surrogate_ad_greeks, surrogate_fd_greeks
from .normalizer import FeatureNormalizer
from .mlp import AmericanPriceMLP
from .run_m8 import teacher_fd_greeks


_TEACHER_STEPS = 100
_PRICE_BATCHES = [1, 8, 32, 128, 256]
_GREEKS_BATCHES = [1, 4, 8, 16]


def _teacher_price(sample: SurrogateTrainSample, K: float = _K, steps: int = _TEACHER_STEPS) -> float:
    S = K * math.exp(sample.log_moneyness)
    opt = AmericanOptionInput(
        S=float(S),
        K=K,
        T=sample.T,
        r=sample.r,
        sigma=sample.sigma,
        option_type="put",
        dividend_model=FlatContinuousDividendYield(rate=sample.q),
    )
    return float(american_binomial_price(opt, steps=steps))


def _latency_summary(seconds: list[float]) -> dict[str, float]:
    arr = np.array(seconds, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.quantile(arr, 0.50) * 1000.0),
        "p95_ms": float(np.quantile(arr, 0.95) * 1000.0),
        "p99_ms": float(np.quantile(arr, 0.99) * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
    }


def _abs_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _measure_latency(task: Callable[[], Any], repeats: int, warmup: int = 1) -> list[float]:
    for _ in range(warmup):
        task()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        task()
        times.append(time.perf_counter() - t0)
    return times


def _batch(samples: list[SurrogateTrainSample], size: int) -> list[SurrogateTrainSample]:
    if size <= len(samples):
        return samples[:size]
    # Deterministic wrap-around for larger requested batch sizes.
    out: list[SurrogateTrainSample] = []
    while len(out) < size:
        need = size - len(out)
        out.extend(samples[:need])
    return out


def _price_accuracy(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    eval_samples: list[SurrogateTrainSample],
    K: float = _K,
) -> dict[str, float]:
    X = normalizer.transform_features(eval_samples)
    z = model.predict(X).ravel()
    pred = normalizer.inverse_transform_label(z) * K
    true = np.array([s.price_normalized for s in eval_samples], dtype=np.float64) * K

    abs_err = np.abs(pred - true)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "max_ae": float(abs_err.max()),
    }


def _greeks_accuracy(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    eval_samples: list[SurrogateTrainSample],
    K: float = _K,
) -> dict[str, Any]:
    ad_abs: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}
    fd_abs: dict[str, list[float]] = {g: [] for g in ("delta", "gamma", "vega")}

    for sample in eval_samples:
        S = float(K * math.exp(sample.log_moneyness))
        tfd = teacher_fd_greeks(S, K, sample.T, sample.r, sample.sigma, sample.q)
        ad = surrogate_ad_greeks(model, normalizer, S, K, sample.T, sample.r, sample.sigma, sample.q)
        sfd = surrogate_fd_greeks(model, normalizer, S, K, sample.T, sample.r, sample.sigma, sample.q)

        ad_abs["delta"].append(abs(ad.delta - tfd["delta"]))
        ad_abs["gamma"].append(abs(ad.gamma - tfd["gamma"]))
        ad_abs["vega"].append(abs(ad.vega - tfd["vega"]))

        fd_abs["delta"].append(abs(sfd.delta - tfd["delta"]))
        fd_abs["gamma"].append(abs(sfd.gamma - tfd["gamma"]))
        fd_abs["vega"].append(abs(sfd.vega - tfd["vega"]))

    return {
        "ad_vs_teacher_fd_abs_error": {g: _abs_summary(ad_abs[g]) for g in ("delta", "gamma", "vega")},
        "fd_vs_teacher_fd_abs_error": {g: _abs_summary(fd_abs[g]) for g in ("delta", "gamma", "vega")},
    }


def _benchmark_price_frontier(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    base_samples: list[SurrogateTrainSample],
    batch_sizes: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for b in batch_sizes:
        samples = _batch(base_samples, b)
        X = normalizer.transform_features(samples)

        teacher_times = _measure_latency(
            lambda: [_teacher_price(s) for s in samples],
            repeats=5,
            warmup=1,
        )
        surrogate_times = _measure_latency(
            lambda: model.predict(X),
            repeats=40,
            warmup=2,
        )

        teacher_stats = _latency_summary(teacher_times)
        surrogate_stats = _latency_summary(surrogate_times)

        teacher_mean_s = teacher_stats["mean_ms"] / 1000.0
        surrogate_mean_s = surrogate_stats["mean_ms"] / 1000.0

        out[str(b)] = {
            "teacher_price_binomial": {
                "latency": teacher_stats,
                "throughput_samples_per_sec": float(b / teacher_mean_s),
            },
            "surrogate_price_mlp": {
                "latency": surrogate_stats,
                "throughput_samples_per_sec": float(b / surrogate_mean_s),
            },
            "speedup_surrogate_vs_teacher": float(teacher_mean_s / surrogate_mean_s),
        }

    return out


def _benchmark_greeks_frontier(
    model: AmericanPriceMLP,
    normalizer: FeatureNormalizer,
    base_samples: list[SurrogateTrainSample],
    batch_sizes: list[int],
    K: float = _K,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for b in batch_sizes:
        samples = _batch(base_samples, b)

        def teacher_fd_task() -> None:
            for s in samples:
                S = float(K * math.exp(s.log_moneyness))
                teacher_fd_greeks(S, K, s.T, s.r, s.sigma, s.q)

        def surrogate_ad_task() -> None:
            for s in samples:
                S = float(K * math.exp(s.log_moneyness))
                surrogate_ad_greeks(model, normalizer, S, K, s.T, s.r, s.sigma, s.q)

        def surrogate_fd_task() -> None:
            for s in samples:
                S = float(K * math.exp(s.log_moneyness))
                surrogate_fd_greeks(model, normalizer, S, K, s.T, s.r, s.sigma, s.q)

        teacher_times = _measure_latency(teacher_fd_task, repeats=3, warmup=1)
        ad_times = _measure_latency(surrogate_ad_task, repeats=8, warmup=1)
        fd_times = _measure_latency(surrogate_fd_task, repeats=8, warmup=1)

        teacher_stats = _latency_summary(teacher_times)
        ad_stats = _latency_summary(ad_times)
        fd_stats = _latency_summary(fd_times)

        teacher_mean_s = teacher_stats["mean_ms"] / 1000.0
        ad_mean_s = ad_stats["mean_ms"] / 1000.0
        fd_mean_s = fd_stats["mean_ms"] / 1000.0

        out[str(b)] = {
            "teacher_fd_greeks": {
                "latency": teacher_stats,
                "throughput_samples_per_sec": float(b / teacher_mean_s),
            },
            "surrogate_ad_greeks": {
                "latency": ad_stats,
                "throughput_samples_per_sec": float(b / ad_mean_s),
            },
            "surrogate_fd_greeks": {
                "latency": fd_stats,
                "throughput_samples_per_sec": float(b / fd_mean_s),
            },
            "speedup_ad_vs_teacher_fd": float(teacher_mean_s / ad_mean_s),
            "speedup_fd_vs_teacher_fd": float(teacher_mean_s / fd_mean_s),
        }

    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer = load_surrogate_artifacts(
        out_dir / "m6_mlp_weights.npz",
        out_dir / "m6_normalizer.npz",
    )

    # Shared sample pool for timing.
    benchmark_samples = generate_training_data_stratified(
        n_samples=256,
        seed=909,
        binomial_steps=80,
        option_type="put",
    )
    # Accuracy set uses higher teacher steps.
    accuracy_samples = generate_training_data_stratified(
        n_samples=120,
        seed=910,
        binomial_steps=120,
        option_type="put",
    )

    pricing_accuracy = _price_accuracy(model, normalizer, accuracy_samples)
    greeks_accuracy = _greeks_accuracy(model, normalizer, accuracy_samples)

    price_frontier = _benchmark_price_frontier(
        model=model,
        normalizer=normalizer,
        base_samples=benchmark_samples,
        batch_sizes=_PRICE_BATCHES,
    )
    greeks_frontier = _benchmark_greeks_frontier(
        model=model,
        normalizer=normalizer,
        base_samples=benchmark_samples,
        batch_sizes=_GREEKS_BATCHES,
    )

    payload = {
        "module": "M9",
        "description": "Speed-accuracy frontier for surrogate vs teacher pricing and Greeks",
        "source_model": "m6",
        "teacher_pricer": f"binomial_crr_steps_{_TEACHER_STEPS}",
        "benchmark_sizes": {
            "price_batches": _PRICE_BATCHES,
            "greeks_batches": _GREEKS_BATCHES,
            "n_timing_pool": len(benchmark_samples),
            "n_accuracy": len(accuracy_samples),
        },
        "accuracy": {
            "price_vs_teacher": pricing_accuracy,
            **greeks_accuracy,
        },
        "frontier": {
            "pricing": price_frontier,
            "greeks": greeks_frontier,
        },
    }

    out_path = out_dir / "m9_speed_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Brief console summary.
    print(f"M9 report saved -> {out_path}")
    print("\nPrice accuracy:")
    print(f"  MAE={pricing_accuracy['mae']:.4f}, RMSE={pricing_accuracy['rmse']:.4f}, MaxAE={pricing_accuracy['max_ae']:.4f}")
    print("\nGreeks AD vs Teacher FD mean abs error:")
    for g in ("delta", "gamma", "vega"):
        s = greeks_accuracy["ad_vs_teacher_fd_abs_error"][g]
        print(f"  {g:6s}: mean={s['mean']:.6f}, p95={s['p95']:.6f}, max={s['max']:.6f}")

    print("\nPricing speedups (surrogate vs teacher):")
    for b in _PRICE_BATCHES:
        section = price_frontier[str(b)]
        print(f"  batch={b:3d}: {section['speedup_surrogate_vs_teacher']:.2f}x")

    print("\nGreeks speedups (surrogate AD vs teacher FD):")
    for b in _GREEKS_BATCHES:
        section = greeks_frontier[str(b)]
        print(f"  batch={b:3d}: {section['speedup_ad_vs_teacher_fd']:.2f}x")


if __name__ == "__main__":
    main()
