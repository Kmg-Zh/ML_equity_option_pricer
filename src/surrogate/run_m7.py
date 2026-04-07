"""M7 entry-point: AD Greeks pipeline and AD-vs-FD consistency diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_gen import _K, generate_training_data
from .greeks import load_surrogate_artifacts, surrogate_ad_greeks, surrogate_fd_greeks


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "m6_mlp_weights.npz"
    normalizer_path = out_dir / "m6_normalizer.npz"

    model, normalizer = load_surrogate_artifacts(weights_path, normalizer_path)

    # Hold-out points for Greeks consistency checks.
    eval_samples = generate_training_data(
        n_samples=220,
        seed=303,
        binomial_steps=120,
        option_type="put",
    )

    delta_abs = []
    gamma_abs = []
    vega_abs = []

    for s in eval_samples:
        S = _K * np.exp(s.log_moneyness)

        ad = surrogate_ad_greeks(
            model,
            normalizer,
            S=float(S),
            K=_K,
            T=s.T,
            r=s.r,
            sigma=s.sigma,
            q=s.q,
        )
        fd = surrogate_fd_greeks(
            model,
            normalizer,
            S=float(S),
            K=_K,
            T=s.T,
            r=s.r,
            sigma=s.sigma,
            q=s.q,
        )

        delta_abs.append(abs(ad.delta - fd.delta))
        gamma_abs.append(abs(ad.gamma - fd.gamma))
        vega_abs.append(abs(ad.vega - fd.vega))

    payload = {
        "module": "M7",
        "mode": "AD Greeks on surrogate",
        "source_model": "m6",
        "n_eval": len(eval_samples),
        "ad_vs_fd_abs_error": {
            "delta": _summary(delta_abs),
            "gamma": _summary(gamma_abs),
            "vega": _summary(vega_abs),
        },
    }

    report_path = out_dir / "m7_ad_greeks_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"AD Greeks report saved -> {report_path}")
    print("AD-vs-FD abs error (mean / p95 / max):")
    print(f"  Delta: {payload['ad_vs_fd_abs_error']['delta']}")
    print(f"  Gamma: {payload['ad_vs_fd_abs_error']['gamma']}")
    print(f"  Vega:  {payload['ad_vs_fd_abs_error']['vega']}")


if __name__ == "__main__":
    main()
