# ML Equity Option Pricer

This repository implements a modular research pipeline for equity option pricing.

Current status:
- M1 implemented: European Black-Scholes teacher with analytic Greeks.
- M2 implemented: American teachers with Binomial, Longstaff-Schwartz, and FDM baselines.
- M3 implemented: dividend-aware benchmark harness with bucketed consistency summaries.
- M4 implemented: numpy MLP surrogate baseline for American put pricing.
- M5 implemented: tail-robust training via stratified bucket sampling and weighted loss.
- M6 implemented: financial-constraint penalties (monotonicity + convexity) with violation diagnostics.
- M7 implemented: AD Greeks pipeline (Delta/Gamma/Vega) with AD-vs-FD consistency diagnostics.
- Includes finite-difference Greeks for validation checks.
- Includes baseline unit tests.

## Quick start

1. Install dependencies:
   pip install -r requirements.txt
2. Run tests:
   pytest -q

## Current module

M1 (European teacher):
- Inputs: `S, K, T, r, sigma, q, option_type`
- Outputs: price, delta, gamma, vega
- Validation utilities: finite-difference Greek approximations

M2 (American teachers):
- Inputs: `S, K, T, r, sigma, q, option_type`
- Outputs: price from Binomial, Longstaff-Schwartz, and FDM teachers
- Theory focus: optimal stopping, free-boundary behavior, and cross-method consistency

Dividend handling status:
- Supported exactly in current implementation: flat continuous yield `q`, flat continuous dividend models, and piecewise-continuous yield schedules.
- Explicitly modeled but not yet implemented in current pricers: discrete cash dividend schedules.
- Rationale: discrete cash dividends require jump-condition handling in PDE/Monte Carlo/tree dynamics; we keep that boundary explicit rather than hiding it behind a weak approximation.

## Next modules

- M8: FD Greeks comparison
- M9: speed-accuracy frontier

## M3 benchmark harness

The M3 harness compares American teachers on a shared case grid and reports:
1. Method-consistency metrics: |LSM-Binomial|, |FDM-Binomial|, |FDM-LSM|
2. Bucketed summaries by moneyness and maturity
3. Runtime signals (microseconds) as secondary evidence

Run M3 report generation:
`python -m src.benchmarks.run_m3`

Output:
- `artifacts/benchmarks/m3_summary.json`

## M4/M5 surrogate runs

Run M4 baseline:
`python -m src.surrogate.run_m4`

Outputs:
- `artifacts/surrogate/m4_mlp_weights.npz`
- `artifacts/surrogate/m4_normalizer.npz`
- `artifacts/surrogate/m4_eval_report.json`

Run M5 robust training:
`python -m src.surrogate.run_m5`

Outputs:
- `artifacts/surrogate/m5_mlp_weights.npz`
- `artifacts/surrogate/m5_normalizer.npz`
- `artifacts/surrogate/m5_eval_report.json`

M5 methodology:
- Stratified sampling across 9 buckets (OTM/ATM/ITM x short/medium/long)
- Weighted loss emphasizing short maturities and wing moneyness

Run M6 constrained training:
`python -m src.surrogate.run_m6`

Outputs:
- `artifacts/surrogate/m6_mlp_weights.npz`
- `artifacts/surrogate/m6_normalizer.npz`
- `artifacts/surrogate/m6_eval_report.json`

M6 methodology:
- Keep M5 stratified data + weighted loss
- Add training penalties for put-shape constraints in log-moneyness:
   - monotonicity (non-increasing vs moneyness)
   - convexity (non-negative second finite difference)
- Emit explicit violation metrics on held-out data

Run M7 AD Greeks diagnostics:
`python -m src.surrogate.run_m7`

Outputs:
- `artifacts/surrogate/m7_ad_greeks_report.json`

M7 methodology:
- Use model-input AD chain rule on normalized features to compute Greeks.
- Provide Delta/Gamma/Vega from the surrogate in price units.
- Compare AD Greeks against surrogate FD Greeks on held-out points.

## Theory emphasis

This project is theory-first:
- American options are treated as optimal stopping / free-boundary problems.
- LSM, Binomial, and FDM are compared as distinct numerical approximations.
- ML surrogates will be evaluated as approximators of trusted numerical teachers, not as black-box speed hacks.
- Dividend assumptions are treated as part of the model specification, not as a cosmetic input field.

## Yield modeling roadmap

1. Flat continuous yield:
   exact in current M1/M2 code.
2. Piecewise-continuous yield schedule:
   exact in current M1/M2 code via integrated yield over time intervals.
3. Discrete cash dividend schedule:
   represented in the API, but intentionally deferred until jump-condition support is added.
4. Fundamental-analysis schedule:
   can already enter as a piecewise-continuous yield schedule if expressed as forecast yield by period; if expressed as forecast cash dividends, it maps naturally to the deferred discrete schedule API.



