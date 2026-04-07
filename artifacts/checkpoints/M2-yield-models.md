# M2 Yield Modeling Checkpoint

Date: 2026-03-28

## Goal
Refactor yield handling before M3 so dividend assumptions are explicit in the theory and API.

## Added
- `src/pricing/dividend_models.py`

## Supported now
- Scalar continuous yield `q`
- `FlatContinuousDividendYield`
- `PiecewiseContinuousDividendYield`

## Represented but intentionally deferred
- `DiscreteCashDividendSchedule`

## Theory note
- Continuous and piecewise-continuous yields enter via the integrated carry term $\int_t^T q(u) \, du$.
- This is exact for deterministic continuous yield specifications in the current M1/M2 framework.
- Discrete cash dividends are qualitatively different because they introduce jump conditions in the stock dynamics and option value evolution.
- For that reason, the current code rejects discrete schedules explicitly instead of silently approximating them.

## Validation plan
- Flat dividend model should match scalar `q`.
- Piecewise-continuous schedules should match equivalent integrated flat yields.
- Discrete cash schedules should raise `NotImplementedError` in current pricers.

## Validation result
- Environment: local venv at `.venv`.
- Test command used: `.\\.venv\\bin\\python.exe -m pytest -q`
- Outcome: 14 passed.
- Implementation note: the initial binomial tree treatment used a single carry input, which was not sufficient for time-varying piecewise yields. It was corrected to use step-specific risk-neutral probabilities.

## Next
- Carry this dividend model layer into M3 benchmark inputs.
- Decide whether discrete cash dividends become M2b or a later focused module.