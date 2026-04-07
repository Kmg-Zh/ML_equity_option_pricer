from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter_ns

try:
    from pricing import (
        AmericanOptionInput,
        FlatContinuousDividendYield,
        PiecewiseContinuousDividendYield,
        YieldSegment,
        american_binomial_price,
        american_fd_price,
        american_lsm_price,
    )
except ModuleNotFoundError:  # pragma: no cover
    from src.pricing import (
        AmericanOptionInput,
        FlatContinuousDividendYield,
        PiecewiseContinuousDividendYield,
        YieldSegment,
        american_binomial_price,
        american_fd_price,
        american_lsm_price,
    )


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    option: AmericanOptionInput
    dividend_label: str


@dataclass(frozen=True)
class TeacherBenchmarkRow:
    case_id: str
    option_type: str
    moneyness: float
    maturity: float
    dividend_label: str
    binomial_price: float
    lsm_price: float
    fdm_price: float
    abs_lsm_vs_binomial: float
    abs_fdm_vs_binomial: float
    abs_fdm_vs_lsm: float
    binomial_us: float
    lsm_us: float
    fdm_us: float


@dataclass(frozen=True)
class BucketSummary:
    moneyness_bucket: str
    maturity_bucket: str
    count: int
    mean_abs_lsm_vs_binomial: float
    mean_abs_fdm_vs_binomial: float
    mean_abs_fdm_vs_lsm: float
    mean_binomial_us: float
    mean_lsm_us: float
    mean_fdm_us: float


def _moneyness_bucket(m: float) -> str:
    if m < 0.95:
        return "OTM"
    if m <= 1.05:
        return "ATM"
    return "ITM"


def _maturity_bucket(t: float) -> str:
    if t <= 0.25:
        return "short"
    if t <= 1.0:
        return "medium"
    return "long"


def _timed_call(callable_fn) -> tuple[float, float]:
    start = perf_counter_ns()
    value = callable_fn()
    elapsed_us = (perf_counter_ns() - start) / 1000.0
    return float(value), elapsed_us


def run_teacher_benchmark(
    cases: list[BenchmarkCase],
    binomial_steps: int = 200,
    lsm_time_steps: int = 30,
    lsm_paths: int = 2000,
    lsm_seed: int = 42,
    fdm_spot_steps: int = 150,
    fdm_time_steps: int = 150,
) -> list[TeacherBenchmarkRow]:
    rows: list[TeacherBenchmarkRow] = []

    for case in cases:
        opt = case.option
        moneyness = opt.S / opt.K

        binomial_price, binomial_us = _timed_call(
            lambda: american_binomial_price(opt, steps=binomial_steps)
        )
        lsm_price, lsm_us = _timed_call(
            lambda: american_lsm_price(
                opt,
                time_steps=lsm_time_steps,
                paths=lsm_paths,
                seed=lsm_seed,
            )
        )
        fdm_price, fdm_us = _timed_call(
            lambda: american_fd_price(
                opt,
                spot_steps=fdm_spot_steps,
                time_steps=fdm_time_steps,
            )
        )

        rows.append(
            TeacherBenchmarkRow(
                case_id=case.case_id,
                option_type=opt.option_type,
                moneyness=moneyness,
                maturity=opt.T,
                dividend_label=case.dividend_label,
                binomial_price=binomial_price,
                lsm_price=lsm_price,
                fdm_price=fdm_price,
                abs_lsm_vs_binomial=abs(lsm_price - binomial_price),
                abs_fdm_vs_binomial=abs(fdm_price - binomial_price),
                abs_fdm_vs_lsm=abs(fdm_price - lsm_price),
                binomial_us=binomial_us,
                lsm_us=lsm_us,
                fdm_us=fdm_us,
            )
        )

    return rows


def aggregate_by_buckets(rows: list[TeacherBenchmarkRow]) -> list[BucketSummary]:
    buckets: dict[tuple[str, str], list[TeacherBenchmarkRow]] = {}

    for row in rows:
        key = (_moneyness_bucket(row.moneyness), _maturity_bucket(row.maturity))
        buckets.setdefault(key, []).append(row)

    summaries: list[BucketSummary] = []
    for key, group in sorted(buckets.items()):
        m_bucket, t_bucket = key
        summaries.append(
            BucketSummary(
                moneyness_bucket=m_bucket,
                maturity_bucket=t_bucket,
                count=len(group),
                mean_abs_lsm_vs_binomial=mean(r.abs_lsm_vs_binomial for r in group),
                mean_abs_fdm_vs_binomial=mean(r.abs_fdm_vs_binomial for r in group),
                mean_abs_fdm_vs_lsm=mean(r.abs_fdm_vs_lsm for r in group),
                mean_binomial_us=mean(r.binomial_us for r in group),
                mean_lsm_us=mean(r.lsm_us for r in group),
                mean_fdm_us=mean(r.fdm_us for r in group),
            )
        )

    return summaries


def build_default_m3_cases() -> list[BenchmarkCase]:
    # Compact grid: 1 rate × 2 vols × 3 maturities × 3 moneyness × 1 type = 18 cases
    # Covers every moneyness+maturity bucket needed for BucketSummary.
    rates = [0.02]
    vols = [0.2, 0.35]
    maturities = [0.25, 1.0, 2.0]
    moneyness_levels = [0.9, 1.0, 1.1]
    option_types = ["put"]

    cases: list[BenchmarkCase] = []
    idx = 0
    for r in rates:
        for sigma in vols:
            for t in maturities:
                for m in moneyness_levels:
                    for option_type in option_types:
                        k = 100.0
                        s = m * k

                        flat_case = BenchmarkCase(
                            case_id=f"c{idx:03d}-flat",
                            option=AmericanOptionInput(
                                S=s,
                                K=k,
                                T=t,
                                r=r,
                                sigma=sigma,
                                dividend_model=FlatContinuousDividendYield(rate=0.01),
                                option_type=option_type,
                            ),
                            dividend_label="flat_yield",
                        )
                        idx += 1

                        piecewise_case = BenchmarkCase(
                            case_id=f"c{idx:03d}-piecewise",
                            option=AmericanOptionInput(
                                S=s,
                                K=k,
                                T=t,
                                r=r,
                                sigma=sigma,
                                dividend_model=PiecewiseContinuousDividendYield(
                                    segments=(
                                        YieldSegment(start=0.0, end=min(0.5, t), rate=0.0),
                                        YieldSegment(start=min(0.5, t), end=t, rate=0.02),
                                    )
                                    if t > 0.5
                                    else (YieldSegment(start=0.0, end=t, rate=0.01),)
                                ),
                                option_type=option_type,
                            ),
                            dividend_label="piecewise_yield",
                        )
                        idx += 1

                        cases.append(flat_case)
                        cases.append(piecewise_case)

    return cases
