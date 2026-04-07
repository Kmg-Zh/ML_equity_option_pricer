from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TypeAlias


@dataclass(frozen=True)
class FlatContinuousDividendYield:
    rate: float


@dataclass(frozen=True)
class YieldSegment:
    start: float
    end: float
    rate: float


@dataclass(frozen=True)
class PiecewiseContinuousDividendYield:
    segments: tuple[YieldSegment, ...]


@dataclass(frozen=True)
class DiscreteCashDividend:
    time: float
    amount: float


@dataclass(frozen=True)
class DiscreteCashDividendSchedule:
    dividends: tuple[DiscreteCashDividend, ...]


DividendModel: TypeAlias = (
    FlatContinuousDividendYield
    | PiecewiseContinuousDividendYield
    | DiscreteCashDividendSchedule
)


def validate_dividend_model(dividend_model: DividendModel | None, maturity: float) -> None:
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0")
    if dividend_model is None:
        return

    if isinstance(dividend_model, FlatContinuousDividendYield):
        return

    if isinstance(dividend_model, PiecewiseContinuousDividendYield):
        prev_end = 0.0
        for segment in dividend_model.segments:
            if segment.start < 0.0 or segment.end <= segment.start:
                raise ValueError("piecewise yield segments must have 0 <= start < end")
            if segment.start < prev_end:
                raise ValueError("piecewise yield segments must be sorted and non-overlapping")
            prev_end = segment.end
        return

    if isinstance(dividend_model, DiscreteCashDividendSchedule):
        prev_time = -1.0
        for dividend in dividend_model.dividends:
            if dividend.time <= 0.0 or dividend.amount < 0.0:
                raise ValueError("discrete dividends must have time > 0 and amount >= 0")
            if dividend.time <= prev_time:
                raise ValueError("discrete dividends must be strictly increasing in time")
            prev_time = dividend.time
        return

    raise TypeError("unsupported dividend model")


def uses_discrete_dividends(dividend_model: DividendModel | None) -> bool:
    return isinstance(dividend_model, DiscreteCashDividendSchedule)


def integrated_continuous_dividend_yield(
    q: float,
    dividend_model: DividendModel | None,
    start: float,
    end: float,
) -> float:
    if end < start:
        raise ValueError("end must be >= start")
    if dividend_model is not None and abs(q) > 1e-15:
        raise ValueError("provide either q or dividend_model, not both")
    if dividend_model is None:
        return q * (end - start)
    if isinstance(dividend_model, FlatContinuousDividendYield):
        return dividend_model.rate * (end - start)
    if isinstance(dividend_model, PiecewiseContinuousDividendYield):
        total = 0.0
        for segment in dividend_model.segments:
            overlap_start = max(start, segment.start)
            overlap_end = min(end, segment.end)
            if overlap_end > overlap_start:
                total += segment.rate * (overlap_end - overlap_start)
        return total
    if isinstance(dividend_model, DiscreteCashDividendSchedule):
        raise NotImplementedError(
            "Discrete cash dividend schedules require jump-condition support and are not yet implemented in current pricers."
        )
    raise TypeError("unsupported dividend model")


def average_continuous_dividend_yield(
    q: float,
    dividend_model: DividendModel | None,
    start: float,
    end: float,
) -> float:
    if end <= start:
        return 0.0
    return integrated_continuous_dividend_yield(q, dividend_model, start, end) / (end - start)


def dividend_discount_factor(
    q: float,
    dividend_model: DividendModel | None,
    start: float,
    end: float,
) -> float:
    return exp(-integrated_continuous_dividend_yield(q, dividend_model, start, end))
