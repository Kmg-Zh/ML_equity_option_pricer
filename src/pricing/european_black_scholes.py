from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

from .dividend_models import (
    DividendModel,
    average_continuous_dividend_yield,
    dividend_discount_factor,
    validate_dividend_model,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class EuropeanOptionInput:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    dividend_model: DividendModel | None = None
    option_type: OptionType = "call"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _validate_inputs(opt: EuropeanOptionInput) -> None:
    if opt.S <= 0.0:
        raise ValueError("S must be > 0")
    if opt.K <= 0.0:
        raise ValueError("K must be > 0")
    if opt.T <= 0.0:
        raise ValueError("T must be > 0")
    if opt.sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if opt.option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    validate_dividend_model(opt.dividend_model, opt.T)


def _d1_d2(opt: EuropeanOptionInput) -> tuple[float, float]:
    sqrt_t = sqrt(opt.T)
    q_bar = average_continuous_dividend_yield(opt.q, opt.dividend_model, 0.0, opt.T)
    d1 = (
        log(opt.S / opt.K)
        + (opt.r - q_bar + 0.5 * opt.sigma * opt.sigma) * opt.T
    ) / (opt.sigma * sqrt_t)
    d2 = d1 - opt.sigma * sqrt_t
    return d1, d2


def black_scholes_price(opt: EuropeanOptionInput) -> float:
    _validate_inputs(opt)
    d1, d2 = _d1_d2(opt)

    disc_q = dividend_discount_factor(opt.q, opt.dividend_model, 0.0, opt.T)
    disc_r = exp(-opt.r * opt.T)

    if opt.option_type == "call":
        return opt.S * disc_q * _norm_cdf(d1) - opt.K * disc_r * _norm_cdf(d2)

    return opt.K * disc_r * _norm_cdf(-d2) - opt.S * disc_q * _norm_cdf(-d1)


def black_scholes_analytic_greeks(opt: EuropeanOptionInput) -> dict[str, float]:
    _validate_inputs(opt)
    d1, _d2 = _d1_d2(opt)

    disc_q = dividend_discount_factor(opt.q, opt.dividend_model, 0.0, opt.T)
    sqrt_t = sqrt(opt.T)
    pdf_d1 = _norm_pdf(d1)

    if opt.option_type == "call":
        delta = disc_q * _norm_cdf(d1)
    else:
        delta = disc_q * (_norm_cdf(d1) - 1.0)

    gamma = disc_q * pdf_d1 / (opt.S * opt.sigma * sqrt_t)
    vega = opt.S * disc_q * pdf_d1 * sqrt_t

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }


def black_scholes_fd_greeks(
    opt: EuropeanOptionInput,
    dS: float = 1e-2,
    d_sigma: float = 1e-4,
) -> dict[str, float]:
    _validate_inputs(opt)

    if dS <= 0.0 or d_sigma <= 0.0:
        raise ValueError("dS and d_sigma must be > 0")

    up_s = EuropeanOptionInput(
        S=opt.S + dS,
        K=opt.K,
        T=opt.T,
        r=opt.r,
        sigma=opt.sigma,
        q=opt.q,
        dividend_model=opt.dividend_model,
        option_type=opt.option_type,
    )
    dn_s = EuropeanOptionInput(
        S=opt.S - dS,
        K=opt.K,
        T=opt.T,
        r=opt.r,
        sigma=opt.sigma,
        q=opt.q,
        dividend_model=opt.dividend_model,
        option_type=opt.option_type,
    )
    base = black_scholes_price(opt)
    up_price = black_scholes_price(up_s)
    dn_price = black_scholes_price(dn_s)

    delta = (up_price - dn_price) / (2.0 * dS)
    gamma = (up_price - 2.0 * base + dn_price) / (dS * dS)

    up_sigma = EuropeanOptionInput(
        S=opt.S,
        K=opt.K,
        T=opt.T,
        r=opt.r,
        sigma=opt.sigma + d_sigma,
        q=opt.q,
        dividend_model=opt.dividend_model,
        option_type=opt.option_type,
    )
    dn_sigma = EuropeanOptionInput(
        S=opt.S,
        K=opt.K,
        T=opt.T,
        r=opt.r,
        sigma=opt.sigma - d_sigma,
        q=opt.q,
        dividend_model=opt.dividend_model,
        option_type=opt.option_type,
    )
    up_sigma_price = black_scholes_price(up_sigma)
    dn_sigma_price = black_scholes_price(dn_sigma)
    vega = (up_sigma_price - dn_sigma_price) / (2.0 * d_sigma)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }
