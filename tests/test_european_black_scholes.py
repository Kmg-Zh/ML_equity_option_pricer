import math

from pricing import (
    DiscreteCashDividend,
    DiscreteCashDividendSchedule,
    EuropeanOptionInput,
    FlatContinuousDividendYield,
    PiecewiseContinuousDividendYield,
    YieldSegment,
    black_scholes_analytic_greeks,
    black_scholes_fd_greeks,
    black_scholes_price,
)


def test_put_call_parity_with_dividend_yield() -> None:
    base = dict(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2, q=0.01)
    call = black_scholes_price(EuropeanOptionInput(**base, option_type="call"))
    put = black_scholes_price(EuropeanOptionInput(**base, option_type="put"))

    lhs = call - put
    rhs = base["S"] * math.exp(-base["q"] * base["T"]) - base["K"] * math.exp(
        -base["r"] * base["T"]
    )
    assert abs(lhs - rhs) < 1e-8


def test_analytic_greeks_match_fd_for_call() -> None:
    opt = EuropeanOptionInput(
        S=103.0,
        K=100.0,
        T=0.75,
        r=0.02,
        sigma=0.25,
        q=0.005,
        option_type="call",
    )
    ana = black_scholes_analytic_greeks(opt)
    fd = black_scholes_fd_greeks(opt)

    assert abs(ana["delta"] - fd["delta"]) < 5e-5
    assert abs(ana["gamma"] - fd["gamma"]) < 5e-5
    assert abs(ana["vega"] - fd["vega"]) < 5e-3


def test_analytic_greeks_match_fd_for_put() -> None:
    opt = EuropeanOptionInput(
        S=97.0,
        K=100.0,
        T=1.25,
        r=0.01,
        sigma=0.3,
        q=0.0,
        option_type="put",
    )
    ana = black_scholes_analytic_greeks(opt)
    fd = black_scholes_fd_greeks(opt)

    assert abs(ana["delta"] - fd["delta"]) < 5e-5
    assert abs(ana["gamma"] - fd["gamma"]) < 5e-5
    assert abs(ana["vega"] - fd["vega"]) < 5e-3


def test_reject_invalid_inputs() -> None:
    bad = EuropeanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.2,
        q=0.0,
        option_type="call",
    )

    for field in ("S", "K", "T", "sigma"):
        payload = bad.__dict__.copy()
        payload[field] = 0.0
        try:
            black_scholes_price(EuropeanOptionInput(**payload))
            assert False, f"Expected ValueError for {field}"
        except ValueError:
            pass


def test_piecewise_continuous_yield_matches_flat_yield_when_equivalent() -> None:
    flat = EuropeanOptionInput(
        S=100.0,
        K=95.0,
        T=1.0,
        r=0.03,
        sigma=0.2,
        q=0.02,
        option_type="call",
    )
    piecewise = EuropeanOptionInput(
        S=100.0,
        K=95.0,
        T=1.0,
        r=0.03,
        sigma=0.2,
        dividend_model=PiecewiseContinuousDividendYield(
            segments=(
                YieldSegment(start=0.0, end=0.5, rate=0.01),
                YieldSegment(start=0.5, end=1.0, rate=0.03),
            )
        ),
        option_type="call",
    )

    assert abs(black_scholes_price(flat) - black_scholes_price(piecewise)) < 1e-10


def test_flat_dividend_model_matches_scalar_q() -> None:
    scalar = EuropeanOptionInput(
        S=110.0,
        K=100.0,
        T=0.75,
        r=0.02,
        sigma=0.22,
        q=0.015,
        option_type="put",
    )
    model = EuropeanOptionInput(
        S=110.0,
        K=100.0,
        T=0.75,
        r=0.02,
        sigma=0.22,
        dividend_model=FlatContinuousDividendYield(rate=0.015),
        option_type="put",
    )

    assert abs(black_scholes_price(scalar) - black_scholes_price(model)) < 1e-12


def test_discrete_dividend_schedule_is_explicitly_not_supported_yet() -> None:
    opt = EuropeanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.03,
        sigma=0.2,
        dividend_model=DiscreteCashDividendSchedule(
            dividends=(
                DiscreteCashDividend(time=0.25, amount=1.0),
                DiscreteCashDividend(time=0.75, amount=1.2),
            )
        ),
        option_type="call",
    )

    try:
        black_scholes_price(opt)
        assert False, "Expected NotImplementedError for discrete dividends"
    except NotImplementedError:
        pass
