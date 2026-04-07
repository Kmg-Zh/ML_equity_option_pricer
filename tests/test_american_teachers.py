from pricing import (
    AmericanOptionInput,
    DiscreteCashDividend,
    DiscreteCashDividendSchedule,
    FlatContinuousDividendYield,
    PiecewiseContinuousDividendYield,
    YieldSegment,
    american_binomial_price,
    american_fd_price,
    american_lsm_price,
    black_scholes_price,
    EuropeanOptionInput,
)


def test_american_call_without_dividend_matches_european_call() -> None:
    opt = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.0,
        option_type="call",
    )
    euro = black_scholes_price(
        EuropeanOptionInput(
            S=opt.S,
            K=opt.K,
            T=opt.T,
            r=opt.r,
            sigma=opt.sigma,
            q=opt.q,
            option_type="call",
        )
    )
    bino = american_binomial_price(opt, steps=300)
    fdm = american_fd_price(opt, spot_steps=180, time_steps=180)

    assert abs(bino - euro) < 0.05
    assert abs(fdm - euro) < 0.15


def test_american_put_is_at_least_european_put() -> None:
    a_opt = AmericanOptionInput(
        S=95.0,
        K=100.0,
        T=1.0,
        r=0.04,
        sigma=0.25,
        q=0.0,
        option_type="put",
    )
    e_opt = EuropeanOptionInput(
        S=a_opt.S,
        K=a_opt.K,
        T=a_opt.T,
        r=a_opt.r,
        sigma=a_opt.sigma,
        q=a_opt.q,
        option_type="put",
    )

    euro = black_scholes_price(e_opt)
    bino = american_binomial_price(a_opt, steps=250)
    fdm = american_fd_price(a_opt, spot_steps=180, time_steps=180)
    lsm = american_lsm_price(a_opt, time_steps=40, paths=2500, seed=7)

    assert bino >= euro
    assert fdm >= euro
    assert lsm >= euro - 0.25


def test_three_teachers_are_reasonably_consistent() -> None:
    opt = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.06,
        sigma=0.2,
        q=0.01,
        option_type="put",
    )

    bino = american_binomial_price(opt, steps=300)
    fdm = american_fd_price(opt, spot_steps=220, time_steps=220)
    lsm = american_lsm_price(opt, time_steps=50, paths=4000, seed=11)

    assert abs(bino - fdm) < 0.35
    assert abs(bino - lsm) < 0.6


def test_reject_invalid_american_inputs() -> None:
    opt = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.03,
        sigma=0.2,
        q=0.0,
        option_type="put",
    )

    for field in ("S", "K", "T", "sigma"):
        payload = opt.__dict__.copy()
        payload[field] = 0.0
        try:
            american_binomial_price(AmericanOptionInput(**payload))
            assert False, f"Expected ValueError for {field}"
        except ValueError:
            pass


def test_piecewise_continuous_yield_matches_flat_yield_for_binomial() -> None:
    flat = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.02,
        option_type="put",
    )
    piecewise = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        dividend_model=PiecewiseContinuousDividendYield(
            segments=(
                YieldSegment(start=0.0, end=0.25, rate=0.00),
                YieldSegment(start=0.25, end=1.0, rate=0.02666666666666667),
            )
        ),
        option_type="put",
    )

    flat_price = american_binomial_price(flat, steps=250)
    piecewise_price = american_binomial_price(piecewise, steps=250)
    assert abs(flat_price - piecewise_price) < 0.1


def test_flat_dividend_model_matches_scalar_q_for_lsm() -> None:
    scalar = AmericanOptionInput(
        S=102.0,
        K=100.0,
        T=1.0,
        r=0.04,
        sigma=0.22,
        q=0.01,
        option_type="put",
    )
    model = AmericanOptionInput(
        S=102.0,
        K=100.0,
        T=1.0,
        r=0.04,
        sigma=0.22,
        dividend_model=FlatContinuousDividendYield(rate=0.01),
        option_type="put",
    )

    scalar_price = american_lsm_price(scalar, time_steps=40, paths=3000, seed=5)
    model_price = american_lsm_price(model, time_steps=40, paths=3000, seed=5)
    assert abs(scalar_price - model_price) < 1e-10


def test_discrete_dividend_schedule_is_explicitly_rejected_for_current_american_pricers() -> None:
    opt = AmericanOptionInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.04,
        sigma=0.2,
        dividend_model=DiscreteCashDividendSchedule(
            dividends=(DiscreteCashDividend(time=0.5, amount=1.5),)
        ),
        option_type="put",
    )

    for pricer in (american_binomial_price, american_lsm_price, american_fd_price):
        try:
            pricer(opt)
            assert False, "Expected NotImplementedError for discrete dividends"
        except NotImplementedError:
            pass
