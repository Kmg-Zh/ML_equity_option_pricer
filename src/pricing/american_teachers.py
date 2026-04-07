from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random
from typing import Literal

from .dividend_models import (
    DividendModel,
    average_continuous_dividend_yield,
    dividend_discount_factor,
    validate_dividend_model,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class AmericanOptionInput:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    dividend_model: DividendModel | None = None
    option_type: OptionType = "put"


def _validate_inputs(opt: AmericanOptionInput) -> None:
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


def _intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def american_binomial_price(opt: AmericanOptionInput, steps: int = 200) -> float:
    _validate_inputs(opt)
    if steps < 2:
        raise ValueError("steps must be >= 2")

    dt = opt.T / steps
    up = exp(opt.sigma * sqrt(dt))
    down = 1.0 / up
    probs = []
    discs = []
    for step in range(steps):
        q_bar = average_continuous_dividend_yield(
            opt.q,
            opt.dividend_model,
            step * dt,
            (step + 1) * dt,
        )
        growth = exp((opt.r - q_bar) * dt)
        prob = (growth - down) / (up - down)
        if not 0.0 < prob < 1.0:
            raise ValueError("risk-neutral probability out of bounds; increase steps")
        probs.append(prob)
        discs.append(exp(-opt.r * dt))

    values = []
    for j in range(steps + 1):
        spot = opt.S * (up ** j) * (down ** (steps - j))
        values.append(_intrinsic_value(spot, opt.K, opt.option_type))

    for step in range(steps - 1, -1, -1):
        next_values = []
        prob = probs[step]
        disc = discs[step]
        for j in range(step + 1):
            continuation = disc * (prob * values[j + 1] + (1.0 - prob) * values[j])
            spot = opt.S * (up ** j) * (down ** (step - j))
            exercise = _intrinsic_value(spot, opt.K, opt.option_type)
            next_values.append(max(exercise, continuation))
        values = next_values

    return values[0]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs_val] for row, rhs_val in zip(matrix, rhs)]

    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-12:
            raise ValueError("singular linear system")

        for col in range(pivot, size + 1):
            augmented[pivot][col] /= pivot_value

        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            for col in range(pivot, size + 1):
                augmented[row][col] -= factor * augmented[pivot][col]

    return [augmented[row][size] for row in range(size)]


def _fit_quadratic_basis(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 3:
        avg = sum(ys) / len(ys)
        return avg, 0.0, 0.0

    sum_x = sum(xs)
    sum_x2 = sum(x * x for x in xs)
    sum_x3 = sum(x * x * x for x in xs)
    sum_x4 = sum(x * x * x * x for x in xs)

    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2y = sum((x * x) * y for x, y in zip(xs, ys))

    matrix = [
        [float(len(xs)), sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ]
    rhs = [sum_y, sum_xy, sum_x2y]
    try:
        a0, a1, a2 = _solve_linear_system(matrix, rhs)
    except ValueError:
        avg = sum(ys) / len(ys)
        return avg, 0.0, 0.0
    return a0, a1, a2


def american_lsm_price(
    opt: AmericanOptionInput,
    time_steps: int = 50,
    paths: int = 4000,
    seed: int = 42,
) -> float:
    _validate_inputs(opt)
    if time_steps < 2:
        raise ValueError("time_steps must be >= 2")
    if paths < 10:
        raise ValueError("paths must be >= 10")

    dt = opt.T / time_steps
    diffusion = opt.sigma * sqrt(dt)
    disc = exp(-opt.r * dt)
    rng = Random(seed)

    stock_paths: list[list[float]] = []
    for _ in range(paths):
        path = [opt.S]
        for step in range(time_steps):
            q_bar = average_continuous_dividend_yield(
                opt.q,
                opt.dividend_model,
                step * dt,
                (step + 1) * dt,
            )
            drift = (opt.r - q_bar - 0.5 * opt.sigma * opt.sigma) * dt
            shock = rng.gauss(0.0, 1.0)
            path.append(path[-1] * exp(drift + diffusion * shock))
        stock_paths.append(path)

    cashflows = [
        _intrinsic_value(path[-1], opt.K, opt.option_type) for path in stock_paths
    ]

    for step in range(time_steps - 1, 0, -1):
        cashflows = [disc * cashflow for cashflow in cashflows]
        x_vals = []
        y_vals = []
        exercises = [0.0] * paths

        for path_idx, path in enumerate(stock_paths):
            exercise_value = _intrinsic_value(path[step], opt.K, opt.option_type)
            exercises[path_idx] = exercise_value
            if exercise_value > 0.0:
                x_vals.append(path[step])
                y_vals.append(cashflows[path_idx])

        if x_vals:
            a0, a1, a2 = _fit_quadratic_basis(x_vals, y_vals)
            for path_idx, path in enumerate(stock_paths):
                exercise_value = exercises[path_idx]
                if exercise_value <= 0.0:
                    continue
                spot = path[step]
                continuation = a0 + a1 * spot + a2 * spot * spot
                if exercise_value > continuation:
                    cashflows[path_idx] = exercise_value

    continuation_value = disc * (sum(cashflows) / paths)
    return max(_intrinsic_value(opt.S, opt.K, opt.option_type), continuation_value)


def _solve_tridiagonal(
    lower: list[float],
    diag: list[float],
    upper: list[float],
    rhs: list[float],
) -> list[float]:
    n = len(diag)
    c_star = [0.0] * n
    d_star = [0.0] * n

    c_star[0] = upper[0] / diag[0] if n > 1 else 0.0
    d_star[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        if abs(denom) < 1e-12:
            raise ValueError("tridiagonal system is singular")
        c_star[i] = upper[i] / denom if i < n - 1 else 0.0
        d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom

    solution = [0.0] * n
    solution[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_star[i] - c_star[i] * solution[i + 1]
    return solution


def american_fd_price(
    opt: AmericanOptionInput,
    spot_steps: int = 200,
    time_steps: int = 200,
    spot_multiple: float = 3.0,
) -> float:
    _validate_inputs(opt)
    if spot_steps < 4:
        raise ValueError("spot_steps must be >= 4")
    if time_steps < 2:
        raise ValueError("time_steps must be >= 2")
    if spot_multiple <= 1.0:
        raise ValueError("spot_multiple must be > 1")

    s_max = spot_multiple * max(opt.S, opt.K)
    ds = s_max / spot_steps
    dt = opt.T / time_steps
    grid = [i * ds for i in range(spot_steps + 1)]
    values = [_intrinsic_value(spot, opt.K, opt.option_type) for spot in grid]

    for step in range(time_steps - 1, -1, -1):
        current_time = step * dt
        remaining_time = opt.T - step * dt
        if opt.option_type == "call":
            left_boundary = 0.0
            right_boundary = max(
                s_max - opt.K,
                s_max * dividend_discount_factor(
                    opt.q,
                    opt.dividend_model,
                    current_time,
                    opt.T,
                )
                - opt.K * exp(-opt.r * remaining_time),
            )
        else:
            left_boundary = opt.K
            right_boundary = 0.0

        lower = []
        diag = []
        upper = []
        rhs = []

        for i in range(1, spot_steps):
            idx = float(i)
            q_bar = average_continuous_dividend_yield(
                opt.q,
                opt.dividend_model,
                current_time,
                current_time + dt,
            )
            a = 0.5 * dt * (
                opt.sigma * opt.sigma * idx * idx - (opt.r - q_bar) * idx
            )
            b = 1.0 + dt * (opt.sigma * opt.sigma * idx * idx + opt.r)
            c = 0.5 * dt * (
                opt.sigma * opt.sigma * idx * idx + (opt.r - q_bar) * idx
            )

            lower_coeff = -a
            upper_coeff = -c

            if i > 1:
                lower.append(lower_coeff)
            diag.append(b)
            if i < spot_steps - 1:
                upper.append(upper_coeff)

            rhs_value = values[i]
            if i == 1:
                rhs_value -= lower_coeff * left_boundary
            if i == spot_steps - 1:
                rhs_value -= upper_coeff * right_boundary
            rhs.append(rhs_value)

        interior = _solve_tridiagonal(lower, diag, upper, rhs)
        new_values = [left_boundary]
        for i, continuation in enumerate(interior, start=1):
            exercise = _intrinsic_value(grid[i], opt.K, opt.option_type)
            new_values.append(max(exercise, continuation))
        new_values.append(right_boundary)
        values = new_values

    lower_idx = min(int(opt.S / ds), spot_steps - 1)
    upper_idx = lower_idx + 1
    lower_spot = grid[lower_idx]
    upper_spot = grid[upper_idx]
    weight = (opt.S - lower_spot) / (upper_spot - lower_spot)
    return values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
