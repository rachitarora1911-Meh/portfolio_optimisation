"""SLSQP-based portfolio optimization. Long-only, sum-to-1, per-asset weight cap.

Three profiles:
- Conservative: minimize variance (Σ only, no mu dependence — most stable).
- Balanced: maximize Sharpe.
- Aggressive: maximize expected return subject to vol ceiling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .stats import TRADING_DAYS


def _portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def _portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def _neg_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = _portfolio_vol(weights, cov)
    if vol < 1e-9:
        return 1e6
    return -(weights @ mu - rf) / vol


def _sum_to_one(w: np.ndarray) -> float:
    return float(w.sum() - 1.0)


def _initial_guess(n: int, w_max: float) -> np.ndarray:
    """Equal weight, clipped to cap."""
    return np.minimum(np.ones(n) / n, w_max)


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    w_min: float = 0.0,
    w_max: float = 0.40,
) -> pd.Series:
    mu_arr = mu.values
    cov_arr = cov.values
    n = len(mu_arr)
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": _sum_to_one}]
    x0 = _initial_guess(n, w_max)
    res = minimize(
        _neg_sharpe,
        x0,
        args=(mu_arr, cov_arr, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Max-Sharpe SLSQP failed: {res.message}")
    return pd.Series(res.x, index=mu.index, name="max_sharpe")


def optimize_min_variance(
    cov: pd.DataFrame,
    w_min: float = 0.0,
    w_max: float = 0.40,
) -> pd.Series:
    cov_arr = cov.values
    n = cov_arr.shape[0]
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": _sum_to_one}]
    x0 = _initial_guess(n, w_max)
    res = minimize(
        lambda w: _portfolio_vol(w, cov_arr),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Min-variance SLSQP failed: {res.message}")
    return pd.Series(res.x, index=cov.index, name="min_variance")


def optimize_max_return_at_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    vol_cap: float,
    w_min: float = 0.0,
    w_max: float = 0.40,
) -> pd.Series:
    mu_arr = mu.values
    cov_arr = cov.values
    n = len(mu_arr)
    bounds = [(w_min, w_max)] * n
    constraints = [
        {"type": "eq", "fun": _sum_to_one},
        {"type": "ineq", "fun": lambda w: vol_cap - _portfolio_vol(w, cov_arr)},
    ]
    x0 = _initial_guess(n, w_max)
    res = minimize(
        lambda w: -_portfolio_return(w, mu_arr),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not res.success:
        # Fallback: vol cap may be tighter than min-vol portfolio. Return min-vol as floor.
        return optimize_min_variance(cov, w_min, w_max).rename("aggressive_fallback")
    return pd.Series(res.x, index=mu.index, name="aggressive")


def build_three_profiles(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    vol_cap: float,
    w_min: float = 0.0,
    w_max: float = 0.40,
) -> pd.DataFrame:
    """Returns DataFrame [tickers x {conservative, balanced, aggressive}]."""
    conservative = optimize_min_variance(cov, w_min, w_max).rename("conservative")
    balanced = optimize_max_sharpe(mu, cov, rf, w_min, w_max).rename("balanced")
    aggressive = optimize_max_return_at_vol(mu, cov, vol_cap, w_min, w_max).rename("aggressive")
    return pd.concat([conservative, balanced, aggressive], axis=1)


def annualize_mu_cov(daily_returns: pd.DataFrame, periods: int = TRADING_DAYS) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized mean + cov from daily returns."""
    mu = daily_returns.mean() * periods
    cov = daily_returns.cov() * periods
    return mu, cov
