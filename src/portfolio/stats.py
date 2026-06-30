"""Return / vol / Sharpe / drawdown helpers. Pure functions, no globals."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric annualization of simple returns; arithmetic-scaled mean for log returns."""
    return float(returns.mean() * periods)


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float, periods: int = TRADING_DAYS) -> float:
    excess = annualized_return(returns, periods) - rf
    vol = annualized_vol(returns, periods)
    if vol == 0:
        return float("nan")
    return excess / vol


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Returns (mdd_pct, peak_date, trough_date). mdd_pct is negative."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_date = equity.loc[:trough].idxmax()
    return float(dd.min()), peak_date, trough


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def portfolio_returns(weights: np.ndarray, asset_returns: pd.DataFrame) -> pd.Series:
    """Daily portfolio simple returns for static weights (used in MC sampling)."""
    return pd.Series(asset_returns.values @ weights, index=asset_returns.index)


def equity_curve(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    return initial * (1.0 + returns).cumprod()


def summary_stats(returns: pd.Series, rf: float) -> dict:
    """One-shot kitchen sink for tearsheet / KPI cards."""
    eq = equity_curve(returns)
    mdd, peak_d, trough_d = max_drawdown(eq)
    return {
        "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / len(returns)) - 1.0,
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe(returns, rf),
        "max_drawdown": mdd,
        "mdd_peak": peak_d,
        "mdd_trough": trough_d,
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }


def turnover(weights_t: np.ndarray, weights_drifted: np.ndarray) -> float:
    """One-way turnover = 0.5 * sum(|w_new - w_drifted|)."""
    return 0.5 * float(np.abs(weights_t - weights_drifted).sum())
