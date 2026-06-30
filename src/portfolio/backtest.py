"""Walk-forward backtest with quarterly rebalance, transaction costs, and look-ahead guard.

Strategy callable signature: (mu, cov) -> weights (pd.Series indexed by tickers).
Build via functools.partial if you need to bake in rf, vol_cap, etc.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .stats import TRADING_DAYS, drawdown_series, equity_curve, max_drawdown, summary_stats, turnover

logger = logging.getLogger(__name__)

StrategyFn = Callable[[pd.Series, pd.DataFrame], pd.Series]


def generate_rebalance_dates(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str = "Q",
    index: pd.DatetimeIndex | None = None,
) -> list[pd.Timestamp]:
    """Quarter-ends (or freq-ends) snapped to the nearest prior business day in `index`."""
    raw = pd.date_range(start=start, end=end, freq=freq + "E" if freq == "Q" else freq)
    if index is None:
        return list(raw)
    snapped = []
    for d in raw:
        prior = index[index <= d]
        if len(prior):
            snapped.append(prior[-1])
    # dedupe + sort
    return sorted(set(snapped))


def _annualize_mu_cov(daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return daily.mean() * TRADING_DAYS, daily.cov() * TRADING_DAYS


def walk_forward_weights(
    prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    strategy_fn: StrategyFn,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """At each rebalance date, slice prior `lookback_years` prices and call strategy_fn.

    Hard no-look-ahead guard: lookback prices' max index must be strictly < rebalance date.
    """
    tickers = list(prices.columns)
    rows = []
    for d in rebalance_dates:
        lookback_start = d - pd.DateOffset(years=lookback_years)
        lookback = prices.loc[lookback_start : d - pd.Timedelta(days=1)]
        assert lookback.index.max() < d, (
            f"Look-ahead violation: lookback max {lookback.index.max()} not < rebalance {d}"
        )
        daily = lookback.pct_change().dropna(how="all")
        mu, cov = _annualize_mu_cov(daily)
        w = strategy_fn(mu, cov)
        rows.append(w.reindex(tickers).fillna(0.0).rename(d))
    weights_df = pd.concat(rows, axis=1).T
    weights_df.index.name = "rebalance_date"
    return weights_df


def apply_weights(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    tx_cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate daily portfolio returns with drifting weights between rebalances + tx costs.

    Returns: (gross_daily_returns, net_daily_returns, tx_costs_series).
    """
    tickers = list(prices.columns)
    rebal_dates = list(weights_df.index)
    daily_simple = prices.pct_change().fillna(0.0)

    gross = pd.Series(0.0, index=prices.index)
    net = pd.Series(0.0, index=prices.index)
    costs = pd.Series(0.0, index=prices.index)

    current_w = pd.Series(0.0, index=tickers)
    bps_decimal = tx_cost_bps / 1e4

    for i, d in enumerate(rebal_dates):
        target_w = weights_df.loc[d].reindex(tickers).fillna(0.0)
        if i == 0:
            tover = float(target_w.abs().sum() * 0.5)  # initial buy from cash
        else:
            tover = turnover(target_w.values, current_w.values)
        cost = tover * bps_decimal
        # Apply cost on rebalance day as a return drag
        if d in prices.index:
            net.loc[d] -= cost
            costs.loc[d] = cost
        current_w = target_w.copy()

        # Determine date range until next rebalance (exclusive)
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1] + pd.Timedelta(days=1)
        window = prices.loc[d:next_d - pd.Timedelta(days=1)].index
        if len(window) == 0:
            continue

        # Simulate day-by-day with drifting weights
        w_drift = current_w.values.copy()
        for date in window:
            r = daily_simple.loc[date].values
            day_ret = float(np.dot(w_drift, r))
            gross.loc[date] += day_ret
            net.loc[date] += day_ret
            # Drift weights with returns
            w_drift = w_drift * (1.0 + r)
            s = w_drift.sum()
            if s > 1e-9:
                w_drift = w_drift / s
        current_w = pd.Series(w_drift, index=tickers)

    return gross, net, costs


def equal_weight_strategy(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    n = len(mu)
    return pd.Series(np.ones(n) / n, index=mu.index, name="equal_weight")


def benchmark_buy_and_hold(benchmark_prices: pd.Series) -> pd.Series:
    """Daily simple returns of benchmark, no rebalancing, no tx costs."""
    return benchmark_prices.pct_change().fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    strategy_fns: dict[str, StrategyFn],
    rebalance_dates: list[pd.Timestamp],
    lookback_years: int,
    tx_cost_bps: float,
    rf: float,
    test_start: str | pd.Timestamp,
) -> dict:
    """Run all strategies + equal-weight + benchmark buy-and-hold.

    Out-of-sample reporting starts at `test_start`. Pre-test_start returns are in-sample.
    """
    results = {"weights": {}, "gross": {}, "net": {}, "costs": {}, "equity": {}, "drawdown": {}}

    for name, fn in strategy_fns.items():
        w_df = walk_forward_weights(prices, rebalance_dates, fn, lookback_years)
        gross, net, costs = apply_weights(prices, w_df, tx_cost_bps)
        results["weights"][name] = w_df
        results["gross"][name] = gross
        results["net"][name] = net
        results["costs"][name] = costs
        results["equity"][name] = equity_curve(net)
        results["drawdown"][name] = drawdown_series(results["equity"][name])

    # Equal-weight strategy (rebalanced same cadence, same tx costs)
    ew_w = walk_forward_weights(prices, rebalance_dates, equal_weight_strategy, lookback_years)
    ew_gross, ew_net, ew_costs = apply_weights(prices, ew_w, tx_cost_bps)
    results["weights"]["equal_weight"] = ew_w
    results["gross"]["equal_weight"] = ew_gross
    results["net"]["equal_weight"] = ew_net
    results["costs"]["equal_weight"] = ew_costs
    results["equity"]["equal_weight"] = equity_curve(ew_net)
    results["drawdown"]["equal_weight"] = drawdown_series(results["equity"]["equal_weight"])

    # Benchmark buy-and-hold (no tx costs)
    bench_rets = benchmark_buy_and_hold(benchmark_prices.reindex(prices.index).ffill())
    results["gross"]["benchmark"] = bench_rets
    results["net"]["benchmark"] = bench_rets
    results["costs"]["benchmark"] = pd.Series(0.0, index=prices.index)
    results["equity"]["benchmark"] = equity_curve(bench_rets)
    results["drawdown"]["benchmark"] = drawdown_series(results["equity"]["benchmark"])

    # Summary: OOS only (walk-forward IS the out-of-sample test).
    # Truncate all equity/drawdown/return series to the test window for honest reporting.
    test_start = pd.Timestamp(test_start)
    for key in ("net", "gross", "equity", "drawdown", "costs"):
        results[key] = {n: s.loc[test_start:] for n, s in results[key].items()}
    # Rebase equity to 100 at test start
    for n, eq in results["equity"].items():
        if len(eq) and eq.iloc[0] != 0:
            results["equity"][n] = 100.0 * eq / eq.iloc[0]
        results["drawdown"][n] = drawdown_series(results["equity"][n])

    rows = []
    for name in results["net"]:
        oos_returns = results["net"][name].dropna()
        if not len(oos_returns):
            continue
        s = summary_stats(oos_returns, rf)
        rows.append({
            "strategy": name,
            "oos_cagr": s["cagr"],
            "oos_vol": s["ann_vol"],
            "oos_sharpe": s["sharpe"],
            "oos_mdd": s["max_drawdown"],
            "total_tx_cost_pct": float(results["costs"][name].sum()),
        })
    results["summary"] = pd.DataFrame(rows).set_index("strategy")
    return results
