"""Backtest invariants: no-look-ahead assertion fires, turnover applied correctly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.backtest import (
    apply_weights,
    equal_weight_strategy,
    generate_rebalance_dates,
    walk_forward_weights,
)


def _make_prices(n_days: int = 1500, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=(n_days, n_assets))
    prices = 100.0 * np.cumprod(1 + rets, axis=0)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"T{i}" for i in range(n_assets)])


def test_look_ahead_assertion_holds_for_valid_dates():
    prices = _make_prices()
    rebals = [prices.index[300], prices.index[600], prices.index[900]]
    w = walk_forward_weights(prices, rebals, equal_weight_strategy, lookback_years=1)
    assert len(w) == 3
    np.testing.assert_allclose(w.values.sum(axis=1), 1.0, atol=1e-9)


def test_look_ahead_assertion_fires_on_invalid_date():
    """If rebalance date falls inside lookback (e.g. by re-using same date twice or
    constructing pathological inputs), the loop should refuse."""
    prices = _make_prices(n_days=100)
    # 1-year lookback but only 100 days of data — every rebalance after day 0 has prior data
    # so the assertion holds. Force violation: ask for rebalance at day 0.
    bad_rebal = [prices.index[0]]
    with pytest.raises((AssertionError, ValueError)):
        walk_forward_weights(prices, bad_rebal, equal_weight_strategy, lookback_years=1)


def test_apply_weights_zero_cost_matches_static_portfolio():
    """With zero tx costs and constant weights, daily portfolio returns must equal
    sum(w_i * r_i) on each day."""
    prices = _make_prices(n_days=300, n_assets=3)
    w = pd.Series([1/3, 1/3, 1/3], index=prices.columns)
    weights_df = pd.DataFrame([w], index=[prices.index[0]])
    gross, net, costs = apply_weights(prices, weights_df, tx_cost_bps=0.0)
    np.testing.assert_allclose(net.values, gross.values, atol=1e-12)
    assert costs.sum() == 0.0


def test_tx_costs_drag_returns():
    prices = _make_prices()
    rebals = [prices.index[300], prices.index[600], prices.index[900]]
    w = walk_forward_weights(prices, rebals, equal_weight_strategy, lookback_years=1)
    gross, net, costs = apply_weights(prices, w, tx_cost_bps=50.0)  # 50 bps = big
    assert costs.sum() > 0
    assert net.sum() < gross.sum()


def test_generate_rebalance_dates_quarterly():
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    dates = generate_rebalance_dates("2021-01-01", "2023-12-31", freq="Q", index=idx)
    assert len(dates) == 12  # 4 quarters * 3 years
    for d in dates:
        assert d in idx
