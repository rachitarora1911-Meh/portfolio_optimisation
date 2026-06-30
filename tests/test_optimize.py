"""Optimizer invariants: weights sum to 1, respect bounds, recover known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.optimize import (
    build_three_profiles,
    optimize_max_return_at_vol,
    optimize_max_sharpe,
    optimize_min_variance,
)


@pytest.fixture
def toy_mu_cov():
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.20, 0.05, 0.15], index=tickers)
    rng = np.random.default_rng(0)
    cov = pd.DataFrame(
        np.array([
            [0.04, 0.01, 0.00, 0.005],
            [0.01, 0.09, 0.00, 0.010],
            [0.00, 0.00, 0.01, 0.000],
            [0.005, 0.010, 0.00, 0.025],
        ]),
        index=tickers, columns=tickers,
    )
    return mu, cov


def test_max_sharpe_sum_to_one_and_bounds(toy_mu_cov):
    mu, cov = toy_mu_cov
    w = optimize_max_sharpe(mu, cov, rf=0.02, w_min=0.0, w_max=0.5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()


def test_min_var_sum_to_one_and_no_mu_dependence(toy_mu_cov):
    mu, cov = toy_mu_cov
    w = optimize_min_variance(cov, w_min=0.0, w_max=0.5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()
    # Result must be independent of mu (function signature doesn't take mu)
    w2 = optimize_min_variance(cov, w_min=0.0, w_max=0.5)
    np.testing.assert_allclose(w.values, w2.values)


def test_max_return_at_vol_respects_cap(toy_mu_cov):
    mu, cov = toy_mu_cov
    vol_cap = 0.15
    w = optimize_max_return_at_vol(mu, cov, vol_cap=vol_cap, w_min=0.0, w_max=0.5)
    vol = float(np.sqrt(w.values @ cov.values @ w.values))
    assert vol <= vol_cap + 1e-6


def test_build_three_profiles_distinct_or_equal(toy_mu_cov):
    mu, cov = toy_mu_cov
    df = build_three_profiles(mu, cov, rf=0.02, vol_cap=0.20, w_min=0.0, w_max=0.5)
    assert set(df.columns) == {"conservative", "balanced", "aggressive"}
    for col in df.columns:
        assert abs(df[col].sum() - 1.0) < 1e-6


def test_bounds_clip_tight_cap(toy_mu_cov):
    mu, cov = toy_mu_cov
    w = optimize_max_sharpe(mu, cov, rf=0.02, w_min=0.0, w_max=0.30)
    assert (w <= 0.30 + 1e-6).all()
