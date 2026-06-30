"""Black-Litterman invariants: posterior collapses to prior when confidence -> 0,
and approaches view-implied returns when confidence -> 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.config import View
from portfolio.black_litterman import build_omega_idzorek, build_picking_matrix, market_implied_prior, posterior_returns


@pytest.fixture
def toy_market():
    tickers = ["A", "B", "C"]
    caps = pd.Series([1000.0, 500.0, 250.0], index=tickers)
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00],
         [0.01, 0.09, 0.00],
         [0.00, 0.00, 0.025]],
        index=tickers, columns=tickers,
    )
    return tickers, caps, cov


def test_implied_prior_signs_with_caps(toy_market):
    tickers, caps, cov = toy_market
    pi = market_implied_prior(caps, cov, risk_aversion=2.5, rf=0.0)
    # Reverse CAPM yields positive expected returns for positive risk + positive caps
    assert (pi > 0).all()
    # High-cap, high-vol asset should have higher implied return than low-cap low-vol
    # (Here B has highest vol but lower cap than A; check directional sanity)
    assert pi["A"] > 0


def test_posterior_collapses_to_prior_under_low_confidence(toy_market):
    tickers, caps, cov = toy_market
    prior = market_implied_prior(caps, cov, risk_aversion=2.5, rf=0.0)
    views = [View(name="test", type="relative", expected_return=0.05, confidence=0.001,
                  rationale="", weights={"A": 1.0, "B": -1.0})]
    P, Q = build_picking_matrix(views, tickers)
    omega = build_omega_idzorek(views, P, cov, tau=0.05)
    post = posterior_returns(prior, cov, P, Q, omega, tau=0.05)
    # With near-zero confidence, posterior must be very close to prior
    np.testing.assert_allclose(post.values, prior.values, atol=1e-3)


def test_posterior_moves_toward_view_under_high_confidence(toy_market):
    tickers, caps, cov = toy_market
    prior = market_implied_prior(caps, cov, risk_aversion=2.5, rf=0.0)
    # View says A - B = 10%; high confidence -> posterior spread between A and B should be near 10%
    views = [View(name="test", type="relative", expected_return=0.10, confidence=0.999,
                  rationale="", weights={"A": 1.0, "B": -1.0})]
    P, Q = build_picking_matrix(views, tickers)
    omega = build_omega_idzorek(views, P, cov, tau=0.05)
    post = posterior_returns(prior, cov, P, Q, omega, tau=0.05)
    spread = post["A"] - post["B"]
    assert abs(spread - 0.10) < 0.02


def test_picking_matrix_row_assignment(toy_market):
    tickers, _, _ = toy_market
    views = [
        View(name="v1", type="relative", expected_return=0.03, confidence=0.5,
             rationale="", weights={"A": 0.5, "B": 0.5, "C": -1.0}),
    ]
    P, Q = build_picking_matrix(views, tickers)
    assert P.shape == (1, 3)
    assert P[0, 0] == 0.5 and P[0, 1] == 0.5 and P[0, 2] == -1.0
    assert Q[0] == 0.03
