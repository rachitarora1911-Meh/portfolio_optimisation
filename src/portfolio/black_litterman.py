"""Black-Litterman: market-implied prior + investor views + Idzorek omega -> posterior mu.

Rationale: Markowitz with sample-mean mu is unstable; small mu perturbations
flip weights dramatically (Michaud 1989). BL anchors to market equilibrium (reverse CAPM)
and only deviates where there is a documented view, weighted by a stated confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import View


def market_implied_prior(
    market_caps: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 2.5,
    rf: float = 0.0,
) -> pd.Series:
    """Reverse-optimize equilibrium expected returns: pi = rf + lambda * Σ * w_mkt.

    Market weights = cap-weights normalized. Risk aversion ~2.5 typical for equity markets.
    """
    w_mkt = (market_caps / market_caps.sum()).reindex(cov.index)
    pi = rf + risk_aversion * cov.values @ w_mkt.values
    return pd.Series(pi, index=cov.index, name="implied_prior")


def build_picking_matrix(views: list[View], tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Translate View list to P matrix (k_views x n_assets) and Q vector (k_views)."""
    P = np.zeros((len(views), len(tickers)))
    Q = np.zeros(len(views))
    idx = {t: i for i, t in enumerate(tickers)}
    for k, v in enumerate(views):
        for ticker, w in v.weights.items():
            if ticker not in idx:
                raise KeyError(f"View ticker {ticker} not in universe")
            P[k, idx[ticker]] = w
        Q[k] = v.expected_return
    return P, Q


def build_omega_idzorek(
    views: list[View],
    P: np.ndarray,
    cov: pd.DataFrame,
    tau: float = 0.05,
) -> np.ndarray:
    """Idzorek-style omega from confidence levels.

    omega_ii = (1 - conf) / conf * (p_i' Σ p_i * tau)

    High confidence (conf -> 1) -> small omega -> view dominates posterior.
    Low confidence (conf -> 0) -> large omega -> posterior collapses to prior.
    """
    cov_arr = cov.values
    diag = np.zeros(len(views))
    for i, v in enumerate(views):
        conf = max(min(v.confidence, 0.999), 0.001)  # clamp to avoid div by 0
        pvp = float(P[i] @ cov_arr @ P[i].T)
        diag[i] = (1.0 - conf) / conf * pvp * tau
    return np.diag(diag)


def posterior_returns(
    prior: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected returns.

    mu_BL = [ (tau Σ)^-1 + P' Ω^-1 P ]^-1 * [ (tau Σ)^-1 pi + P' Ω^-1 Q ]
    """
    cov_arr = cov.values
    tau_sigma_inv = np.linalg.inv(tau * cov_arr)
    omega_inv = np.linalg.inv(omega)

    A = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ prior.values + P.T @ omega_inv @ Q
    mu_bl = np.linalg.solve(A, b)
    return pd.Series(mu_bl, index=cov.index, name="bl_posterior")


def run_black_litterman(
    market_caps: pd.Series,
    cov: pd.DataFrame,
    views: list[View],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    rf: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Convenience wrapper: returns (prior, posterior).

    cov should be annualized. Returns annualized mu series.
    """
    prior = market_implied_prior(market_caps, cov, risk_aversion, rf)
    if not views:
        return prior, prior.copy().rename("bl_posterior")
    P, Q = build_picking_matrix(views, list(cov.index))
    omega = build_omega_idzorek(views, P, cov, tau)
    posterior = posterior_returns(prior, cov, P, Q, omega, tau)
    return prior, posterior
