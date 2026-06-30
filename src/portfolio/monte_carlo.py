"""Monte Carlo portfolio simulator. Dirichlet sampling gives better simplex coverage
than uniform-then-normalize (which collapses toward 1/N as n grows)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_portfolios(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    n: int = 15000,
    alpha: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate n random long-only portfolios via Dirichlet(alpha).

    alpha<1 favors concentrated portfolios; alpha=1 is uniform on simplex; alpha>1 concentrates near 1/N.
    Returns DataFrame with columns: ret, vol, sharpe, w_<ticker>...
    """
    rng = np.random.default_rng(seed)
    tickers = list(mu.index)
    k = len(tickers)
    weights = rng.dirichlet(np.full(k, alpha), size=n)  # shape (n, k)

    mu_arr = mu.values
    cov_arr = cov.values

    rets = weights @ mu_arr
    # vol per portfolio via einsum: sqrt(w' Σ w)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_arr, weights))
    sharpes = (rets - rf) / np.where(vols < 1e-9, np.nan, vols)

    out = pd.DataFrame(weights, columns=[f"w_{t}" for t in tickers])
    out.insert(0, "sharpe", sharpes)
    out.insert(0, "vol", vols)
    out.insert(0, "ret", rets)
    return out


def extract_frontier(sim: pd.DataFrame, n_bins: int = 50) -> pd.DataFrame:
    """Empirical efficient frontier: bin by vol, take max-return portfolio per bin."""
    bins = pd.cut(sim["vol"], bins=n_bins)
    idx = sim.groupby(bins, observed=True)["ret"].idxmax().dropna().astype(int)
    return sim.loc[idx].sort_values("vol").reset_index(drop=True)
