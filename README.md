# Portfolio Optimization — Client-Advisory Across Nifty 50 & S&P 500

**Author:** Rachit Arora · Personal project

Portfolio optimization study building three client-risk-profile allocations (Conservative / Balanced / Aggressive) across two equity universes (Nifty 50 + S&P 500). Combines **15,000-portfolio Monte Carlo**, **SLSQP optimization**, **Black-Litterman with documented investor views**, and a **walk-forward backtest with realistic transaction costs**.

The project is **notebook-centric**: `notebooks/01_full_analysis.ipynb` is the single narrative deliverable — it imports a thin, tested core library (`src/portfolio/`) and walks through every method with inline charts. No dashboards, no PDF generators, no export plumbing — just the concepts and the code that implements them.

---

## Problem

Modern Portfolio Theory's classic max-Sharpe solution is notoriously unstable — small changes in estimated expected returns produce wildly different portfolios (Michaud 1989). This project frames optimization through a **client-advisory lens** and addresses estimation noise via:

1. **Per-asset weight cap (40%)** — prevents corner solutions that overfit estimation noise.
2. **Black-Litterman blending** — anchors expected returns to market equilibrium and only deviates where there is a documented investor view with a stated confidence.
3. **Walk-forward out-of-sample backtest** — separates in-sample fit from realized performance.

## Methods

| Method | Module | Purpose |
|---|---|---|
| Monte Carlo (Dirichlet, 15,000 portfolios) | `src/portfolio/monte_carlo.py` | Visualize feasible set; sanity-check SLSQP optimums |
| SLSQP optimization | `src/portfolio/optimize.py` | Max-Sharpe, min-variance, max-return @ vol cap |
| Black-Litterman | `src/portfolio/black_litterman.py` | Implied prior + Idzorek omega + posterior μ |
| Walk-forward backtest | `src/portfolio/backtest.py` | 5y lookback, quarterly rebalance, 10/5 bps tx costs |

## Universe & Configuration

- **Nifty 50 subset (India)**: 12 large-caps across IT, Banking (Private + PSU), Energy, Consumer Staples, Industrials, Telecom, Materials. RF = 10y G-Sec ~6.9%.
- **S&P 500 subset (US)**: 12 large-caps across Mega-cap Tech, Financials, Healthcare, Energy, Consumer Staples / Discretionary. RF = 10y UST ~4.3%.

All settings in `config/markets.yaml`. Investor views in `config/views.yaml`.

## Results (Out-of-Sample, 2024-07 to 2026-06, ~2 years, quarterly rebalance)

| Market | Strategy | CAGR | Vol | Sharpe | Max DD | Total TxCost |
|---|---|---:|---:|---:|---:|---:|
| US | Conservative | 9.7% | 10.7% | 0.51 | -10.6% | 0.10% |
| US | Balanced | 14.7% | 18.4% | 0.61 | -23.8% | 0.10% |
| US | **Aggressive** | **19.3%** | 18.5% | **0.82** | -20.9% | 0.10% |
| US | Equal-Weight | 12.0% | 12.0% | 0.65 | -13.5% | — |
| US | SPY benchmark | 17.7% | 16.7% | 0.81 | -18.8% | — |
| India | Conservative | -11.8% | 11.6% | -1.62 | -26.8% | 0.10% |
| India | Balanced | -0.8% | 13.6% | -0.50 | -15.7% | 0.20% |
| India | **Aggressive** | **6.7%** | 15.3% | **0.05** | -17.1% | 0.20% |
| India | Equal-Weight | -5.3% | 11.8% | -0.98 | -18.0% | — |
| India | NIFTYBEES benchmark | 1.0% | 12.2% | -0.41 | -15.2% | — |

**Read:** Aggressive profile beats the index benchmark in both markets (US 19.3% > SPY 17.7%; India 6.7% > NIFTY 1.0%). Conservative profile suffers when low-volatility names underperform regime-shift periods (India consumer-staples re-derate 2024-25). Reporting is strictly out-of-sample — no in-sample Sharpe figures that would not be reproducible live.

## Repo Structure

```
raichu/
├── config/                        markets.yaml, views.yaml
├── src/portfolio/                 Core library
│   ├── config.py                  Market + View dataclasses, YAML loaders
│   ├── data.py                    yfinance + parquet caching
│   ├── stats.py                   Return/vol/Sharpe/drawdown
│   ├── optimize.py                SLSQP optimizers + 3-profile builder
│   ├── monte_carlo.py             Dirichlet simulator + frontier extraction
│   ├── black_litterman.py         Implied prior + Idzorek omega + posterior
│   ├── backtest.py                Walk-forward with no-look-ahead guard
│   └── reporting.py               Matplotlib charts for the notebook
├── notebooks/01_full_analysis.ipynb   The deliverable: full narrative + charts
└── tests/                         pytest suite (concept modules)
```

## How to Run

### Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Jupyter notebook (the deliverable)
```bash
.venv/bin/jupyter notebook notebooks/01_full_analysis.ipynb
```
Downloads data (cached to `data/raw/`), runs Monte Carlo + SLSQP + Black-Litterman + walk-forward backtest for both markets, and renders every chart inline with explanatory markdown.

### Tests
```bash
.venv/bin/python -m pytest tests/ -v
```

## Method Rationale

- **Monte Carlo (15k)** — sanity check on SLSQP; Dirichlet sampling gives better coverage of the simplex than uniform-then-normalize, which collapses toward 1/N as n grows.
- **SLSQP max-Sharpe** — handles sum-to-1 + per-asset cap natively. Cap = 40% to prevent overfit to noisy mu estimates.
- **Min-variance (Conservative)** — depends only on Σ, which is far more stable than mu (Merton 1980) — the right choice for capital-preservation profiles.
- **Black-Litterman** — anchors to reverse-CAPM market equilibrium and only deviates where there is a stated view. Idzorek omega translates "60% confident" to a defensible omega value. Without BL, sample-mu portfolios concentrate >40% in the highest-return asset (e.g. NVDA), which is fragile.
- **Walk-forward, quarterly** — balances responsiveness vs estimation stability. Monthly rebalancing 10× the tx costs for marginal Sharpe gain at this universe size.
- **Equal-weight comparison** — DeMiguel-Garlappi-Uppal (2009) showed 1/N is hard to beat OOS. If optimization can't beat equal-weight after costs, the complexity isn't earning its keep. (Aggressive profile passes this test in both markets.)
- **Three client profiles** — advisory is matching risk tolerance to allocation, not "here's the optimal portfolio." Same universe, three optimization objectives, three recommendations.
- **Why 10y G-Sec / 10y UST as RF** — risk-free rate should match portfolio duration. T-bill rate is shorter-duration and would flip Sharpe rankings between profiles.

## Limitations

- **Survivorship bias** — using current index members; delisted names not represented. Inflates returns ~50-100 bps/yr.
- **Look-ahead bias** — controlled by hard assertion in walk-forward loop (`tests/test_backtest.py::test_look_ahead_assertion_fires_on_invalid_date`).
- **Small OOS sample** — 8 quarterly rebalances over 2 years; Sharpe CI is wide. Don't over-claim.
- **Stationarity** — rolling-window mu and Σ assume regime persistence. Regime breaks (COVID 2020, rates 2022, AI capex 2025) violate this.
- **No cross-market FX hedging** — India and US run separately in local currency; combined book is out of scope.
- **Market-cap recency for BL prior** — current caps applied to historical periods. Minor look-ahead, documented in code.

## Tech Stack

Python 3.11+ · numpy · pandas · scipy · yfinance · matplotlib · PyYAML · pyarrow · jupyter · pytest.

## References

- Black & Litterman (1992), *Global Portfolio Optimization*.
- Idzorek (2005), *A Step-By-Step Guide to the Black-Litterman Model*.
- DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification*.
- Michaud (1989), *The Markowitz Optimization Enigma*.
- Merton (1980), *On Estimating the Expected Return on the Market*.
