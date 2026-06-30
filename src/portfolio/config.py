"""Market + View configuration. YAML-backed dataclasses, single source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(frozen=True)
class Market:
    key: str
    name: str
    currency: str
    benchmark: str
    benchmark_etf: str
    rf_rate: float
    tx_cost_bps: float
    start_date: str
    end_date: str
    tickers: list[str]
    sectors: dict[str, str]


@dataclass(frozen=True)
class View:
    name: str
    type: str  # "relative" or "absolute"
    expected_return: float
    confidence: float
    rationale: str
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rebalance_freq: str
    lookback_years: int


@dataclass(frozen=True)
class OptimizeConfig:
    weight_min: float
    weight_max: float
    aggressive_vol_cap: float


@dataclass(frozen=True)
class MonteCarloConfig:
    n_simulations: int
    dirichlet_alpha: float
    seed: int


@dataclass(frozen=True)
class BlackLittermanConfig:
    tau: float
    risk_aversion: float


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_markets(path: Path | None = None) -> dict[str, Market]:
    path = path or CONFIG_DIR / "markets.yaml"
    raw = _load_yaml(path)
    markets = {}
    for key in ("india", "us"):
        m = raw[key]
        markets[key] = Market(
            key=key,
            name=m["name"],
            currency=m["currency"],
            benchmark=m["benchmark"],
            benchmark_etf=m["benchmark_etf"],
            rf_rate=float(m["rf_rate"]),
            tx_cost_bps=float(m["tx_cost_bps"]),
            start_date=m["start_date"],
            end_date=m["end_date"],
            tickers=list(m["tickers"]),
            sectors=dict(m["sectors"]),
        )
    return markets


def load_views(market_key: str, path: Path | None = None) -> list[View]:
    path = path or CONFIG_DIR / "views.yaml"
    raw = _load_yaml(path)
    out = []
    for v in raw[market_key]:
        out.append(
            View(
                name=v["name"],
                type=v["type"],
                expected_return=float(v["expected_return"]),
                confidence=float(v["confidence"]),
                rationale=v.get("rationale", "").strip(),
                weights={k: float(val) for k, val in v["weights"].items()},
            )
        )
    return out


def load_backtest_config(path: Path | None = None) -> BacktestConfig:
    path = path or CONFIG_DIR / "markets.yaml"
    raw = _load_yaml(path)["backtest"]
    return BacktestConfig(
        train_start=raw["train_start"],
        train_end=raw["train_end"],
        test_start=raw["test_start"],
        test_end=raw["test_end"],
        rebalance_freq=raw["rebalance_freq"],
        lookback_years=int(raw["lookback_years"]),
    )


def load_optimize_config(path: Path | None = None) -> OptimizeConfig:
    path = path or CONFIG_DIR / "markets.yaml"
    raw = _load_yaml(path)["optimize"]
    return OptimizeConfig(
        weight_min=float(raw["weight_min"]),
        weight_max=float(raw["weight_max"]),
        aggressive_vol_cap=float(raw["aggressive_vol_cap"]),
    )


def load_mc_config(path: Path | None = None) -> MonteCarloConfig:
    path = path or CONFIG_DIR / "markets.yaml"
    raw = _load_yaml(path)["monte_carlo"]
    return MonteCarloConfig(
        n_simulations=int(raw["n_simulations"]),
        dirichlet_alpha=float(raw["dirichlet_alpha"]),
        seed=int(raw["seed"]),
    )


def load_bl_config(path: Path | None = None) -> BlackLittermanConfig:
    path = path or CONFIG_DIR / "markets.yaml"
    raw = _load_yaml(path)["black_litterman"]
    return BlackLittermanConfig(
        tau=float(raw["tau"]),
        risk_aversion=float(raw["risk_aversion"]),
    )
