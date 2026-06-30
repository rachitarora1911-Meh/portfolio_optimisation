"""Price + market-cap data layer with parquet caching.

Cache contract: keyed by market label. Re-runs read from parquet; only deltas hit yfinance.
All prices stored as adjusted close, business-day index, local currency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

MARKET_CAP_TTL_DAYS = 30


def _cache_paths(label: str) -> tuple[Path, Path]:
    return DATA_DIR / f"{label}_prices.parquet", DATA_DIR / f"{label}_prices.meta.json"


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2, default=str))


def _is_cache_valid(meta: dict, tickers: list[str], start: str, end: str) -> bool:
    if not meta:
        return False
    cached = set(meta.get("tickers", []))
    if not set(tickers).issubset(cached):
        return False
    if pd.Timestamp(meta["start"]) > pd.Timestamp(start):
        return False
    if pd.Timestamp(meta["end"]) < pd.Timestamp(end):
        return False
    return True


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_label: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted-close prices for tickers in [start, end]. Cached to parquet by label."""
    prices_path, meta_path = _cache_paths(cache_label)
    meta = _read_meta(meta_path)

    if not force_refresh and prices_path.exists() and _is_cache_valid(meta, tickers, start, end):
        logger.info("Cache hit: %s", cache_label)
        df = pd.read_parquet(prices_path)
        return df.loc[start:end, tickers].copy()

    logger.info("Downloading %d tickers from yfinance: %s ... %s", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        # Standard multi-ticker layout: (ticker, field).
        df = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().asfreq("B").ffill(limit=2)

    missing_pct = df.isna().mean()
    bad = missing_pct[missing_pct > 0.05].index.tolist()
    if bad:
        logger.warning("Dropping tickers with >5%% missing: %s", bad)
        df = df.drop(columns=bad)

    df.to_parquet(prices_path)
    _write_meta(
        meta_path,
        {
            "tickers": list(df.columns),
            "start": start,
            "end": end,
            "downloaded_at": datetime.now().isoformat(),
        },
    )
    return df.loc[start:end].copy()


def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Daily returns. 'log' for additivity in compounding; 'simple' for portfolio aggregation."""
    if kind == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    if kind == "simple":
        return prices.pct_change().dropna(how="all")
    raise ValueError(f"Unknown return kind: {kind}")


def get_market_caps(tickers: list[str], cache_label: str) -> pd.Series:
    """Market caps for BL implied prior. 30-day TTL — caps drift slowly."""
    cache_path = DATA_DIR / f"{cache_label}_caps.parquet"
    if cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(days=MARKET_CAP_TTL_DAYS):
            cached = pd.read_parquet(cache_path)["market_cap"]
            if set(tickers).issubset(cached.index):
                return cached.loc[tickers]

    logger.info("Fetching market caps for %d tickers", len(tickers))
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap") or info.get("market_cap")
            if mc:
                caps[t] = float(mc)
        except Exception as e:
            logger.warning("Failed cap fetch for %s: %s", t, e)
    s = pd.Series(caps, name="market_cap")
    # Fallback: if any tickers missing, fill with median to keep BL solvable.
    missing = [t for t in tickers if t not in s.index]
    if missing:
        fill = s.median() if len(s) else 1e10
        logger.warning("Filling %d missing market caps with median %.2e", len(missing), fill)
        for t in missing:
            s[t] = fill
    s = s.loc[tickers]
    s.to_frame().to_parquet(cache_path)
    return s


def load_benchmark(symbol: str, start: str, end: str, cache_label: str = "benchmarks") -> pd.Series:
    """Single-ticker benchmark series, cached separately."""
    df = download_prices([symbol], start, end, cache_label=f"{cache_label}_{symbol.replace('^','idx').replace('.','_')}")
    return df[symbol]
