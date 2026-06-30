"""Matplotlib charts for the analysis notebook.

Visual system from theme.py: navy/slate neutrals, coral accent, generous whitespace,
horizontal-grid-only, no chartjunk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import theme

theme.apply_matplotlib_theme()


# ============================================================ CHARTS

def plot_efficient_frontier(
    sim: pd.DataFrame,
    profiles: dict[str, dict],
    title: str = "Efficient frontier",
    subtitle: str = "15,000 Monte Carlo portfolios with three optimal allocations",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sc = ax.scatter(
        sim["vol"] * 100, sim["ret"] * 100,
        c=sim["sharpe"], cmap=_make_cmap(theme.SEQUENTIAL),
        s=5, alpha=0.55, linewidth=0,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Sharpe ratio", color=theme.SLATE_DARK, fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=theme.SLATE, labelsize=8)

    for label in ("conservative", "balanced", "aggressive"):
        if label not in profiles:
            continue
        p = profiles[label]
        ax.scatter(
            [p["vol"] * 100], [p["ret"] * 100],
            s=130, color=theme.PROFILE_COLORS[label],
            edgecolor="white", linewidth=1.8, zorder=5,
        )
        ax.annotate(
            f"{label.title()}\nSR {p['sharpe']:.2f}",
            xy=(p["vol"] * 100, p["ret"] * 100),
            xytext=(10, 8), textcoords="offset points",
            fontsize=8, color=theme.NAVY, fontweight="bold",
        )

    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    _set_title(ax, title, subtitle)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_allocation_bar(
    weights: pd.Series,
    sectors: dict[str, str] | None = None,
    title: str = "Recommended allocation",
    subtitle: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal allocation bar — much cleaner than pie for >5 names."""
    w = weights[weights > 0.005].sort_values()
    sectors = sectors or {}
    sect_list = [sectors.get(t, "Other") for t in w.index]
    uniq_sectors = list(dict.fromkeys(sect_list))
    sector_color = {s: theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i, s in enumerate(uniq_sectors)}
    bar_colors = [sector_color[s] for s in sect_list]

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.35 * len(w) + 0.6)))
    y = np.arange(len(w))
    ax.barh(y, w.values * 100, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(w.index, fontsize=8)
    ax.set_xlabel("Weight (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", color=theme.HAIRLINE, linewidth=0.5)
    ax.grid(axis="y", visible=False)

    for i, val in enumerate(w.values):
        ax.text(val * 100 + 0.3, i, f"{val*100:.1f}%", va="center",
                fontsize=8, color=theme.SLATE_DARK)

    if sectors:
        handles = [Line2D([0], [0], marker="s", linestyle="", markersize=7,
                          markerfacecolor=sector_color[s], markeredgecolor="white", label=s)
                   for s in uniq_sectors]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=min(len(uniq_sectors), 4), fontsize=7,
                  frameon=False, columnspacing=1.4, handletextpad=0.4)
    _set_title(ax, title, subtitle)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "Cumulative return",
    subtitle: str = "Rebased to 100 at out-of-sample start",
    highlight: str = "balanced",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for label, s in curves.items():
        is_highlight = label == highlight
        color = theme.STRATEGY_COLORS.get(label, theme.SLATE)
        ax.plot(
            s.index, s.values,
            color=color,
            linewidth=2.2 if is_highlight else 1.1,
            alpha=1.0 if is_highlight else 0.7,
            label=label.replace("_", " ").title(),
            zorder=3 if is_highlight else 2,
        )

    last_date = max(s.index.max() for s in curves.values())
    # Add right-side padding so end-of-line labels don't clip
    span = last_date - min(s.index.min() for s in curves.values())
    ax.set_xlim(min(s.index.min() for s in curves.values()), last_date + span * 0.07)
    for label, s in curves.items():
        end_val = s.iloc[-1]
        color = theme.STRATEGY_COLORS.get(label, theme.SLATE)
        ax.text(last_date, end_val, f"  {end_val:.1f}",
                va="center", fontsize=8, color=color,
                fontweight="bold" if label == highlight else "normal")

    ax.set_ylabel("Index (base = 100)")
    ax.axhline(100, color=theme.SLATE_LIGHT, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", ncol=len(curves))
    _set_title(ax, title, subtitle)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_drawdown(
    drawdown: pd.Series,
    title: str = "Drawdown",
    subtitle: str = "Peak-to-trough decline of the Balanced profile",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    color=theme.CORAL, alpha=0.25, linewidth=0)
    ax.plot(drawdown.index, drawdown.values * 100, color=theme.CORAL, linewidth=1.0)
    ax.axhline(0, color=theme.SLATE_LIGHT, linewidth=0.5)
    mdd = drawdown.min()
    mdd_date = drawdown.idxmin()
    ax.scatter([mdd_date], [mdd * 100], s=40, color=theme.CORAL, zorder=5, edgecolor="white", linewidth=1.2)
    ax.annotate(f"Max DD {mdd*100:.1f}%", (mdd_date, mdd * 100),
                xytext=(8, -8), textcoords="offset points",
                fontsize=8, color=theme.CORAL, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    _set_title(ax, title, subtitle)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_bl_delta(
    prior: pd.Series,
    posterior: pd.Series,
    title: str = "Black-Litterman: prior vs posterior expected return",
    subtitle: str = "Posterior shrinks aggressive prior toward investor views",
    save_path: Path | None = None,
) -> plt.Figure:
    df = pd.DataFrame({"prior": prior * 100, "posterior": posterior * 100}).sort_values("posterior", ascending=False)
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.32 * n + 1.0)))
    y = np.arange(n)
    ax.hlines(y, df["prior"], df["posterior"], color=theme.SLATE_LIGHT, linewidth=1.0)
    ax.scatter(df["prior"], y, color=theme.SLATE, s=42, label="Implied prior (π)", zorder=4)
    ax.scatter(df["posterior"], y, color=theme.CORAL, s=58,
               edgecolor="white", linewidth=1.0, label="Posterior (μ_BL)", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Expected return (%)")
    ax.legend(loc="lower right")
    _set_title(ax, title, subtitle)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


# ============================================================ HELPERS

def _make_cmap(colors: list[str]):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", colors)


def _set_title(ax, title: str, subtitle: str = ""):
    """Title + subtitle, left-aligned, consulting-style."""
    if subtitle:
        ax.set_title(title, loc="left", fontsize=12, color=theme.NAVY, fontweight="bold", pad=22)
        ax.text(0, 1.03, subtitle, transform=ax.transAxes,
                fontsize=9, color=theme.SLATE, style="normal")
    else:
        ax.set_title(title, loc="left", fontsize=12, color=theme.NAVY, fontweight="bold")
