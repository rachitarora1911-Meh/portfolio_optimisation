"""Consulting-deck visual theme: palette + matplotlib setup.

Palette: navy primary, slate neutrals, coral accent. Generous whitespace, thin gridlines,
serif-free typography, no chartjunk. One call to apply_matplotlib_theme() configures
rcParams for any downstream plot.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- palette

NAVY = "#0B1F3A"        # primary text / headers
SLATE_DARK = "#374151"  # axis labels
SLATE = "#6B7280"       # secondary text, tick labels
SLATE_LIGHT = "#9CA3AF" # rules, light separators
HAIRLINE = "#E5E7EB"    # gridlines
BG = "#FFFFFF"          # page background
BG_ALT = "#F8F9FB"      # panel / KPI strip background

CORAL = "#E15A4C"       # accent / highlight
TEAL = "#0E8A8A"        # second accent
GOLD = "#D5A93C"        # tertiary accent
GREEN = "#2C8A4E"       # positive
RED = "#C9453B"         # negative

# Strategy / profile color map (consult-deck convention: highlight one, mute the rest)
PROFILE_COLORS = {
    "conservative": "#5A7AA5",     # muted blue
    "balanced": CORAL,             # highlight
    "aggressive": "#A04CCB",       # plum
}

STRATEGY_COLORS = {
    "conservative": "#5A7AA5",
    "balanced": CORAL,
    "aggressive": "#A04CCB",
    "equal_weight": SLATE_LIGHT,
    "benchmark": NAVY,
}

# Sequential palette for MC scatter (low Sharpe -> high Sharpe)
SEQUENTIAL = ["#DCE3EE", "#9FB4D2", "#4A6D9B", "#0B1F3A"]

# Categorical palette for sectors / tickers
CATEGORICAL = [
    "#0B1F3A", "#E15A4C", "#0E8A8A", "#D5A93C",
    "#5A7AA5", "#A04CCB", "#2C8A4E", "#8B5E3C",
    "#445E80", "#C9453B", "#7E9CB8", "#B36B2E",
]


def apply_matplotlib_theme():
    """Set global rcParams. Call once at process start (or before any figure)."""
    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.titlecolor": NAVY,
        "axes.labelsize": 9,
        "axes.labelcolor": SLATE_DARK,
        "axes.labelpad": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "legend.fontsize": 8,
        "legend.frameon": False,

        # Spines and ticks
        "axes.edgecolor": SLATE_LIGHT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Grid: subtle horizontal only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": HAIRLINE,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,

        # Background
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,

        # Lines and patches
        "lines.linewidth": 1.6,
        "patch.linewidth": 0.5,
        "patch.edgecolor": BG,

        # Default cycler -> categorical palette
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })
