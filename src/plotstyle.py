"""Shared chart style, so every notebook's charts look the same.

Use it like this:
    import plotstyle as ps
    ps.apply_style()
    fig, ax = plt.subplots()
    ...
    ps.title(ax, "Main message", "What is plotted, and for which flights")
    ps.save(fig, "chart_name")   # writes reports/figures/chart_name.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Colors. One accent color for the subject of the chart; gray for context.
SURFACE = "#fcfcfb"   # chart background
INK = "#0b0b0b"       # main text
INK_2 = "#52514e"     # secondary text
MUTED = "#898781"     # axis labels, context lines
GRID = "#e1e0d9"      # gridlines
AXIS = "#c3c2b7"      # baseline, context bars
BLUE = "#2a78d6"      # the accent: Delta, or the single series
RED = "#e34948"       # the opposite of blue in up/down charts (red raises, blue lowers)

# Light-to-dark blue scale for heatmaps (more = darker).
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
     "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def apply_style() -> None:
    """Set matplotlib defaults: quiet axes, thin gridlines, readable text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.constrained_layout.use": True,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    })


def title(ax, main: str, sub: str | None = None) -> None:
    """Left-aligned title with an optional smaller line under it."""
    ax.set_title(main, loc="left", fontsize=13, fontweight="bold", color=INK, pad=26 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom", ha="left")


def barh_rounded(ax, positions, widths, colors, height: float = 0.6, radius_pt: float = 3) -> None:
    """Horizontal bars that start square at zero and end with slightly rounded corners.

    Negative widths draw to the left of zero, for charts that show pushes up and down.

    Set the axis limits and tick labels BEFORE calling this, so the corner size is right.
    `colors` is one color or a list with one color per bar.
    """
    fig = ax.figure
    fig.canvas.draw()  # settle the layout so pixel sizes are final
    inv = ax.transData.inverted()
    r_px = radius_pt * fig.dpi / 72
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (r_px, r_px)])
    rx_full, ry_full = abs(x1 - x0), abs(y1 - y0)
    if isinstance(colors, str):
        colors = [colors] * len(widths)

    for pos, width, color in zip(positions, widths, colors):
        if width == 0:
            continue
        direction = 1 if width > 0 else -1          # negative bars grow to the left
        rx, ry = direction * min(rx_full, abs(width)), min(ry_full, height / 2)
        bottom, top = pos - height / 2, pos + height / 2
        verts = [
            (0, bottom), (width - rx, bottom),
            (width, bottom), (width, bottom + ry),
            (width, top - ry),
            (width, top), (width - rx, top),
            (0, top), (0, bottom),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def save(fig, name: str) -> Path:
    """Save the chart as reports/figures/<name>.png and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path
