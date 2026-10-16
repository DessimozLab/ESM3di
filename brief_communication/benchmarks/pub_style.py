"""Shared matplotlib style for the ESM3Di benchmark figures.

Import this in every benchmark notebook so all figures use the same fonts,
sizes, colours and export settings:

    import sys; sys.path.insert(0, "..")   # from quality/ or speed/
    from pub_style import *
    use_pub_style()
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Figure widths (Nature-style column widths) -------------------------------
MM = 1 / 25.4  # matplotlib sizes are in inches
WIDTH_SINGLE = 89 * MM
WIDTH_ONEHALF = 120 * MM
WIDTH_DOUBLE = 183 * MM

# --- Method colours: a method always has the same colour in every figure -------
# ESM3Di / ProstT5 are the Okabe-Ito vermillion / blue pair: distinguishable under
# all common colour-vision deficiencies and dark enough for coloured labels on white.
METHOD_COLOURS = {
    "ESM3Di": "#D55E00",
    "ProstT5": "#0072B2",
    "Baseline": "#8C8C8C",
    "AF3": "#2B2B2B",
    # Foldseek on real structures is the structural reference, like AF3; MMseqs2
    # (sequence only) is a lower reference, like the random baseline.
    "Foldseek": "#2B2B2B",
    "MMseqs2": "#8C8C8C",
    # ColabFold predicts full structures, the structure-based route to 3Di
    "ColabFold": "#2B2B2B",
}
TIE_COLOUR = "#999999"
MUTED_TEXT = "#595959"  # secondary annotations (n, ties)

# Line widths in points
AXIS_LINEWIDTH = 0.6
DATA_LINEWIDTH = 1.2
REFERENCE_LINEWIDTH = 0.7


def use_pub_style(base_size=7):
    """Apply the publication style to all following matplotlib figures."""
    mpl.rcParams.update({
        # Fonts: Arial/Helvetica if installed, else the metric-compatible Nimbus Sans
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "mathtext.default": "regular",
        # Axes: left and bottom spines only, ticks pointing out
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": AXIS_LINEWIDTH,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": AXIS_LINEWIDTH,
        "ytick.major.width": AXIS_LINEWIDTH,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.frameon": False,
        "lines.linewidth": DATA_LINEWIDTH,
        # Export: editable text in PDFs, white background
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "figure.facecolor": "white",
    })


def add_panel_labels(axes, labels="abcdefgh", x=-0.22, y=1.04):
    """Bold lower-case panel letters (a, b, c, ...) at the top-left of each axis."""
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left")


def save_pub(fig, name, outdir="figures"):
    """Save `fig` as <outdir>/<name>.pdf (vector) and <name>.png (600 dpi)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.pdf", outdir / f"{name}.png"]
    for path in paths:
        fig.savefig(path)
    return paths
