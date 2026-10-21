"""Plotting functions for the ESM3Di benchmark panels.

Shared by the benchmark notebooks (quality/, speed/) and the assembled figures
(paper/), so every panel is drawn by the same code wherever it appears.
Each function draws into a given matplotlib axes and takes its data as arguments.
"""

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pub_style import METHOD_COLOURS, MUTED_TEXT, REFERENCE_LINEWIDTH, TIE_COLOUR


# =============================================================================
# 3Di representation benchmark (quality/3di_alignment.ipynb)
# =============================================================================

BITSCORE_ARGS = dict(
    x_col="prostt5_bitscore", y_col="esm3di_bitscore",
    x_label="ProstT5 bitscore", y_label="ESM3Di bitscore", log_scale=True,
)
IDENTITY_ARGS = dict(
    x_col="prostt5_identity", y_col="esm3di_identity",
    x_label="ProstT5 3Di identity (%)", y_label="ESM3Di 3Di identity (%)",
)


def plot_method_scatter(ax, df, x_col, y_col, x_label, y_label, title=None, log_scale=False,
                        baseline=None):
    """One point per protein: ESM3Di (y) vs ProstT5 (x).

    Points are coloured by the method that scores higher; the share of proteins on
    each side of y = x is written in the matching triangle. `baseline` (e.g. the median
    identity of the random baseline) is drawn as dashed guides on both axes.
    """
    values = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if log_scale:
        # A bitscore of 0 cannot be shown on log axes
        values = values[(values[x_col] > 0) & (values[y_col] > 0)]
    x = values[x_col].to_numpy()
    y = values[y_col].to_numpy()

    groups = {
        "ESM3Di": (y > x, METHOD_COLOURS["ESM3Di"]),
        "ProstT5": (y < x, METHOD_COLOURS["ProstT5"]),
        "Tie": (y == x, TIE_COLOUR),
    }
    for mask, colour in groups.values():
        # Unclipped, so points at 0 or 100 % are drawn whole rather than cut at the axes edge
        ax.scatter(x[mask], y[mask], s=2, alpha=0.35, color=colour,
                   edgecolors="none", rasterized=True, clip_on=False)

    if log_scale:
        lims = (min(x.min(), y.min()) / 1.3, max(x.max(), y.max()) * 1.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
            axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
            axis.set_minor_formatter(ticker.NullFormatter())
    else:
        lims = (0, 100)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    ax.plot(lims, lims, color="0.2", lw=REFERENCE_LINEWIDTH, zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Win rate of each method, in its own triangle (upper left: ESM3Di, lower right: ProstT5)
    # (line offsets in points, so the spacing holds at any panel size)
    def corner_text(xy, text, dy, ha, va, **kwargs):
        ax.annotate(text, xy, xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                    ha=ha, va=va, **kwargs)

    # With baseline guides, the win-rate labels move clear of them: ESM3Di's right of the
    # vertical guide, ProstT5's above the horizontal one
    esm_x, prost_y, prost_dy = 0.04, 0.025, 7
    if baseline is not None:
        # Lighter than the y = x line, which carries the comparison; labelled along the
        # vertical guide in the empty strip left of it, in the guides' own grey
        guide_colour = "0.6"
        guide = dict(color=guide_colour, lw=0.5, ls=(0, (3, 2)), zorder=0.5)
        ax.axvline(baseline, **guide)
        ax.axhline(baseline, **guide)
        ax.annotate("Random baseline", (baseline, 0.98), xycoords=("data", "axes fraction"),
                    xytext=(-2, 0), textcoords="offset points", ha="right", va="top",
                    rotation=90, fontsize=5.5, color=guide_colour)
        position = ax.transLimits.transform((baseline, baseline))  # data -> axes fraction
        esm_x, prost_y, prost_dy = position[0] + 0.03, position[1] + 0.02, 0

    esm_mask, esm_colour = groups["ESM3Di"]
    corner_text((esm_x, 0.97), f"{100 * esm_mask.mean():.1f}%", 0, "left", "top",
                color=esm_colour, fontsize=8, fontweight="bold")
    corner_text((esm_x, 0.97), "ESM3Di higher", -9, "left", "top", color=esm_colour, fontsize=6)
    corner_text((0.97, 0.025), f"n = {len(x):,}; ties {100 * groups['Tie'][0].mean():.1f}%", 0,
                "right", "bottom", fontsize=5.5, color=MUTED_TEXT)
    # Stacked upwards from the n / ties line, or from the baseline guide
    prost_mask, prost_colour = groups["ProstT5"]
    corner_text((0.97, prost_y), "ProstT5 higher", prost_dy, "right", "bottom",
                color=prost_colour, fontsize=6)
    corner_text((0.97, prost_y), f"{100 * prost_mask.mean():.1f}%", prost_dy + 7, "right", "bottom",
                color=prost_colour, fontsize=8, fontweight="bold")
    return ax


def kde_reflected(values, grid, lower=0.0, upper=100.0, bw_adjust=0.8):
    """Gaussian KDE (Scott bandwidth) reflected at the bounds, so no density leaks past 0 or 100%."""
    bandwidth = bw_adjust * values.std(ddof=1) * len(values) ** (-1 / 5)
    mirrored = np.concatenate([values, 2 * lower - values, 2 * upper - values])
    density = np.zeros_like(grid)
    for chunk in np.array_split(mirrored, 20):
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bandwidth) ** 2).sum(axis=1)
    return density / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def plot_identity_density(ax, df, title=None):
    """Density of per-protein 3Di identity to AF3; each curve is labelled at its (dashed) median."""
    columns = {
        "ESM3Di": "esm3di_identity",
        "ProstT5": "prostt5_identity",
        "Baseline": "baseline_identity",
    }
    names = {"Baseline": "Random baseline"}

    grid = np.linspace(0, 100, 501)
    medians, densities, peak = {}, {}, 0.0
    for method, column in columns.items():
        values = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        colour = METHOD_COLOURS[method]
        density = kde_reflected(values, grid)
        densities[method] = density
        ax.plot(grid, density, color=colour, zorder=3 if method == "ESM3Di" else 2)
        medians[method] = np.median(values)
        peak = max(peak, density.max())

    # Transparent fill under each curve. ESM3Di and ProstT5 overlap almost entirely and
    # blending their fills gives a muddy grey, so their shared area takes the colour of
    # whichever curve is on top.
    esm, prost = densities["ESM3Di"], densities["ProstT5"]
    # (anti-aliasing off, or a hairline gap shows where the colour switches)
    fill = dict(alpha=0.15, lw=0, interpolate=True, antialiased=False)
    ax.fill_between(grid, esm, where=esm >= prost, color=METHOD_COLOURS["ESM3Di"], **fill)
    ax.fill_between(grid, prost, where=prost >= esm, color=METHOD_COLOURS["ProstT5"], **fill)
    ax.fill_between(grid, densities["Baseline"], color=METHOD_COLOURS["Baseline"], **fill)

    top = peak * 1.06
    ax.set_xlim(0, 100)
    ax.set_ylim(0, top)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    def median_label(method, x, y, ha):
        """Name and median (bold) as a two-line label with its top-left/right corner at (x, y)."""
        colour = METHOD_COLOURS[method]
        ax.text(x, y, names.get(method, method), color=colour, ha=ha, va="top", fontsize=6)
        ax.text(x, y - 0.075 * top, f"{medians[method]:.1f}%", color=colour, ha=ha, va="top",
                fontsize=7, fontweight="bold")

    dashed = dict(ls=(0, (3, 2)), lw=REFERENCE_LINEWIDTH, zorder=1)
    # The model labels sit in the empty space above their curves, on either side of the
    # close medians; their dashed median lines run up to the labels.
    model_label_y = 0.62 * top
    for method, ha, dx in (("ProstT5", "right", -1.5), ("ESM3Di", "left", 1.5)):
        median = medians[method]
        ax.plot([median, median], [0, model_label_y], color=METHOD_COLOURS[method], **dashed)
        median_label(method, median + dx, model_label_y, ha)
    # The baseline median line ends on its curve; the label sits beside the falling flank
    baseline = medians["Baseline"]
    ax.plot([baseline, baseline], [0, np.interp(baseline, grid, densities["Baseline"])],
            color=METHOD_COLOURS["Baseline"], **dashed)
    median_label("Baseline", baseline + 5, 0.92 * top, "left")

    ax.set_xlabel("3Di identity to AF3 (%)")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    return ax


# =============================================================================
# SCOPe domain classification benchmark (quality/scope_family.ipynb)
# =============================================================================

# Fixed order everywhere (ESM3Di drawn last, so on top)
SCOPE_METHODS = {"MMseqs2": "mmseqs", "Foldseek": "foldseek", "ProstT5": "prostt5", "ESM3Di": "esm3di"}
SCOPE_LEVELS = {"Family": "fam", "Superfamily": "sup", "Fold": "fol"}
# Top-left corner of the AUC table, in an empty part of each panel
TABLE_XY = {"Family": (0.05, 0.8), "Superfamily": (0.58, 0.97), "Fold": (0.58, 0.97)}


def load_scope_sensitivity(datapath):
    """Per-query sensitivity up to the first FP (4th column of the result TSVs), as a long table."""
    tables = []
    for level, level_suffix in SCOPE_LEVELS.items():
        for method, method_suffix in SCOPE_METHODS.items():
            path = datapath / f"scope/scope40_viral_{method_suffix}_{level_suffix}.tsv"
            sensitivity = np.loadtxt(path, usecols=3)
            tables.append(pd.DataFrame({"level": level, "method": method, "sensitivity": sensitivity}))
    return pd.concat(tables, ignore_index=True)


def plot_scope_level(ax, sensitivity_df, level, show_ylabel=True, show_title=True, table_xy=None):
    """Cumulative sensitivity-up-to-first-FP curves for one SCOPe level, with an AUC table."""
    aucs = {}
    for method in SCOPE_METHODS:
        rows = (sensitivity_df["level"] == level) & (sensitivity_df["method"] == method)
        sensitivity = np.sort(sensitivity_df.loc[rows, "sensitivity"].to_numpy())[::-1]
        fraction = np.arange(1, len(sensitivity) + 1) / len(sensitivity)
        ax.step(np.r_[0, fraction], np.r_[sensitivity[0], sensitivity], where="pre",
                color=METHOD_COLOURS[method], clip_on=False)
        aucs[method] = sensitivity.mean()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.set_xlabel("Fraction of queries")
    if show_ylabel:
        ax.set_ylabel("Sensitivity up to 1st FP")
    if show_title:
        ax.set_title(level)

    # AUC table: method names and values in two aligned columns, best method first.
    # table_xy is the top-left corner in axes coordinates; column and row offsets are in
    # points, so the table keeps its spacing at any panel size.
    table_xy = table_xy or TABLE_XY[level]
    value_dx, row_dy = 46, 7.5

    def table_cell(text, dx, row, ha, **kwargs):
        ax.annotate(text, table_xy, xycoords="axes fraction", xytext=(dx, -row_dy * row),
                    textcoords="offset points", ha=ha, va="top", fontsize=6, **kwargs)

    table_cell("AUC", value_dx, 0, "right", color="0.35")
    for row, method in enumerate(sorted(aucs, key=aucs.get, reverse=True), start=1):
        colour = METHOD_COLOURS[method]
        table_cell(method, 0, row, "left", color=colour)
        table_cell(f"{aucs[method]:.3f}", value_dx, row, "right", color=colour,
                   fontweight="bold" if method == "ESM3Di" else "normal")
    return aucs


# =============================================================================
# 3Di inference speed benchmark (speed/plot_results.ipynb)
# =============================================================================

MODEL_LABELS = {"esm3di": "ESM3Di", "prostt5": "ProstT5", "colabfold": "ColabFold"}
# Repetition 1 of the PLMs is a warm-up run and is left out
MODEL_REPEATS = {
    "esm3di": [2, 3, 4, 5, 6],
    "prostt5": [2, 3, 4, 5, 6],
    "colabfold": [1, 2],
}
MAX_EXTRAPOLATE_SEQ = 10_000
MARKER = dict(marker="o", ms=3.5, mew=0, ls="none", zorder=3)


def model_colour(model):
    return METHOD_COLOURS[MODEL_LABELS[model]]


def runtime_summary(df, model, by):
    """Mean and s.d. of wall time per `by` value for one model (kept repetitions only)."""
    rows = df[(df["model_name"] == model) & (df["repetition"].isin(MODEL_REPEATS[model]))]
    return (
        rows.groupby(by)["wall_time_seconds"]
        .agg(mean="mean", std="std")
        .sort_index()
        .dropna(subset=["mean"])
    )


def use_log_axis(axis):
    """Ticks at powers of ten only, labelled 10^n."""
    axis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    axis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
    axis.set_minor_locator(ticker.NullLocator())


def label_line(ax, x, y, text, model, dx=0, dy=6, ha="left", va="bottom", **kwargs):
    """Method name next to its line, offset by (dx, dy) points from the data point (x, y)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, color=model_colour(model), fontsize=6.5, **kwargs)


def plot_runtime_vs_nseq(ax, nseq_df, title=None):
    """Runtime vs number of sequences (log-log), measured means connected.

    ColabFold was only run up to 300 sequences; a power-law fit to its measurements is drawn
    through them (solid) and extrapolated (dashed) to 10,000 sequences.
    """
    summaries, extrapolations = {}, {}
    for model in MODEL_LABELS:
        summary = runtime_summary(nseq_df, model, "num_sequences")
        n = summary.index.to_numpy(dtype=float)
        if n.max() < MAX_EXTRAPOLATE_SEQ:
            # Power-law fit (straight on log-log axes): solid over the measured range, dashed
            # beyond it up to the extrapolated point
            slope, intercept = np.polyfit(np.log10(n), np.log10(summary["mean"]), 1)
            extrapolations[model] = lambda x, b=slope, a=intercept: 10 ** a * x ** b
            x_fit = np.array([n.min(), n.max()])
            ax.plot(x_fit, extrapolations[model](x_fit), color=model_colour(model), lw=0.8, zorder=2)
            x_ext = np.array([n.max(), MAX_EXTRAPOLATE_SEQ])
            y_ext = extrapolations[model](x_ext)
            ax.plot(x_ext, y_ext, color=model_colour(model), lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.plot(x_ext[-1], y_ext[-1], marker="o", ms=3.5, mfc="white", mec=model_colour(model),
                    mew=0.8, zorder=3)
        else:
            ax.plot(n, summary["mean"], color=model_colour(model), lw=0.8, zorder=2)
        ax.errorbar(n, summary["mean"], yerr=summary["std"].fillna(0), color=model_colour(model),
                    elinewidth=0.6, capsize=0, **MARKER)
        summaries[model] = summary

    ax.set_xscale("log")
    ax.set_yscale("log")
    use_log_axis(ax.xaxis)
    use_log_axis(ax.yaxis)
    ax.set_xlim(6, 1.6e4)
    # The axis reaches just below the fastest run (ESM3Di, 10 sequences: ~0.2 s), but its
    # ticks start at 10^0 rather than adding a lone 10^-1
    ax.set_ylim(0.15, 1e5)
    ax.yaxis.set_major_locator(ticker.FixedLocator(10.0 ** np.arange(0, 6)))
    ax.set_box_aspect(1)
    ax.set_xlabel("Number of sequences")
    ax.set_ylabel("Runtime (s)")
    if title:
        ax.set_title(title)

    # ColabFold (on its extrapolated segment) and ProstT5 above-left of their lines, ESM3Di
    # below-right, so the two language-model labels frame the gap between their lines
    label_line(ax, 1000, extrapolations["colabfold"](1000), "ColabFold", "colabfold",
               dx=-3, dy=4, ha="right")
    label_line(ax, 1000, summaries["prostt5"].loc[1000, "mean"], "ProstT5", "prostt5",
               dx=-3, dy=4, ha="right")
    label_line(ax, 1000, summaries["esm3di"].loc[1000, "mean"], "ESM3Di", "esm3di",
               dx=3, dy=-4, va="top")

    # Line-style key: what solid vs dashed (open circle) means, in neutral grey
    key_colour = "0.45"
    ax.legend(
        handles=[
            Line2D([], [], color=key_colour, lw=0.8, marker="o", ms=3, mew=0, label="Measured"),
            Line2D([], [], color=key_colour, lw=0.8, ls=(0, (3, 2)), marker="o", ms=3,
                   mfc="white", mec=key_colour, mew=0.8, label="Extrapolated"),
        ],
        loc="lower right", handlelength=1.6, handletextpad=0.4, borderaxespad=0.2,
        labelspacing=0.25, fontsize=5.5,
    )
    return summaries


def plot_runtime_vs_length(ax, length_df, models, log_y, title=None):
    """Runtime vs sequence length (one point per length bin, mean ± s.d. over repetitions)."""
    length_df = length_df.assign(seq_length=length_df["total_residues"] / length_df["num_sequences"])
    curves = {}
    for model in models:
        rows = length_df[(length_df["model_name"] == model)
                         & (length_df["repetition"].isin(MODEL_REPEATS[model]))]
        summary = (
            rows.groupby("dataset_name")
            .agg(mean=("wall_time_seconds", "mean"), std=("wall_time_seconds", "std"),
                 seq_length=("seq_length", "mean"))
            .sort_values("seq_length")
        )
        ax.plot(summary["seq_length"], summary["mean"], color=model_colour(model), zorder=2)
        ax.errorbar(summary["seq_length"], summary["mean"], yerr=summary["std"].fillna(0),
                    color=model_colour(model), elinewidth=0.6, capsize=0, **MARKER)
        curves[model] = summary

    ax.set_xlim(0, 1050)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    if log_y:
        ax.set_yscale("log")
        use_log_axis(ax.yaxis)
    else:
        ax.set_ylim(0, None)
    ax.set_box_aspect(1)
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Runtime (s)")
    if title:
        ax.set_title(title)
    return curves


def plot_esm3di_runtime_modes(ax, nseq_df, esm3di_cli_df):
    """ESM3Di runtime vs number of sequences, timed end to end (CLI call, including model
    loading; filled markers) and inference only (model already loaded; open markers), with
    ProstT5 (always end to end) as a thin reference line. Measured means connected."""
    colour = model_colour("esm3di")
    end_to_end = runtime_summary(esm3di_cli_df, "esm3di", "num_sequences")
    inference = runtime_summary(nseq_df, "esm3di", "num_sequences")
    prostt5 = runtime_summary(nseq_df, "prostt5", "num_sequences")

    ax.plot(prostt5.index, prostt5["mean"], color=model_colour("prostt5"), lw=0.8, alpha=0.7, zorder=1)
    for summary, open_marker in ((end_to_end, False), (inference, True)):
        ax.plot(summary.index, summary["mean"], color=colour, zorder=2)
        marker = dict(MARKER, mfc="white", mec=colour, mew=0.8) if open_marker else MARKER
        ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), color=colour,
                    elinewidth=0.6, capsize=0, **marker)

    ax.set_xscale("log")
    ax.set_yscale("log")
    use_log_axis(ax.xaxis)
    use_log_axis(ax.yaxis)
    ax.set_xlim(6, 1.6e4)
    ax.set_ylim(0.15, 1e3)
    ax.yaxis.set_major_locator(ticker.FixedLocator(10.0 ** np.arange(0, 4)))
    ax.set_box_aspect(1)
    ax.set_xlabel("Number of sequences")
    ax.set_ylabel("Runtime (s)")

    # Both ESM3Di labels where the two curves are a decade apart: end to end above its flat
    # (start-up dominated) part, inference only below-right of its line; ProstT5 above-left
    label_line(ax, 10, end_to_end.loc[10, "mean"], "ESM3Di\n(end to end)", "esm3di", dy=5,
               linespacing=1.1)
    label_line(ax, 100, inference.loc[100, "mean"], "ESM3Di\n(inference only)", "esm3di",
               dx=4, dy=-4, va="top", linespacing=1.1)
    label_line(ax, 1000, prostt5.loc[1000, "mean"], "ProstT5", "prostt5", dx=-3, dy=4, ha="right")
    return {"end to end": end_to_end, "inference only": inference, "ProstT5": prostt5}
