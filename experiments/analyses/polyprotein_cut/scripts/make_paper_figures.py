#!/usr/bin/env python3
"""Figures and tables for the polyprotein cut-site comparison (Fig. 2d and supplements).

Consumes the output of ``run_comparison.py`` and writes, into the same directory:

  roc_curves_by_genus.{png,pdf,svg}  Fig. 2d: seed-averaged ROC per held-out genus
  paired_roc_auc.{png,pdf,svg}    alternative panel: paired dot plot
  paired_delta.{png,pdf,svg}      alternative panel: paired difference per genus
  supp_table_metrics.csv                  all metrics, mean ± sd over seeds
  supp_fig_pick_distances.{png,pdf,svg}   distance from each top-10 pick to the nearest site
  supp_table_pick_distances.csv           the same as a table (preferred over the figure)

Usage
-----
    python make_paper_figures.py --results-dir results/esm3di_vs_esmpp_standardised \
                                 --predictions /path/to/oof_ffn_predictions.csv

``seed_metrics.csv`` must be in --results-dir. ``oof_ffn_predictions.csv`` is several
hundred MB and is therefore not kept in the repository; pass its location with
--predictions. Without it the ROC-curve panel and the pick-distance outputs are skipped
and the rest are still written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from pub_style import MM, use_pub_style        # the style panels a-c of Fig. 2 use

use_pub_style()                                 # Nimbus Sans / Arial at 7 pt, as elsewhere
# Panel d is drawn at 3.6 in and composited into Fig. 2 at 54 mm. use_pub_style() above only
# fixes the typeface (Arial/Helvetica/Nimbus Sans, as in panels a-c); the sizes below are the
# ones the published layout was built around.

ARCH = "ffn"
SHORT = {"Hepacivirus-CHH": "Hepacivirus", "Orthoflavivirus-CHH": "Orthoflavivirus",
         "Pegivirus-CHH": "Pegivirus", "Pestivirus-CHH": "Pestivirus"}
TICK = {"Hepacivirus-CHH": "Hepaci", "Orthoflavivirus-CHH": "Orthoflavi",
        "Pegivirus-CHH": "Pegi", "Pestivirus-CHH": "Pesti"}
CLADE = {"Hepacivirus-CHH": "#1f77b4", "Orthoflavivirus-CHH": "#ff7f0e",
         "Pegivirus-CHH": "#2ca02c", "Pestivirus-CHH": "#d62728"}
ARM = {"esmpp_base": "ESM++", "esm3di": "ESM3Di"}
ARM_LONG = {"esmpp_base": "ESM++ (base)", "esm3di": "ESM3Di"}
ARM_COLOR = {"esmpp_base": "#9aa0a6", "esm3di": "#2f6fb5"}
METRICS = [("roc_auc", "ROC AUC"), ("ap_strict", "AP"), ("ap_tolerance_2", "AP ±2"),
           ("ap_tolerance_5", "AP ±5"), ("top10_site_recovery", "top-10 recovery")]
TOP_K, NMS_RADIUS = 10, 5          # must match run_comparison.TOP_K / TOP_K_TOLERANCE
GRID = np.linspace(0, 1, 501)


def save(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.{{png,pdf,svg}}")


def mean_roc(sub: pd.DataFrame) -> np.ndarray:
    """TPR averaged over seeds on a common FPR grid."""
    curves = []
    for _, g in sub.groupby("seed"):
        y = g.label.values.astype(bool)
        order = np.argsort(-g.score.values, kind="mergesort")
        y = y[order]
        tp, fp = np.cumsum(y), np.cumsum(~y)
        curves.append(np.interp(GRID, np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]))
    return np.mean(curves, axis=0)


def pick_peaks(scores: np.ndarray, k: int = TOP_K, nms: int = NMS_RADIUS) -> list[int]:
    """Greedy peak picking with non-maximum suppression, as in run_comparison.py."""
    available, picks = np.ones(len(scores), bool), []
    for idx in np.argsort(-scores, kind="mergesort"):
        if len(picks) >= k:
            break
        if not available[idx]:
            continue
        picks.append(int(idx))
        available[max(0, idx - nms):idx + nms + 1] = False
    return picks


# --------------------------------------------------------------------------- panels
def panel_roc_curves(pred: pd.DataFrame, met: pd.DataFrame, out: Path) -> None:
    """Fig. 2d. Curves are seed means; the inset table gives seed-mean ROC AUC."""
    groups = sorted(pred.heldout_group.unique())
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=400)
    ax.plot([0, 1], [0, 1], ls="--", lw=.8, color="#bbb", zorder=1)
    for g in groups:
        for arm, ls in (("esm3di", "-"), ("esmpp_base", ":")):
            ax.plot(GRID, mean_roc(pred[(pred.heldout_group == g) & (pred.embedding_source == arm)]),
                    ls=ls, lw=1.5, color=CLADE[g], zorder=3)
    ax.set_xlabel("False positive rate", fontsize=8)
    ax.set_ylabel("True positive rate", fontsize=8)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.005)
    ax.tick_params(labelsize=7)
    ax.grid(color="#f0f1f3", lw=.6, zorder=0), ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # the table is kept strictly under the identity line: top-left corner (x0, y0+h) with y0+h < x0
    x0, y0, w, h = 0.47, 0.03, 0.52, 0.39
    assert y0 + h < x0, "table would cross the identity line"
    rows = [[SHORT[g],
             f"{met[(met.heldout_group == g) & (met.embedding_source == 'esm3di')].roc_auc.mean():.3f}",
             f"{met[(met.heldout_group == g) & (met.embedding_source == 'esmpp_base')].roc_auc.mean():.3f}"]
            for g in groups]
    table = ax.table(cellText=rows, colLabels=["", "ESM3Di", "ESM++"], cellLoc="center",
                     bbox=[x0, y0, w, h], zorder=5)
    table.auto_set_font_size(False), table.set_fontsize(6.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(.4), cell.set_edgecolor("#ccc")
        cell.set_facecolor("white")        # opaque, so the grid does not strike through
        cell.set_width([.32, .16, .16][c])
        if c == 1 and r > 0:                       # the ESM3Di values, not the column headers
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(color=CLADE[groups[r - 1]], ha="left")
    ax.text(x0 + w / 2, y0 + h + 0.025, f"ROC AUC (mean of {met.seed.nunique()} seeds)",
            transform=ax.transAxes, ha="center", fontsize=6.4, fontweight="bold")

    ax.plot([], [], "-", color="#555", lw=1.4, label="ESM3Di")
    ax.plot([], [], ":", color="#555", lw=1.4, label="ESM++")
    legend = ax.legend(fontsize=6.8, frameon=False, loc="upper left", bbox_to_anchor=(0.012, 0.995),
                       handlelength=1.6, handletextpad=.45, labelspacing=.3, borderpad=0)
    # the legend must not sit on top of any curve
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    inside = (GRID >= max(box.x0, 0)) & (GRID <= box.x1)
    peak = max(mean_roc(pred[(pred.heldout_group == g) & (pred.embedding_source == a)])[inside].max()
               for g in groups for a in ("esm3di", "esmpp_base"))
    assert box.y0 > peak, f"legend overlaps a curve (box bottom {box.y0:.3f} <= peak {peak:.3f})"
    print(f"  legend clears the curves by {box.y0 - peak:.3f}")
    fig.tight_layout()
    save(fig, out, "roc_curves_by_genus")


def panel_paired_dots(met: pd.DataFrame, out: Path) -> None:
    """Alternative panel: both arms per genus, seed spread as error bars."""
    groups = sorted(met.heldout_group.unique())
    fig, ax = plt.subplots(figsize=(3.3, 2.9), dpi=300)
    for i, g in enumerate(groups):
        sub = met[met.heldout_group == g]
        b = sub[sub.embedding_source == "esmpp_base"].roc_auc
        e = sub[sub.embedding_source == "esm3di"].roc_auc
        ax.plot([i - .16, i + .16], [b.mean(), e.mean()], "-", color="#c8ccd0", lw=1.4, zorder=1)
        for x, v, arm in ((i - .16, b, "esmpp_base"), (i + .16, e, "esm3di")):
            ax.errorbar(x, v.mean(), yerr=v.std(), fmt="o", ms=6, color=ARM_COLOR[arm],
                        capsize=3, lw=1.2, zorder=3)
            ax.plot(np.full(len(v), x), v, ".", ms=2.5, color=ARM_COLOR[arm], alpha=.55, zorder=2)
        ax.text(i, max(b.mean() + b.std(), e.mean() + e.std()) + 0.011,
                f"+{e.mean() - b.mean():.3f}", ha="center", fontsize=6.4, color="#333")
    ax.set_xticks(range(len(groups))), ax.set_xticklabels([TICK[g] for g in groups], fontsize=8)
    ax.set_ylabel("ROC AUC (cut sites)", fontsize=8.5), ax.set_xlabel("held-out genus", fontsize=8.5)
    ax.set_ylim(0.77, 0.93), ax.set_xlim(-0.5, len(groups) - 0.5), ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#eceef0", lw=.7, zorder=0), ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ARM_COLOR[a], ms=5, label=ARM_LONG[a])
                       for a in ("esmpp_base", "esm3di")],
              fontsize=6.8, frameon=False, loc="lower left", handletextpad=.4)
    fig.tight_layout()
    save(fig, out, "paired_roc_auc")


def panel_paired_delta(met: pd.DataFrame, out: Path) -> None:
    """Alternative panel: within-seed difference per genus.

    Both arms share the seed, so they get identical head initialisation, dropout masks and
    batch order; the difference therefore isolates the effect of the embeddings.
    """
    groups = sorted(met.heldout_group.unique())
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    ax.axvline(0, color="#444", lw=1.0, zorder=1)
    ys = np.arange(len(groups))[::-1]
    for y, g in zip(ys, groups):
        sub = met[met.heldout_group == g]
        b = sub[sub.embedding_source == "esmpp_base"].set_index("seed").roc_auc.sort_index()
        e = sub[sub.embedding_source == "esm3di"].set_index("seed").roc_auc.sort_index()
        diff = (e - b).values
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
        ax.plot(diff, np.full(len(diff), y), "o", ms=3.2, color=ARM_COLOR["esm3di"], alpha=.45, zorder=2)
        ax.errorbar(diff.mean(), y, xerr=ci, fmt="o", ms=6.5, color=ARM_COLOR["esm3di"],
                    capsize=3.5, lw=1.5, zorder=3)
    ax.set_yticks(ys), ax.set_yticklabels([SHORT[g] for g in groups], fontsize=8)
    ax.set_xlabel("Δ ROC AUC  (ESM3Di − ESM++ base)", fontsize=8.5)
    ax.set_ylim(-0.6, len(groups) - 0.4), ax.tick_params(labelsize=7.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#eef0f2", lw=.7, zorder=0), ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "paired_delta")


# --------------------------------------------------------------------------- tables
def table_metrics(met: pd.DataFrame, out: Path) -> None:
    rows = []
    for g in sorted(met.heldout_group.unique()):
        for arm in ("esm3di", "esmpp_base"):
            sub = met[(met.heldout_group == g) & (met.embedding_source == arm)]
            row = {"Held-out genus": SHORT[g], "Embeddings": ARM_LONG[arm],
                   "n seq": int(sub.n_test_sequences.iloc[0]),
                   "n sites": int(sub.n_test_positives.iloc[0])}
            row.update({label: f"{sub[key].mean():.3f} ± {sub[key].std():.3f}"
                        for key, label in METRICS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "supp_table_metrics.csv", index=False)
    print("  wrote supp_table_metrics.csv")

    print("\n  mean over the four genera (paired, ESM3Di - base):")
    for key, label in METRICS:
        t = met.groupby(["heldout_group", "embedding_source"])[key].mean().unstack()
        d = t["esm3di"] - t["esmpp_base"]
        print(f"    {label:17s} {t['esmpp_base'].mean():.3f} -> {t['esm3di'].mean():.3f}"
              f"   delta {d.mean():+.3f}   genera {(d > 0).sum()}/4")


def pick_distances(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    """Distance from every top-10 pick to the nearest annotated cut site, pooled over seeds."""
    out = {}
    for arm, sub in pred.groupby("embedding_source"):
        distances = []
        for _, g in sub.groupby(["seed", "sequence_id"]):
            g = g.sort_values("position")
            true = np.nonzero(g.label.values)[0]
            if len(true) == 0:
                continue
            distances += [int(np.abs(true - p).min()) for p in pick_peaks(g.score.values)]
        out[arm] = np.array(distances)
    return out


def distances_outputs(pred: pd.DataFrame, out: Path) -> None:
    res = pick_distances(pred)
    buckets = [(0, 0, "exact"), (1, 2, "1–2"), (3, 5, "3–5"),
               (6, 20, "6–20"), (21, 100, "21–100"), (101, 10 ** 9, ">100")]
    rows = [{"distance (residues)": name,
             "ESM++ base (%)": round(100 * ((res["esmpp_base"] >= lo) & (res["esmpp_base"] <= hi)).mean(), 1),
             "ESM3Di (%)": round(100 * ((res["esm3di"] >= lo) & (res["esm3di"] <= hi)).mean(), 1)}
            for lo, hi, name in buckets]
    for label, fn in (("median distance", np.median),
                      ("within 5 (%)", lambda x: 100 * (x <= 5).mean()),
                      ("within 20 (%)", lambda x: 100 * (x <= 20).mean()),
                      ("within 100 (%)", lambda x: 100 * (x <= 100).mean())):
        rows.append({"distance (residues)": label,
                     "ESM++ base (%)": round(float(fn(res["esmpp_base"])), 1),
                     "ESM3Di (%)": round(float(fn(res["esm3di"])), 1)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "supp_table_pick_distances.csv", index=False)
    print("  wrote supp_table_pick_distances.csv")
    print(table.to_string(index=False))
    print(f"  n = {len(res['esm3di'])} picks per arm")

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), dpi=400)
    x, width = np.arange(len(buckets)), .38
    for i, arm in enumerate(("esmpp_base", "esm3di")):
        vals = [100 * ((res[arm] >= lo) & (res[arm] <= hi)).mean() for lo, hi, _ in buckets]
        ax[0].bar(x + (i - .5) * width, vals, width, color=ARM_COLOR[arm], label=ARM_LONG[arm])
    ax[0].set_xticks(x), ax[0].set_xticklabels([b[2] for b in buckets], fontsize=7.5)
    ax[0].set_xlabel("distance to nearest true cut site (residues)", fontsize=8)
    ax[0].set_ylabel("% of top-10 picks", fontsize=8)
    ax[0].legend(fontsize=7, frameon=False)
    for arm in ("esmpp_base", "esm3di"):
        s = np.sort(res[arm])
        ax[1].plot(np.r_[0, s], np.r_[0, np.arange(1, len(s) + 1) / len(s) * 100],
                   color=ARM_COLOR[arm], lw=1.7, label=ARM_LONG[arm])
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xlabel("distance to nearest true cut site (residues)", fontsize=8)
    ax[1].set_ylabel("cumulative % of picks", fontsize=8)
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")
    for a in ax:
        a.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(color="#eef0f2", lw=.6, zorder=0), a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "supp_fig_pick_distances")


def main() -> None:
    here = Path(__file__).resolve().parent.parent   # .../polyprotein_cut
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=here / "results" / "esm3di_vs_esmpp_standardised",
                   help="directory holding seed_metrics.csv; outputs are written here")
    p.add_argument("--predictions", type=Path, default=None,
                   help="oof_ffn_predictions.csv from the same run (large; kept outside the repo)")
    args = p.parse_args()

    met = pd.read_csv(args.results_dir / "seed_metrics.csv")
    met = met[met.architecture == ARCH]
    seeds = sorted(met.seed.unique())
    print(f"{len(met)} rows, architecture {ARCH}, seeds {seeds}\n")

    table_metrics(met, args.results_dir)
    print()
    panel_paired_dots(met, args.results_dir)
    panel_paired_delta(met, args.results_dir)

    if args.predictions is None or not args.predictions.exists():
        print("\n  --predictions not given: skipping the ROC panel and the pick distances")
        return
    pred = pd.read_csv(args.predictions)
    pred = pred[pred.seed.isin(seeds)]
    # run_comparison.py appends to oof_ffn_predictions.csv, so an interrupted and resumed run
    # can leave a (group, arm, seed, sequence) written twice. Duplicates are not harmless here:
    # pick_distances() works in per-sequence row-index space, so a doubled sequence doubles every
    # distance and roughly doubles the ">100 residues" bin. Drop them and say so.
    keys = ["heldout_group", "embedding_source", "seed", "sequence_id", "position"]
    n_raw = len(pred)
    pred = pred.drop_duplicates(keys, keep="last")
    if len(pred) < n_raw:
        print(f"\n  WARNING: dropped {n_raw - len(pred):,} duplicated prediction rows "
              f"({100 * (1 - len(pred) / n_raw):.1f}% of the file) before analysis")
    print(f"\n{len(pred):,} prediction rows, seeds {sorted(pred.seed.unique())}")
    panel_roc_curves(pred, met, args.results_dir)
    distances_outputs(pred, args.results_dir)


if __name__ == "__main__":
    main()
