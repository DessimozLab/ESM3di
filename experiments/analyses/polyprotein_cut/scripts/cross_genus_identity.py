#!/usr/bin/env python3
"""Sequence divergence between the leave-one-genus-out folds.

Supports the Methods statement that a held-out genus has no training-set homologue
detectable above the twilight zone, i.e. that the evaluation is not homology recognition.

Every cross-genus pair of annotated polyproteins is aligned (Smith-Waterman, BLOSUM62,
gap open -11, extend -1) and two identities are reported:

  identity within the aligned region  matches / aligned columns
  alignable fraction                  matches / length of the shorter sequence
                                      (identity x coverage; low when only a fragment aligns)

Usage
-----
    python cross_genus_identity.py [--out results/esm3di_vs_esmpp_standardised/cross_genus_identity.csv]

Runs all 2,704 pairs in roughly ten minutes on one core.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

SCRIPTS = Path(__file__).resolve().parent
HERE = SCRIPTS.parent                                   # .../polyprotein_cut
sys.path.insert(0, str(SCRIPTS))                        # run_comparison.load_records
sys.path.insert(0, str(HERE.parents[2] / "src"))


def aligner() -> Align.PairwiseAligner:
    a = Align.PairwiseAligner(mode="local",
                              substitution_matrix=substitution_matrices.load("BLOSUM62"),
                              open_gap_score=-11, extend_gap_score=-1)
    return a


def identities(a: Align.PairwiseAligner, s1: str, s2: str) -> tuple[float, float]:
    aln = a.align(s1, s2)[0]
    matches = sum(1 for x, y in zip(aln[0], aln[1]) if x == y and x != "-")
    return matches / min(len(s1), len(s2)), matches / len(aln[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=HERE / "results" / "esm3di_vs_esmpp_standardised" / "cross_genus_identity.csv")
    args = p.parse_args()

    from run_comparison import load_records          # same 86 records the comparison uses
    records = load_records()
    counts = {g: sum(1 for r in records if r["group"] == g)
              for g in sorted({r["group"] for r in records})}
    print(f"{len(records)} annotated records; {counts}", flush=True)

    a = aligner()
    rows = []
    for i, j in itertools.combinations(range(len(records)), 2):
        if records[i]["group"] == records[j]["group"]:
            continue
        over_shorter, over_aln = identities(a, records[i]["sequence"], records[j]["sequence"])
        rows.append((records[i]["group"], records[j]["group"],
                     records[i]["identifier"], records[j]["identifier"], over_shorter, over_aln))
    df = pd.DataFrame(rows, columns=["group_1", "group_2", "id_1", "id_2",
                                     "alignable_fraction", "identity_in_alignment"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} cross-genus pairs -> {args.out}")

    df["pair"] = [" / ".join(sorted([g1.replace("-CHH", ""), g2.replace("-CHH", "")]))
                  for g1, g2 in zip(df.group_1, df.group_2)]
    summary = df.groupby("pair").agg(
        n=("alignable_fraction", "size"),
        identity_in_alignment_median=("identity_in_alignment", "median"),
        identity_in_alignment_max=("identity_in_alignment", "max"),
        alignable_fraction_median=("alignable_fraction", "median"),
        alignable_fraction_max=("alignable_fraction", "max"))
    for c in summary.columns[1:]:
        summary[c] = (100 * summary[c]).round(1)
    print("\nper genus pair (%):")
    print(summary.to_string())

    best = []
    for r in records:
        sub = df[(df.id_1 == r["identifier"]) | (df.id_2 == r["identifier"])]
        best.append(sub.alignable_fraction.max())
    best = np.array(best)
    print(f"\nbest cross-genus match per sequence (alignable fraction): "
          f"median {100 * np.median(best):.1f}%, max {100 * best.max():.1f}%")
    print(f"maximum identity within an alignment, any pair: "
          f"{100 * df.identity_in_alignment.max():.1f}%")


if __name__ == "__main__":
    main()
