"""Rebuild esm3di foldtree trees from an existing all-vs-all search, for resampling.

Reproduces the tree steps of the esm3di foldtree workflow
(src/esm3di/workflow/scripts/foldseekres2distmat_simple.py, QuickTree,
postprocess.py, MAD rooting) on a subset of the sequences, without rerunning
3Di prediction or Foldseek. Foldseek's fident of a pair does not depend on the
other sequences in the database (the workflow searches exhaustively with
-e inf), so dropping rows of the all-vs-all table equals rerunning the search
on the reduced set.
"""

import multiprocessing
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from esm3di.workflow.scripts.run_madroot import run_mad_root
from monophyly import clade_recovery, relabel_tree, score_monophyly, ultrametricity, ultrametricity_best_root

BFACTOR = 0.93       # foldseekres2distmat_simple.py, fident kernel
OFFSET = 10 ** -5    # added before the Jukes-Cantor transform
TAYLOR_TERMS = 100


def load_allvall(path):
    """Query, target and fident of a workflow allvall_1.csv, without '.pdb'."""
    table = pd.read_table(path, header=None, usecols=[0, 1, 2], names=["query", "target", "fident"])
    for column in ("query", "target"):
        table[column] = table[column].str.replace(".pdb", "", regex=False)
    table["fident"] = table["fident"].astype(float).fillna(0)
    return table


def jukes_cantor_taylor(kn_ratio, bfactor=BFACTOR, terms=TAYLOR_TERMS):
    """Same series as foldseek2tree.jukes_cantor_taylor, summed term by term (less memory)."""
    distances = np.zeros_like(kn_ratio)
    power = np.ones_like(kn_ratio)
    for i in range(1, terms):
        power = power * kn_ratio
        distances += power / (bfactor ** (i - 1) * i)
    np.fill_diagonal(distances, 0)
    return distances


def distance_matrix(allvall, transform=jukes_cantor_taylor):
    """Sorted identifiers and the distance matrix for an all-vs-all table; the transform of the workflow
    by default, or another function of the 1 - fident matrix."""
    ids = sorted(set(allvall["query"]) | set(allvall["target"]))
    index = {name: i for i, name in enumerate(ids)}
    q = allvall["query"].map(index).to_numpy()
    t = allvall["target"].map(index).to_numpy()
    similarity = np.zeros((len(ids), len(ids)))
    np.add.at(similarity, (q, t), allvall["fident"].to_numpy())
    np.add.at(similarity, (t, q), allvall["fident"].to_numpy())
    distances = transform(1 - similarity / 2 + OFFSET)
    np.fill_diagonal(distances, 0)
    return ids, distances


def build_tree(allvall, workdir, quicktree, transform=jukes_cantor_taylor):
    """Rooted Newick tree for an all-vs-all table, built like esm3di foldtree."""
    workdir = Path(workdir)
    ids, distances = distance_matrix(allvall, transform)
    # QuickTree truncates names; use placeholders as the workflow does
    placeholders = [f"seq{i:06d}" for i in range(len(ids))]
    matrix = workdir / "distmat.txt"
    with open(matrix, "w") as handle:
        handle.write(f"{len(ids)}\n")
        for name, row in zip(placeholders, distances):
            handle.write(name + " " + " ".join(f"{d:.4f}" for d in row) + "\n")
    raw = workdir / "tree.nwk"
    with open(raw, "w") as handle:
        subprocess.run([str(quicktree), "-i", "m", str(matrix)], stdout=handle, check=True)
    relabel_tree(raw, raw, dict(zip(placeholders, ids)))

    # postprocess.py: negative branch lengths become 0 (the tree is otherwise unchanged)
    newick = "".join(line.strip() for line in raw.read_text().splitlines())
    processed = workdir / "tree.PP.nwk"
    processed.write_text(re.sub(r":-[0-9.eE+-]+", ":0", newick) + "\n")
    return Path(run_mad_root(processed))


def score_subset(allvall, keep, labels, quicktree):
    """Monophyly score of the tree built from the sequences in ``keep``, the groups that do not form
    a clade in it, and its root-to-tip coefficient of variation at its MAD root and at its best root."""
    subset = allvall[allvall["query"].isin(keep) & allvall["target"].isin(keep)]
    with tempfile.TemporaryDirectory() as workdir:
        rooted = build_tree(subset, workdir, quicktree)
        labelled = relabel_tree(rooted, Path(workdir) / "labelled.nwk", labels)
        not_clades = [row["group"] for row in clade_recovery(labelled) if not row["clade"]]
        return (*score_monophyly(labelled, verbose=False), not_clades, ultrametricity(labelled),
                ultrametricity_best_root(labelled))


# --- Paired delete-d jackknife ---------------------------------------------

_STATE = {}


def _replicate(replicate):
    state = _STATE
    rng = np.random.default_rng([state["seed"], replicate])
    keep = set(rng.choice(state["ids"], size=round(state["fraction"] * len(state["ids"])), replace=False))
    rows = []
    for method, allvall in state["allvall"].items():
        mono, internal, fraction, not_clades, cv, best_cv = score_subset(allvall, keep, state["labels"], state["quicktree"])
        rows.append(dict(replicate=replicate, method=method, tips=len(keep),
                         monophyletic=mono, internal_nodes=internal, fraction=fraction,
                         not_clades=";".join(not_clades), root_to_tip_cv=cv, best_root_cv=best_cv))
    return rows


def jackknife(allvall_by_method, labels, quicktree, replicates=100, fraction=0.9, seed=0, workers=8):
    """Score trees built from random subsets of the sequences.

    Each replicate keeps the same random ``fraction`` of the sequences for every
    method, so the methods are compared on identical subsets (paired). Replicate
    ``r`` always draws the same subset for a given ``seed``. Returns one row per
    replicate and method.
    """
    ids = sorted(set.intersection(*(set(table["query"]) for table in allvall_by_method.values())))
    _STATE.update(allvall=allvall_by_method, labels=labels, quicktree=quicktree,
                  ids=ids, fraction=fraction, seed=seed)
    # fork: workers share the all-vs-all tables instead of receiving copies
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = [row for result in pool.imap_unordered(_replicate, range(replicates)) for row in result]
    return pd.DataFrame(rows).sort_values(["replicate", "method"], ignore_index=True)
