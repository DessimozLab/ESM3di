"""Monophyly scoring, tip relabelling and tree comparison for the fold-tree notebooks.

Leaf names start with a group code followed by an underscore, e.g. ``E1HPHV_...``
(Flaviviridae glycoproteins) or ``Orthoebolavirus-zairense_...`` (Filoviridae).
"""

import re
from pathlib import Path

from Bio import Phylo
import numpy as np

GROUP_PATTERN = re.compile(r"^([A-Za-z0-9\-]+)_")
# A tip label in Newick: after "(" or ",", up to ":", ",", ")" or ";" (QuickTree
# also puts line breaks between tips, which are kept)
TIP_LABEL = re.compile(r"(?<=[(,])(\s*)([^(),:;\s]+)(?=\s*[:,);])")
PROTEIN_CODES = {"E": "E0", "E1": "E1", "E2": "E2"}


def score_monophyly(filename, verbose=True):
    """
    Calculate the fraction of internal nodes whose descendant leaves all
    belong to the same group.

    Leaf names must follow this format:
        ABCD_[unique_name]

    Parameters
    ----------
    filename : str
        Path to the input Newick file.
    verbose : bool
        Print the three numbers, as the original notebooks did.

    Returns
    -------
    tuple
        (monophyletic_internal_nodes, all_internal_nodes, fraction)
    """
    tree = Phylo.read(filename, "newick")

    def get_group(leaf):
        if not leaf.name:
            raise ValueError("Encountered a leaf without a name.")

        match = GROUP_PATTERN.match(leaf.name)
        if not match:
            raise ValueError(
                f"Invalid leaf name {leaf.name!r}: expected a group code "
                "followed by an underscore."
            )

        return match.group(1)

    internal_nodes = tree.get_nonterminals()
    monophyletic_nodes = 0

    for node in internal_nodes:
        descendant_groups = {
            get_group(leaf) for leaf in node.get_terminals()
        }

        if len(descendant_groups) == 1:
            monophyletic_nodes += 1

    total_nodes = len(internal_nodes)
    fraction = monophyletic_nodes / total_nodes if total_nodes else 0.0

    if verbose:
        print(f"Monophyletic internal nodes: {monophyletic_nodes}")
        print(f"All internal nodes: {total_nodes}")
        print(f"Fraction of monophyletic internal nodes: {fraction:.6f}")
        print("-" * 40)

    return monophyletic_nodes, total_nodes, fraction


def ultrametricity(filename):
    """Variance of the root-to-tip distances normalised by their mean (Moi et al. 2025, eq. 2).

    The distances are divided by their mean, so that their distribution is centred on 1,
    and their variance is taken. 0 for an ultrametric tree (all tips equally far from the
    root), larger for trees whose tips sit at unequal distances. Normalising makes it
    comparable between trees with different distance scales (3Di, structure, amino acid).
    The tree is used with the root it has, the MAD root for esm3di foldtree trees; for an
    arbitrarily rooted tree the value is not meaningful.
    """
    depths = np.array(list(root_to_tip_depths(filename).values()))
    return float((depths / depths.mean()).var())


def root_to_tip_depths(filename):
    """Distance from the root to each tip, by tip name."""
    tree = Phylo.read(filename, "newick")
    return {clade.name: depth for clade, depth in tree.depths().items() if clade.is_terminal()}


def ultrametricity_best_root(filename, positions=11):
    """The lowest ``ultrametricity`` over all root positions.

    Tries ``positions`` evenly spaced points on every branch, so the value does not
    depend on the root of the tree: it shows how ultrametric the tree can be at best.
    """
    tree = Phylo.read(filename, "newick")
    nodes = list(tree.find_clades())
    index = {id(node): i for i, node in enumerate(nodes)}
    tips = [node for node in nodes if node.is_terminal()]
    edges = [(index[id(parent)], index[id(child)], child.branch_length or 0.0)
             for parent in nodes for child in parent.clades]
    neighbours = [[] for _ in nodes]
    for parent, child, length in edges:
        neighbours[parent].append((child, length))
        neighbours[child].append((parent, length))
    # to_tip[node, tip]: distance between a node and a tip
    to_tip = np.zeros((len(nodes), len(tips)))
    for t, tip in enumerate(tips):
        stack, seen = [(index[id(tip)], 0.0)], {index[id(tip)]}
        while stack:
            node, distance = stack.pop()
            to_tip[node, t] = distance
            for other, length in neighbours[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append((other, distance + length))
    tip_ids = {id(tip): t for t, tip in enumerate(tips)}
    best = np.inf
    for parent, child, length in edges:
        below = np.zeros(len(tips), bool)
        below[[tip_ids[id(leaf)] for leaf in nodes[child].get_terminals()]] = True
        for t in np.linspace(0, length, positions):
            depths = np.where(below, to_tip[child] + t, to_tip[parent] + length - t)
            best = min(best, (depths / depths.mean()).var())
    return float(best)


# --- Relabelling -------------------------------------------------------------

def fasta_ids(path):
    """Identifiers (first word of each header) of a FASTA file, in file order."""
    with open(path) as handle:
        return [line[1:].split()[0] for line in handle if line.startswith(">")]


def glycoprotein_labels(fasta_by_protein):
    """Map each glycoprotein tree tip to its six-letter label.

    ``fasta_by_protein`` maps "E", "E1" and "E2" to the FASTA holding that
    protein. The FASTA header ``HPHV_name.pdb`` of the E1 file appears in the
    tree as ``HPHV_name`` (the workflow drops ".pdb") and becomes ``E1HPHV_name``,
    the format of the archived trees (E is written as E0).
    """
    labels = {}
    for protein, path in fasta_by_protein.items():
        for identifier in fasta_ids(path):
            tip = identifier.removesuffix(".pdb")
            if tip in labels:
                raise ValueError(f"{tip!r} occurs in more than one protein FASTA.")
            labels[tip] = PROTEIN_CODES[protein] + tip
    return labels


def filoviridae_labels(taxonomy_csv, fasta=None):
    """Map each accession to ``Genus-species_accession`` from a taxonomic_info table.

    The species is the last entry of the NCBI ``taxonomy`` lineage. That lineage
    excludes the record's own taxon, so for records filed directly under a
    species it ends at the genus; then the ``scientific_name`` is used if it is a
    binomial of that genus. Accessions still without a species become
    ``Genus-unassigned_...``; accessions without a lineage become
    ``Unknown-unassigned_...``.

    With ``fasta``, accessions without a species (e.g. failed NCBI lookups) take
    it from the organism in their FASTA header (``[Zaire ebolavirus]``), if all
    other accessions of that organism share one species.
    """
    import pandas as pd

    table = pd.read_csv(taxonomy_csv)
    # Rows of failed lookups carry the accession only in the first (index) column
    table["accession"] = table["accession"].fillna(table.iloc[:, 0])
    labels = {}
    for accession, lineage, name in zip(table["accession"], table["taxonomy"], table["scientific_name"]):
        ranks = [] if pd.isna(lineage) else [r.strip() for r in lineage.split(";")]
        last = ranks[-1].split() if ranks else []
        name = [] if pd.isna(name) else name.split()
        if len(last) == 1 and len(name) == 2 and name[0] == last[0]:
            last = name
        if len(last) >= 2:
            group = f"{last[0]}-{'-'.join(last[1:])}"
        elif last:
            group = f"{last[0]}-unassigned"
        else:
            group = "Unknown-unassigned"
        labels[accession] = f"{group}_{accession}"

    if fasta is not None:
        species_of = {}
        for accession, organism in zip(table["accession"], table["organism"]):
            if not pd.isna(organism) and "-unassigned_" not in labels[accession]:
                species_of.setdefault(organism, set()).add(labels[accession].split("_")[0])
        with open(fasta) as handle:
            for line in handle:
                header = re.match(r">(\S+).*\[([^\]]+)\]\s*$", line)
                if not header:
                    continue
                accession, organism = header.groups()
                species = species_of.get(organism, set())
                if "-unassigned_" in labels.get(accession, "-unassigned_") and len(species) == 1:
                    labels[accession] = f"{next(iter(species))}_{accession}"
    return labels


def relabel_tree(tree_in, tree_out, labels, missing="error"):
    """Rewrite the tip names of a Newick file with ``labels`` (old name -> new name).

    Only tip labels are changed; topology, branch lengths and formatting stay as
    they are. ``missing="error"`` fails on tips without a label, ``"unknown"``
    prefixes them with ``Unknown-unassigned_``. Returns the output path.
    """
    newick = Path(tree_in).read_text()
    unlabelled = []

    def replace(match):
        space, name = match.groups()
        if name in labels:
            return space + labels[name]
        unlabelled.append(name)
        return f"{space}Unknown-unassigned_{name}"

    relabelled = TIP_LABEL.sub(replace, newick)
    if unlabelled and missing == "error":
        raise ValueError(f"{len(unlabelled)} tips have no label, e.g. {unlabelled[:3]}")
    Path(tree_out).parent.mkdir(parents=True, exist_ok=True)
    Path(tree_out).write_text(relabelled)
    return Path(tree_out)


# --- Comparison --------------------------------------------------------------

def _splits(tree):
    tips = frozenset(leaf.name for leaf in tree.get_terminals())
    splits = set()
    for node in tree.get_nonterminals():
        side = frozenset(leaf.name for leaf in node.get_terminals())
        if 1 < len(side) < len(tips) - 1:
            splits.add(min(side, tips - side, key=sorted))
    return tips, splits


def clade_recovery(filename):
    """Whether each group (code before the first "_") forms a clade.

    Rooting is ignored: a group counts as a clade if it is one side of a
    bipartition. Returns one dict per group with at least two tips: ``group``,
    ``tips``, ``clade`` and ``foreign``, the tips of other groups in the smallest
    clade (either side of a bipartition) that holds the whole group.
    """
    tips, splits = _splits(Phylo.read(filename, "newick"))
    sides = [side for split in splits for side in (split, tips - split)]
    groups = {}
    for tip in tips:
        groups.setdefault(GROUP_PATTERN.match(tip).group(1), set()).add(tip)
    rows = []
    for group, members in sorted(groups.items()):
        if len(members) < 2 or len(members) > len(tips) - 2:
            continue
        smallest = min((side for side in sides if members <= side), key=len, default=tips)
        rows.append({"group": group, "tips": len(members), "clade": smallest == members,
                     "foreign": sorted(smallest - members)})
    return rows


def _expand_truncated(tree, full_names):
    """Rename tips cut off by QuickTree (100 characters) to the one full name they start."""
    expanded = 0
    for leaf in tree.get_terminals():
        if leaf.name not in full_names:
            matches = [name for name in full_names if name.startswith(leaf.name)]
            if len(matches) == 1:
                leaf.name = matches[0]
                expanded += 1
    return expanded


def compare_trees(tree_a, tree_b, digits=5):
    """Compare two Newick files with the same tip names.

    Tip names that older workflow runs truncated to 100 characters are matched
    to the full names of the other tree first. Returns a dict with: same tip set,
    number of truncated names matched, Robinson-Foulds distance (number of
    bipartitions found in only one tree; rooting is ignored), and whether the
    sorted branch lengths agree after rounding to ``digits``.
    """
    a, b = Phylo.read(tree_a, "newick"), Phylo.read(tree_b, "newick")
    names_a = {leaf.name for leaf in a.get_terminals()}
    names_b = {leaf.name for leaf in b.get_terminals()}
    truncated = _expand_truncated(b, names_a) + _expand_truncated(a, names_b)
    tips_a, splits_a = _splits(a)
    tips_b, splits_b = _splits(b)
    lengths = [sorted(round(clade.branch_length or 0.0, digits) for clade in tree.find_clades()
                      if clade is not tree.root) for tree in (a, b)]
    return {
        "same_tips": tips_a == tips_b,
        "tips": len(tips_a),
        "truncated_names_matched": truncated,
        "robinson_foulds": len(splits_a ^ splits_b) if tips_a == tips_b else None,
        "splits": len(splits_a),
        "same_branch_lengths": lengths[0] == lengths[1],
    }
