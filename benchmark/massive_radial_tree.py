
#!/usr/bin/env python3
"""Fast, approximate circular or unrooted-style phylograms for large Newick trees.

Install: python -m pip install numpy matplotlib
Run:     python massive_radial_tree.py tree.nwk metadata.tsv -o tree.svg
Server:  python massive_radial_tree.py tree.nwk metadata.tsv -o tree.svg --no-show
Unrooted: python massive_radial_tree.py tree.nwk metadata.tsv --layout unrooted --theme dark
PNG:     python massive_radial_tree.py tree.nwk metadata.tsv -o tree.png --dpi 600

Design
------
* Iterative Newick parser and array-based traversals; no recursion limit changes.
* Finite negative branch lengths are accepted and preserved: those edges run
  inward. If cumulative distances become negative, all plotted radii receive
  the same offset. The input root then lies away from the center and is marked.
* Metadata is streamed once, retaining only uVIGs found in the tree. TSV really
  means tab-delimited, not whitespace-delimited. Plain and .gz files supported.
* Auto coloring evaluates phylum, class, order and family, balances taxonomic
  resolution against assignment coverage, and reports the selected rank.
* Optional --prune-unassigned removes tips without an assignment at that rank
  in the selected taxonomy column. Auto rank selection happens BEFORE pruning.
  Layout, branch colors and legend percentages then use all RETAINED tips.
  The input files are not modified; the original root and path lengths remain.
* Optional --max-branch-length X cuts each original non-root edge longer than
  X, removing ALL descendant tips of that edge. It runs before metadata/rank
  selection, unassigned pruning and drawing reduction. Equality is retained;
  negative lengths are retained. This tests individual Newick edge lengths,
  not cumulative root-to-tip distances or summed lengths of contracted paths.
  It uses input lengths even with --length-mode topology, and requires a
  length on every non-root edge. Remaining edge lengths are never truncated.
* Optional --max-root-to-tip-factor F removes tips whose cumulative distance
  from the input root is > F times the arithmetic mean across tips. Compute
  the mean ONCE, after any --max-branch-length filtering but before taxonomy
  selection/unassigned pruning. Uses signed input branch lengths even in
  topology drawing mode; requires complete lengths and a positive mean.
  This is tip filtering, not an individual-edge cutoff or iterative trimming.
* At most --max-tips representatives are drawn. Consecutive bins in original
  leaf order each contribute their farthest tip (ties favor the bin midpoint).
  Their induced subtree is drawn after suppressing unary nodes. Original tip
  angles and root distances are retained; small clades can be omitted visually.
* Branch colors use ALL original descendant tips, including tips not drawn.
  A color requires --purity agreement, including unknown tips in the denominator.
  Grouping rare taxa as Other happens AFTER purity is evaluated, so different
  rare taxa cannot combine to produce a falsely homogeneous branch.
* Legend percentages count protein tips, not unique uVIGs. A uVIG can map to
  multiple protein tips. Missing ranks are not filled from a different rank.
* Output is tree-only SVG plus a separate <output_stem>_legend.svg. Use -o
  tree.png for a raster tree instead, with --dpi (default 600). No tree SVG is
  generated in PNG mode. Tree dimensions are --size inches square; 12 inches
  at 600 dpi gives 7200 x 7200 pixels. The legend always stays editable SVG.
  Neither output has a title; plot metadata appears above single-line legend
  entries. --legend-output overrides the automatically chosen legend filename.
* Taxonomy z-order, back to front: unassigned/unmatched, Other assigned taxa,
  mixed branches, then named groups from smallest to largest. Counts include
  all tips remaining after optional pruning, before drawing reduction. Each
  group's dots sit above its own branches but below every foreground group.
* --layout unrooted uses a linear-time equal-angle placement of the reduced
  tree: each clade's angular sector is proportional to its original tip count;
  straight edge lengths are the signed sums along contracted paths. Negative
  lengths reverse an edge's direction. This display does not reroot the tree.
  It shows tip dots by default; adjust their diameter with --tip-size.

Memory is O(N + U), plus a few O(N) temporary arrays, independent of metadata
row count. Runtime is O(B + M + N*log(N) + K), with B Newick characters,
M metadata characters, and K drawn tips. Taxon-purity checks use leaf-order
range queries, not N-by-taxon matrices.
The input root and tip order are used unchanged; no biological root is inferred.

Use --max-tips 0 to draw every tip, --rank class to fix the rank, and --purity 1
to color only unanimous clades. Run --help for all options. matplotlib.show()
opens a window when a GUI backend/display is available; use --no-show on HPC.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import math
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass

import numpy as np


RANKS = {"realm": "r", "kingdom": "k", "phylum": "p", "class": "c",
         "order": "o", "family": "f", "genus": "g", "species": "s"}
AUTO_RANKS = ("phylum", "class", "order", "family")
MISSING = {"", "\\N", "NA", "N/A", "null", "None", "nan"}
TOKEN = re.compile(r"\s+|\[[^\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s(),:;\[\]']+")
# Distinct categorical colors, in a fixed order (not a continuous gradient).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#332288", "#882255", "#44AA99", "#AA4499", "#999933", "#661100",
           "#6699CC", "#117733", "#DDCC77", "#AA4466", "#4477AA", "#66CCEE",
           "#228833", "#EE6677"]
UNKNOWN_COLOR, OTHER_COLOR, MIXED_COLOR = "#CCCCCC", "#8C8C8C", "#555555"


class Progress:
    def __init__(self, quiet=False):
        self.start = self.last = time.perf_counter()
        self.quiet = quiet

    def log(self, message):
        if not self.quiet:
            print(f"[{time.perf_counter() - self.start:8.1f}s] {message}",
                  file=sys.stderr, flush=True)

    def tick(self, stage, current, total=None):
        now = time.perf_counter()
        if now - self.last >= 1.0:
            self.last = now
            suffix = f" / {total:,} ({current / total:.0%})" if total else ""
            self.log(f"{stage}: {current:,}{suffix}")


def open_text(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


@dataclass
class Tree:
    parent: np.ndarray
    length: np.ndarray
    tips: np.ndarray
    keys: list[str]


def read_newick(path, progress):
    """Read one Newick tree; support polytomies, labels, quotes and comments.

    Internal labels/supports and bracket annotations are ignored. Standard
    single-quoted labels (including doubled quote escapes) are accepted.
    Nested square-bracket comments and NEXUS wrappers are not supported.
    Node indices are preorder: parents always precede children.
    """
    progress.log(f"Reading Newick: {path}")
    with open_text(path) as handle:
        text = handle.read()
    parents, lengths, tips, keys, stack = [], [], [], [], []
    state, last, end, token_count = "subtree", -1, 0, 0

    def new_node():
        node = len(parents)
        parents.append(stack[-1] if stack else -1)
        lengths.append(float("nan"))
        return node

    for match in TOKEN.finditer(text):
        if match.start() != end:
            raise ValueError(f"Invalid Newick near character {end:,}.")
        end = match.end()
        token = match.group()
        if token.isspace() or token.startswith("["):
            continue
        token_count += 1
        if token_count % 32768 == 0:
            progress.tick("Parsing Newick characters", end, len(text))
        if state == "ended":
            raise ValueError("Expected one Newick tree; found content after ';'.")
        if state == "length":
            try:
                value = float(token)
            except ValueError:
                raise ValueError(f"Invalid branch length {token!r}.") from None
            if not math.isfinite(value):
                raise ValueError(f"Branch lengths must be finite: {token!r}.")
            lengths[last] = value
            state = "after_length"
        elif token == "(":
            if state != "subtree":
                raise ValueError(f"Unexpected '(' near character {end:,}.")
            last = new_node()
            stack.append(last)
        elif token == ",":
            if state == "subtree" or not stack:
                raise ValueError(f"Unexpected ',' near character {end:,}.")
            state = "subtree"
        elif token == ")":
            if state == "subtree" or not stack:
                raise ValueError(f"Unexpected ')' near character {end:,}.")
            last = stack.pop()
            state = "internal_label"
        elif token == ":":
            if state not in ("suffix", "internal_label"):
                raise ValueError(f"Unexpected ':' near character {end:,}.")
            state = "length"
        elif token == ";":
            if stack or state == "subtree":
                raise ValueError("Unbalanced or empty Newick tree.")
            state = "ended"
        elif state == "subtree":
            last = new_node()
            label = token[1:-1].replace("''", "'") if token.startswith("'") else token
            key = label.split("|", 1)[0].strip()
            if not key:
                raise ValueError("A tip has an empty uVIG identifier.")
            tips.append(last)
            keys.append(key)
            state = "suffix"
        elif state == "internal_label":
            state = "suffix"  # Internal node name/support is intentionally ignored.
        else:
            raise ValueError(f"Unexpected Newick token {token!r} near character {end:,}.")
    if end != len(text) or state != "ended" or not tips:
        raise ValueError("Incomplete Newick: require one nonempty tree ending in ';'.")
    if len(tips) < 2:
        raise ValueError("At least two tips are required for a radial tree.")
    tree = Tree(np.asarray(parents, dtype=np.int32),
                np.asarray(lengths, dtype=np.float64),
                np.asarray(tips, dtype=np.int32), keys)
    progress.log(f"Parsed {len(tips):,} tips and {len(parents):,} total nodes.")
    return tree


def index_tree(tree, mode, progress):
    """Compute root distances and inclusive leaf-order intervals in O(N)."""
    n, nt = len(tree.parent), len(tree.tips)
    missing = np.isnan(tree.length[1:])
    if mode == "branch" and missing.all():
        progress.log("No branch lengths: using unit-edge topology distances.")
        mode = "topology"
    elif mode == "branch" and missing.any():
        raise ValueError(f"{missing.sum():,} edges lack lengths; use --length-mode topology "
                         "or supply complete branch lengths.")
    distance = np.zeros(n, dtype=np.float64)
    left = np.full(n, nt, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    left[tree.tips] = right[tree.tips] = np.arange(nt, dtype=np.int32)
    progress.log(f"Indexing full tree ({mode} distances; input root retained).")
    if mode == "branch":
        negative_count = int(np.count_nonzero(tree.length[1:] < 0))
        if negative_count:
            progress.log(f"Preserving {negative_count:,} negative branch lengths (inward edges).")
    for node in range(1, n):
        parent = tree.parent[node]
        distance[node] = distance[parent] + (1.0 if mode == "topology" else tree.length[node])
        if node % 32768 == 0:
            progress.tick("Root distances", node, n)
    if not np.isfinite(distance).all():
        raise ValueError("Root-to-tip distances overflowed; check the branch lengths.")
    if distance.max() == distance.min():
        raise ValueError("All distances are zero; use --length-mode topology.")
    for node in range(n - 1, 0, -1):
        parent = tree.parent[node]
        if left[node] < left[parent]:
            left[parent] = left[node]
        if right[node] > right[parent]:
            right[parent] = right[node]
        if node % 32768 == 0:
            progress.tick("Descendant intervals", n - node, n)
    return distance, left, right, mode


def read_metadata(path, keys, id_column, tax_column, progress):
    """Stream lines; keep only requested IDs and intern repeated taxonomies."""
    wanted = dict.fromkeys(keys)  # Unique uVIGs, independent of protein tip count.
    pool, matched, duplicate_rows, rows = {"": ""}, 0, 0, 0
    progress.log(f"Scanning metadata for {len(wanted):,} unique uVIGs: {path}")
    with open_text(path) as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        header = [x.strip() for x in header]
        for name in (id_column, tax_column):
            if header.count(name) != 1:
                raise ValueError(f"Metadata must have exactly one {name!r} column. "
                                 "The file must be tab-delimited.")
        id_idx, tax_idx = header.index(id_column), header.index(tax_column)
        last_idx = max(id_idx, tax_idx)
        for rows, line in enumerate(handle, 1):
            if rows % 65536 == 0:
                progress.tick("Metadata rows scanned", rows)
            if not line.strip():
                continue
            # IMG/VR has the ID first: reject irrelevant rows before splitting
            # their many other columns. This matters for multi-million-row TSVs.
            if id_idx == 0:
                tab = line.find("\t")
                if tab < 0:
                    raise ValueError(f"Metadata row {rows + 1:,} is not tab-delimited.")
                key = line[:tab].strip()
                if key not in wanted:
                    continue
            # Split just enough columns to extract ID and taxonomy, not all 25.
            fields = line.rstrip("\r\n").split("\t", last_idx + 1)
            if len(fields) <= last_idx:
                raise ValueError(f"Metadata row {rows + 1:,} has too few TSV columns.")
            key = fields[id_idx].strip()
            if key not in wanted:
                continue
            tax = fields[tax_idx].strip()
            if tax in MISSING:
                tax = ""
            previous = wanted[key]
            if previous is not None:
                duplicate_rows += 1
                if previous != tax:
                    raise ValueError(f"Conflicting taxonomy rows for uVIG {key!r}.")
            else:
                wanted[key] = pool.setdefault(tax, tax)
                matched += 1
    matched_tips = sum(wanted[k] is not None for k in keys)
    progress.log(f"Metadata: {rows:,} rows; matched {matched:,}/{len(wanted):,} uVIGs; "
                 f"{matched_tips:,}/{len(keys):,} protein tips ({matched_tips / len(keys):.1%}).")
    if duplicate_rows:
        progress.log(f"Ignored {duplicate_rows:,} identical duplicate metadata rows.")
    if not matched:
        raise ValueError("No tree tip matched the metadata uVIG column. Check delimiters and IDs.")
    return wanted


def parse_taxonomy(text):
    result = {}
    for item in text.split(";"):
        prefix, sep, name = item.strip().partition("__")
        name = name.strip()
        if sep and prefix in RANKS.values() and name not in MISSING:
            result[prefix] = name
    return result


def choose_taxonomy(tree, metadata, rank, max_colors, min_coverage, progress):
    """Choose a single rank; return original (ungrouped) taxonomy IDs per tip."""
    weights = collections.Counter(metadata[key] or "" for key in tree.keys)
    parsed = {tax: parse_taxonomy(tax) for tax in weights}
    nt = len(tree.tips)

    def counts_at(name):
        prefix = RANKS[name]
        counts = collections.Counter()
        for tax, weight in weights.items():
            label = parsed[tax].get(prefix)
            if label:
                counts[label] += weight
        return counts

    candidates = AUTO_RANKS if rank == "auto" else (rank,)
    summaries = []
    progress.log("Taxonomic resolution (all tips after any branch-length pruning):")
    for name in candidates:
        counts = counts_at(name)
        assigned = sum(counts.values())
        coverage = assigned / nt
        ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        # Coverage in individually named colors penalizes long rare-taxon tails.
        shown = [count / nt for _, count in ordered[:max_colors]]
        remainder = 1 - sum(shown)
        entropy = -sum(p * math.log(p) for p in shown if p)
        if remainder > 0:
            entropy -= remainder * math.log(remainder)
        score = sum(shown) * entropy
        progress.log(f"  {name:7s}: {len(counts):6,d} taxa; "
                     f"assigned {coverage:6.1%}; named-color coverage {sum(shown):6.1%}")
        summaries.append((name, counts, coverage, score))
    if rank == "auto":
        eligible = [s for s in summaries if s[2] >= min_coverage]
        if not eligible:
            best_coverage = max(s[2] for s in summaries)
            eligible = [s for s in summaries if s[2] == best_coverage]
            progress.log(f"No rank reaches {min_coverage:.0%} assignment; using best available coverage.")
        # On ties prefer class, then order, phylum, family; never realm/species.
        preference = {"class": 3, "order": 2, "phylum": 1, "family": 0}
        chosen = max(eligible, key=lambda s: (round(s[3], 12), preference[s[0]]))
    else:
        chosen = summaries[0]
    rank, counts, coverage, _ = chosen
    if not counts:
        progress.log(f"WARNING: no assignments at rank {rank}; tree will be gray.")
    progress.log(f"Selected rank: {rank} ({coverage:.1%} assigned).")
    # Original exact taxon IDs, 0 for no assignment at this rank.
    labels = [""] + sorted(counts)
    label_id = {label: i for i, label in enumerate(labels)}
    tax_id = {tax: label_id[parsed[tax].get(RANKS[rank], "")] for tax in weights}
    tip_ids = np.fromiter((tax_id[metadata[key] or ""] for key in tree.keys),
                          dtype=np.int32, count=nt)
    return rank, counts, labels, tip_ids


def summarize_colors(tree, tip_ids, left, right, labels, counts,
                     max_colors, purity, progress):
    """Find strict majority candidates, then verify exact descendant counts.

    A weighted Boyer-Moore pass finds the only possible >50% candidate at each
    node. Searchsorted queries against (taxon ID, leaf index) pairs then verify
    each candidate's frequency. This also handles many thousands of rare taxa
    without a node-by-taxon count matrix.
    """
    progress.log(f"Computing full-tree branch colors (purity >= {purity:.0%}).")
    n, nt = len(tree.parent), len(tree.tips)
    candidate = np.zeros(n, dtype=np.int32)
    balance = np.zeros(n, dtype=np.int32)
    candidate[tree.tips], balance[tree.tips] = tip_ids, 1
    for node in range(n - 1, 0, -1):
        p = tree.parent[node]
        if candidate[p] == candidate[node]:
            balance[p] += balance[node]
        elif balance[p] >= balance[node]:
            balance[p] -= balance[node]
        else:
            candidate[p] = candidate[node]
            balance[p] = balance[node] - balance[p]
        if node % 32768 == 0:
            progress.tick("Taxonomy majority candidates", n - node, n)
    stride = nt + 1
    keyed_tips = np.sort(tip_ids.astype(np.int64) * stride + np.arange(nt))
    base = candidate.astype(np.int64) * stride
    frequency = (np.searchsorted(keyed_tips, base + right + 1)
                 - np.searchsorted(keyed_tips, base + left))
    homogeneous = frequency / (right - left + 1) >= purity
    top = sorted(counts, key=lambda label: (-counts[label], label))[:max_colors]
    colors = [MIXED_COLOR, UNKNOWN_COLOR, OTHER_COLOR] + PALETTE[:len(top)]
    tax_to_color = np.full(len(labels), 2, dtype=np.int16)
    tax_to_color[0] = 1
    label_id = {label: idx for idx, label in enumerate(labels)}
    for idx, label in enumerate(top, 3):
        tax_to_color[label_id[label]] = idx
    node_colors = np.where(homogeneous, tax_to_color[candidate], 0).astype(np.int16)
    entries = [(label, counts[label], colors[i]) for i, label in enumerate(top, 3)]
    other = sum(counts.values()) - sum(counts[label] for label in top)
    unknown = nt - sum(counts.values())
    if other:
        entries.append(("Other assigned taxa", other, OTHER_COLOR))
    if unknown:
        entries.append(("Unassigned / unmatched", unknown, UNKNOWN_COLOR))
    return node_colors, colors, entries


def prune_unassigned(tree, tip_ids, progress):
    """Remove unknown tips and empty ancestors in O(N), preserving the root.

    All ancestors of retained tips are kept, so branch and topology distances
    remain unchanged. Unary paths are contracted later by the drawing reducer.
    The rank and the nonzero taxon IDs chosen before pruning are unchanged.
    """
    assigned = tip_ids != 0
    retained = int(assigned.sum())
    removed = len(tree.tips) - retained
    if retained < 2:
        raise ValueError(f"--prune-unassigned leaves {retained:,} assigned tips; "
                         "at least two are required. Try a broader --rank, "
                         "another --taxonomy-column, or omit --prune-unassigned.")
    progress.log(f"Pruning unassigned/unmatched tips: removed {removed:,}; "
                 f"retained {retained:,}/{len(tree.tips):,} tips.")
    if not removed:
        return tree, tip_ids, 0
    return retain_tips(tree, assigned, progress), tip_ids[assigned], removed


def retain_tips(tree, keep_tips, progress):
    """Keep selected tips and all their ancestors, preserving every path length."""
    n = len(tree.parent)
    active = np.zeros(n, dtype=bool)
    active[tree.tips[keep_tips]] = True
    for node in range(n - 1, 0, -1):
        if active[node]:
            active[tree.parent[node]] = True
        if node % 32768 == 0:
            progress.tick("Pruning empty subtrees", n - node, n)
    nodes = np.flatnonzero(active)
    remap = np.full(n, -1, dtype=np.int32)
    remap[nodes] = np.arange(len(nodes), dtype=np.int32)
    parent = remap[tree.parent[nodes]]
    parent[0] = -1
    return Tree(parent, tree.length[nodes], remap[tree.tips[keep_tips]],
                [key for key, keep in zip(tree.keys, keep_tips) if keep])


def prune_long_branches(tree, maximum, progress):
    """Cut edges > maximum and remove their descendant clades in O(N)."""
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("--max-branch-length must be finite and >= 0.")
    if np.isnan(tree.length[1:]).any():
        raise ValueError("--max-branch-length requires a length on every non-root "
                         "Newick edge, including when --length-mode topology is used.")
    n = len(tree.parent)
    blocked = np.zeros(n, dtype=bool)
    cut_edges = 0
    for node in range(1, n):
        if blocked[tree.parent[node]]:
            blocked[node] = True
        elif tree.length[node] > maximum:
            blocked[node] = True
            cut_edges += 1
        if node % 32768 == 0:
            progress.tick("Checking long branches", node, n)
    keep = ~blocked[tree.tips]
    retained = int(keep.sum())
    removed = len(tree.tips) - retained
    progress.log(f"Branch cutoff > {maximum:g}: cut {cut_edges:,} branches; "
                 f"removed {removed:,} descendant tips; retained {retained:,}/{len(tree.tips):,} tips.")
    if retained < 2:
        raise ValueError(f"--max-branch-length {maximum:g} leaves {retained:,} tips; "
                         "at least two are required. Increase the cutoff or omit this option.")
    if not removed:
        return tree, 0
    return retain_tips(tree, keep, progress), removed


def prune_root_to_tip(tree, factor, progress):
    """Filter tips by a fixed multiple of the pre-filter mean root distance."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("--max-root-to-tip-factor must be finite and > 0.")
    if np.isnan(tree.length[1:]).any():
        raise ValueError("--max-root-to-tip-factor requires a length on every non-root Newick edge.")
    n = len(tree.parent)
    distance = np.zeros(n, dtype=np.float64)
    progress.log("Computing input root-to-tip distances for relative pruning.")
    for node in range(1, n):
        distance[node] = distance[tree.parent[node]] + tree.length[node]
        if node % 32768 == 0:
            progress.tick("Relative-pruning distances", node, n)
    if not np.isfinite(distance).all():
        raise ValueError("Root-to-tip distances overflowed; check input branch lengths.")
    tip_distance = distance[tree.tips]
    # Scale the reduction to avoid overflowing a sum of large finite lengths.
    magnitude = float(np.max(np.abs(tip_distance)))
    mean = float(np.mean(tip_distance / magnitude)) * magnitude if magnitude else 0.0
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("Relative root-to-tip pruning requires a positive mean distance. "
                         "The signed input branch lengths give a nonpositive mean.")
    cutoff = factor * mean
    if not math.isfinite(cutoff):
        raise ValueError("Relative pruning cutoff overflowed; reduce the factor or branch lengths.")
    keep = tip_distance <= cutoff
    retained = int(keep.sum())
    removed = len(tree.tips) - retained
    progress.log(f"Root-to-tip mean = {mean:.6g}; cutoff = {factor:g} x mean = {cutoff:.6g} "
                 f"(one pass; {len(tree.tips):,} tips before filtering).")
    progress.log(f"Relative pruning: removed {removed:,}; retained {retained:,} tips.")
    if retained < 2:
        raise ValueError(f"--max-root-to-tip-factor {factor:g} leaves {retained:,} tips; "
                         "at least two are required. Increase the factor or omit this option.")
    if not removed:
        return tree, 0, mean, cutoff
    return retain_tips(tree, keep, progress), removed, mean, cutoff


def reduce_tree(tree, distance, left, right, max_tips, progress):
    """Choose representatives and contract unary paths, without recursion.

    Only geometry is thinned. Node colors are calculated before this step from
    full descendant sets. Internal angles use full descendant leaf spans.
    Summed path lengths are implicit in the original root-distance coordinates.
    """
    nt, n = len(tree.tips), len(tree.parent)
    k = nt if max_tips == 0 else min(max_tips, nt)
    edges = np.arange(k + 1, dtype=np.int64) * nt // k
    tip_dist = distance[tree.tips]
    selected = np.empty(k, dtype=np.int32)
    for idx in range(k):
        lo, hi = int(edges[idx]), int(edges[idx + 1])
        values = tip_dist[lo:hi]
        ties = np.flatnonzero(values == values.max())
        midpoint = (hi - lo - 1) / 2
        selected[idx] = lo + ties[np.argmin(np.abs(ties - midpoint))]
    selected_nodes = tree.tips[selected]
    active = np.zeros(n, dtype=bool)
    active[selected_nodes] = True
    active_children = np.zeros(n, dtype=np.int32)
    for node in range(n - 1, 0, -1):
        if active[node]:
            p = tree.parent[node]
            active[p] = True
            active_children[p] += 1
        if node % 32768 == 0:
            progress.tick("Reducing drawing", n - node, n)
    keep = active & (active_children != 1)
    keep[0] = True  # Preserve the supplied root even if it has a single active child.
    nearest = np.full(n, -1, dtype=np.int32)
    nearest[0] = 0
    draw_parent = np.full(n, -1, dtype=np.int32)
    for node in range(1, n):
        if active[node]:
            ancestor = nearest[tree.parent[node]]
            if keep[node]:
                draw_parent[node] = ancestor
                nearest[node] = node
            else:
                nearest[node] = ancestor
    nodes = np.flatnonzero(keep)
    progress.log(f"Drawing {k:,}/{nt:,} tip representatives and {len(nodes):,} retained nodes.")
    if k < nt:
        progress.log("Approximation: one farthest tip per leaf-order bin; small clades may be omitted.")
    return nodes, draw_parent, selected_nodes


def unrooted_coordinates(distance, left, right, nodes, draw_parent, nt, rotation=0.0):
    """Equal-angle embedding of the reduced tree, with exact signed path lengths.

    Original descendant intervals specify wedge boundaries, so thinning does
    not change clade angular weights. Preorder node IDs permit one forward
    pass. Return normalized xy coordinates and branch-length units per xy unit.
    The input root is a traversal origin, not an inferred biological root.
    """
    unit = float(distance.max()) - float(distance.min())
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("Invalid distance range for unrooted layout.")
    angle = (math.pi / 2 + math.radians(rotation)
             + (left.astype(float) + right + 1) * (math.pi / nt))
    xy = np.zeros((len(distance), 2), dtype=float)
    for node in nodes:
        if node == 0:
            continue
        parent = draw_parent[node]
        length = (distance[node] - distance[parent]) / unit
        xy[node, 0] = xy[parent, 0] + length * math.cos(angle[node])
        xy[node, 1] = xy[parent, 1] + length * math.sin(angle[node])
    drawn = xy[nodes]
    if not np.isfinite(drawn).all():
        raise ValueError("Unrooted coordinates overflowed; check branch-length magnitudes.")
    lower, upper = drawn.min(axis=0), drawn.max(axis=0)
    half_span = float(np.max(upper - lower)) / 2
    scale = unit * half_span
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("The reduced tree has no drawable spatial extent; "
                         "try --max-tips 0 or --length-mode topology.")
    xy[nodes] = (drawn - (lower + upper) / 2) / half_span
    return xy, scale


def add_taxonomy_layers(ax, segments, edge_colors, arcs, arc_colors,
                        tip_positions, tip_colors, rgba, linewidth, tip_size):
    """Batch by taxonomy so branch, arc and marker artists share a group layer.

    Color indices 0/1/2 mean mixed/unknown/other. Indices >=3 are already
    ordered by decreasing full-tip count (alphabetically for count ties).
    At most three collections per color keep artist overhead bounded.
    """
    from matplotlib.collections import LineCollection

    back_to_front = [1, 2, 0] + list(range(len(rgba) - 1, 2, -1))
    for layer, color_id in enumerate(back_to_front, 1):
        z = float(layer)
        edge_mask = edge_colors == color_id
        if np.any(edge_mask):
            collection = LineCollection(segments[edge_mask], colors=[rgba[color_id]],
                                        linewidths=linewidth, alpha=0.92, zorder=z)
            collection.set_gid(f"taxon-{color_id}-branches")
            ax.add_collection(collection)
        if arcs:
            indices = np.flatnonzero(arc_colors == color_id)
            if len(indices):
                collection = LineCollection([arcs[i] for i in indices], colors=[rgba[color_id]],
                                            linewidths=linewidth, alpha=0.92, zorder=z + 0.05)
                collection.set_gid(f"taxon-{color_id}-arcs")
                ax.add_collection(collection)
        if tip_size:
            tip_mask = tip_colors == color_id
            if np.any(tip_mask):
                dots = ax.scatter(tip_positions[tip_mask, 0], tip_positions[tip_mask, 1],
                                  s=tip_size ** 2, c=[rgba[color_id]], edgecolors="none",
                                  zorder=z + 0.2)
                dots.set_gid(f"taxon-{color_id}-tips")
    return len(back_to_front) + 1.0


def make_legend_figure(entries, mixed_color, nt, metadata_lines, purity,
                       background, foreground, secondary):
    """Independent editable legend with metadata above single-line entries."""
    import matplotlib.pyplot as plt

    rows = [(f"{label} - {count:,} ({count / nt:.1%})", color)
            for label, count, color in entries]
    rows.append((f"Mixed branches - <{purity:.0%} agreement", mixed_color))
    # Positions are specified in inches. Tight SVG bounds also accommodate
    # unusually long taxon names without wrapping entries or clipping text.
    meta_step, row_step = 0.19, 0.25
    width = max(4.5, max([len(t) for t in metadata_lines] + [len(t) for t, _ in rows]) * 0.065 + 0.75)
    height = 0.45 + len(metadata_lines) * meta_step + len(rows) * row_step
    fig = plt.figure(figsize=(width, height), facecolor=background)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    y = height - 0.20
    for line in metadata_lines:
        ax.text(0.15, y, line, ha="left", va="top", fontsize=9, color=secondary)
        y -= meta_step
    y -= 0.16
    for text, color in rows:
        ax.plot([0.15, 0.40], [y, y], color=color, lw=3, solid_capstyle="butt")
        ax.text(0.53, y, text, ha="left", va="center", fontsize=10, color=foreground)
        y -= row_step
    return fig


def output_paths(args):
    tree_output = Path(args.output).expanduser()
    legend_output = (Path(args.legend_output).expanduser() if args.legend_output
                     else tree_output.with_name(tree_output.stem + "_legend.svg"))
    return tree_output, legend_output


def render(tree, distance, left, right, nodes, draw_parent, selected_nodes,
           node_colors, colors, entries, rank, mode, args, progress):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    progress.log(f"Building {args.layout} phylogram with batched line collections.")
    nt = len(tree.tips)
    unassigned_pruned = getattr(args, "pruned_tips", 0)
    long_pruned = getattr(args, "long_pruned_tips", 0)
    relative_pruned = getattr(args, "relative_pruned_tips", 0)
    pruned_tips = unassigned_pruned + long_pruned + relative_pruned
    edge_nodes = nodes[nodes != 0]
    parents = draw_parent[edge_nodes]
    arcs = []
    arc_nodes = np.empty(0, dtype=np.int32)
    radial_offset = 0.0
    if args.layout == "unrooted":
        xy, rmax = unrooted_coordinates(distance, left, right, nodes, draw_parent,
                                        nt, args.rotation)
        segments = np.stack((xy[parents], xy[edge_nodes]), axis=1)
        tip_positions = xy[selected_nodes]
    else:
        gap = math.radians(args.gap)
        span = 2 * math.pi - gap
        start = math.pi / 2 + gap / 2 + math.radians(args.rotation)
        theta = start + (left.astype(float) + right + 1) * (span / (2 * nt))
        # Offset avoids negative polar radii flipping nodes across the circle.
        radial_offset = max(0.0, -float(distance.min()))
        rmax = float(distance.max()) + radial_offset
        if not math.isfinite(rmax) or rmax <= 0:
            raise ValueError("Invalid radial distance range; check branch-length magnitudes.")
        radius = (distance + radial_offset) / rmax
        if radial_offset:
            progress.log(f"Applying radial offset +{radial_offset:.6g} branch-length units; "
                         "signed edge lengths are preserved and the input root is marked.")
        ct, st = np.cos(theta), np.sin(theta)
        segments = np.empty((len(edge_nodes), 2, 2), dtype=float)
        segments[:, 0, 0] = radius[parents] * ct[edge_nodes]
        segments[:, 0, 1] = radius[parents] * st[edge_nodes]
        segments[:, 1, 0] = radius[edge_nodes] * ct[edge_nodes]
        segments[:, 1, 1] = radius[edge_nodes] * st[edge_nodes]
        tip_positions = np.column_stack((radius[selected_nodes] * ct[selected_nodes],
                                         radius[selected_nodes] * st[selected_nodes]))
        min_angle, max_angle = theta.copy(), theta.copy()
        np.minimum.at(min_angle, parents, theta[edge_nodes])
        np.maximum.at(max_angle, parents, theta[edge_nodes])
        arc_nodes = nodes[(max_angle[nodes] > min_angle[nodes]) & (radius[nodes] > 0)]
        for node in arc_nodes:
            delta = max_angle[node] - min_angle[node]
            count = max(2, int(math.ceil(delta / math.radians(1.5))) + 1)
            angle = np.linspace(min_angle[node], max_angle[node], count)
            arcs.append(np.column_stack((radius[node] * np.cos(angle),
                                         radius[node] * np.sin(angle))))
    dark = args.theme == "dark"
    background = "#111111" if dark else "white"
    foreground = "#EEEEEE" if dark else "#111111"
    secondary = "#BBBBBB" if dark else "#555555"
    plot_colors = list(colors)
    if dark:
        plot_colors[0] = "#A0A0A0"  # Mixed branches stay visible on a dark background.
    rgba = np.asarray([matplotlib.colors.to_rgba(c) for c in plot_colors])
    matplotlib.rcParams["svg.fonttype"] = "none"  # Keep legend text editable.
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(args.size, args.size), facecolor=background)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    tip_size = args.tip_size if args.tip_size is not None else (3.0 if args.layout == "unrooted" else 0.0)
    progress.log("Layering back to front: unassigned, other, mixed, named taxa (smallest to largest).")
    annotation_z = add_taxonomy_layers(
        ax, segments, node_colors[edge_nodes], arcs, node_colors[arc_nodes],
        tip_positions, node_colors[selected_nodes], rgba, args.linewidth, tip_size)
    if radial_offset:
        ax.plot(radius[0] * ct[0], radius[0] * st[0], "o", color=foreground,
                markersize=4, zorder=annotation_z)
    ax.set(xlim=(-1.06, 1.06), ylim=(-1.13, 1.06), aspect="equal")
    ax.axis("off")
    # Scale refers to radial segments (circular) or straight edges (unrooted).
    ticks = MaxNLocator(nbins=4).tick_values(0, rmax)
    options = ticks[(ticks > 0) & (ticks <= rmax / 3)]
    scale = float(options[-1]) if len(options) else rmax / 5
    ax.plot([-1.0, -1.0 + scale / rmax], [-1.075, -1.075], color=foreground, lw=1.5)
    unit = "branch-length units" if mode == "branch" else "edges"
    ax.text(-1.0, -1.10, f"{scale:.3g} {unit}", ha="left", va="top", fontsize=9, color=foreground)
    metadata_lines = [
        f"Tree: {Path(args.tree).name}",
        f"Tips: {nt + pruned_tips:,} input | {nt:,} retained | {len(selected_nodes):,} drawn",
        f"Taxonomy: {args.taxonomy_column} | Rank: {rank}",
        f"Layout: {args.layout} | Lengths: {mode} | Branch agreement: >= {args.purity:.0%}",
        "Percentages: all retained tips, before drawing reduction",
        "Root: input Newick; no rerooting",
    ]
    if args.max_branch_length is not None:
        metadata_lines.append(f"Edge cutoff: > {args.max_branch_length:g} | Removed: {long_pruned:,}")
    if args.max_root_to_tip_factor is not None:
        metadata_lines.append(
            f"Root-to-tip cutoff: {args.max_root_to_tip_factor:g} x mean = {args.root_to_tip_cutoff:.6g}"
            f" | Removed: {relative_pruned:,}")
        metadata_lines.append(f"Root-to-tip mean: {args.root_to_tip_mean:.6g} (fixed before relative pruning)")
    if args.prune_unassigned:
        metadata_lines.append(f"Unassigned/unmatched removed: {unassigned_pruned:,}")
    if radial_offset:
        metadata_lines.append(f"Radial offset: +{radial_offset:.6g}; root marked by a dot")
    if len(selected_nodes) < nt:
        metadata_lines.append("Drawing: farthest representative per angular bin; unary paths contracted")
    legend_fig = make_legend_figure(entries, plot_colors[0], nt, metadata_lines,
                                    args.purity, background, foreground, secondary)

    output, legend_output = output_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    legend_output.parent.mkdir(parents=True, exist_ok=True)
    tree_format = output.suffix.lower()[1:]
    if tree_format == "png":
        pixels = round(args.size * args.dpi)
        progress.log(f"Writing tree PNG: {output} ({pixels:,} x {pixels:,} pixels; {args.dpi} dpi)")
    else:
        progress.log(f"Writing tree SVG: {output}")
    # Fixed square canvas ensures PNG dimensions are exactly size * dpi.
    fig.savefig(output, format=tree_format, dpi=args.dpi, facecolor=background)
    progress.log(f"Saved {output} ({output.stat().st_size / 1e6:.2f} MB).")
    progress.log(f"Writing separate legend SVG: {legend_output}")
    legend_fig.savefig(legend_output, format="svg", bbox_inches="tight",
                       pad_inches=0.05, facecolor=background)
    progress.log(f"Saved {legend_output} ({legend_output.stat().st_size / 1e3:.1f} KB).")
    if args.no_show:
        plt.close(fig)
        plt.close(legend_fig)
    else:
        progress.log("Opening separate tree and legend windows; close them to exit.")
        plt.show()
    return output, legend_output



def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("tree", help="Input Newick (.nwk or .nwk.gz).")
    parser.add_argument("metadata", help="Input tab-delimited metadata (.tsv or .tsv.gz).")
    parser.add_argument("-o", "--output", default="radial_tree.svg",
                        help="Tree-only output (.svg or .png). A separate *_legend.svg is always written.")
    parser.add_argument("--legend-output", help="Separate legend SVG filename; default: <output_stem>_legend.svg.")
    parser.add_argument("--dpi", type=int, default=600, help="Tree PNG resolution in dots per inch.")
    parser.add_argument("--layout", choices=["circular", "unrooted"], default="circular",
                        help="Circular with arcs, or equal-angle unrooted-style with straight branches.")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Plot color theme.")
    parser.add_argument("--tip-size", type=float,
                        help="Tip dot diameter in points; 0 hides dots. Default: 3 for unrooted, 0 for circular.")
    parser.add_argument("--rotation", type=float, default=0.0, help="Counterclockwise rotation in degrees.")
    parser.add_argument("--rank", choices=["auto", *RANKS], default="auto",
                        help="Coloring rank. Auto compares phylum/class/order/family.")
    parser.add_argument("--max-tips", type=int, default=6000,
                        help="Maximum representative tips; 0 draws all tips.")
    parser.add_argument("--max-colors", type=int, default=14,
                        help="Maximum named taxon colors (1-20); rarer taxa become Other.")
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="Minimum assigned-tip fraction for an auto rank (0-1).")
    parser.add_argument("--purity", type=float, default=0.9,
                        help="Minimum same-taxon descendant fraction to color a branch (>0.5, <=1).")
    parser.add_argument("--length-mode", choices=["branch", "topology"], default="branch",
                        help="Use signed branch lengths, or assign length 1 to every original edge.")
    parser.add_argument("--id-column", default="uvig", help="Metadata identifier column.")
    parser.add_argument("--taxonomy-column", default="ictv_taxonomy", help="Metadata taxonomy column.")
    parser.add_argument("--prune-unassigned", action="store_true",
                        help="Remove tips unassigned/unmatched at the selected rank before plotting. "
                             "Auto rank is chosen before pruning; colors and percentages are recalculated.")
    parser.add_argument("--max-branch-length", type=float,
                        help="Cut original Newick edges longer than this value and remove all their "
                             "descendant tips. Runs before taxonomy selection. Default: no length pruning.")
    parser.add_argument("--max-root-to-tip-factor", type=float,
                        help="Remove tips with input root-to-tip distance > FACTOR times the mean "
                             "(e.g. 2). Mean is computed once after any absolute branch filtering, "
                             "before taxonomy selection/unassigned pruning. Default: disabled.")
    parser.add_argument("--gap", type=float, default=4.0, help="Circular-layout angular gap in degrees (0-90).")
    parser.add_argument("--size", type=float, default=12.0, help="Tree panel size in inches (>=6).")
    parser.add_argument("--linewidth", type=float, default=0.4, help="Branch width in points.")
    parser.add_argument("--no-show", action="store_true", help="Save both outputs without GUI windows.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress; still print the legend.")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.max_tips < 0 or args.max_tips == 1:
        parser.error("--max-tips must be 0 (all) or >=2")
    if not 1 <= args.max_colors <= len(PALETTE):
        parser.error("--max-colors must be between 1 and 20")
    if not 0.5 < args.purity <= 1:
        parser.error("--purity must be >0.5 and <=1")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1")
    if args.tip_size is not None and not 0 <= args.tip_size <= 40:
        parser.error("--tip-size must be between 0 and 40")
    if not math.isfinite(args.rotation):
        parser.error("--rotation must be finite")
    if args.max_branch_length is not None and (not math.isfinite(args.max_branch_length)
                                               or args.max_branch_length < 0):
        parser.error("--max-branch-length must be finite and >= 0")
    if args.max_root_to_tip_factor is not None and (not math.isfinite(args.max_root_to_tip_factor)
                                                   or args.max_root_to_tip_factor <= 0):
        parser.error("--max-root-to-tip-factor must be finite and > 0")
    if not 0 <= args.gap <= 90 or not 6 <= args.size <= 100 or not 0 < args.linewidth <= 10:
        parser.error("Require --gap in [0,90], --size in [6,100], --linewidth in (0,10]")
    if args.dpi <= 0:
        parser.error("--dpi must be a positive integer")
    output, legend_output = output_paths(args)
    if output.suffix.lower() not in (".svg", ".png"):
        parser.error("--output must have an .svg or .png extension")
    if legend_output.suffix.lower() != ".svg":
        parser.error("--legend-output must have an .svg extension")
    inputs = {Path(args.tree).expanduser().resolve(), Path(args.metadata).expanduser().resolve()}
    if output.resolve() in inputs or legend_output.resolve() in inputs:
        parser.error("Both outputs must differ from the input paths")
    if output.resolve() == legend_output.resolve():
        parser.error("Tree output and legend output must be different files")
    progress = Progress(args.quiet)
    try:
        tree = read_newick(args.tree, progress)
        args.long_pruned_tips = 0
        if args.max_branch_length is not None:
            tree, args.long_pruned_tips = prune_long_branches(tree, args.max_branch_length, progress)
        args.relative_pruned_tips = 0
        if args.max_root_to_tip_factor is not None:
            tree, args.relative_pruned_tips, args.root_to_tip_mean, args.root_to_tip_cutoff = prune_root_to_tip(
                tree, args.max_root_to_tip_factor, progress)
        metadata = read_metadata(args.metadata, tree.keys, args.id_column,
                                 args.taxonomy_column, progress)
        rank, counts, labels, tip_ids = choose_taxonomy(
            tree, metadata, args.rank, args.max_colors, args.min_coverage, progress)
        args.pruned_tips = 0
        if args.prune_unassigned:
            tree, tip_ids, args.pruned_tips = prune_unassigned(tree, tip_ids, progress)
        distance, left, right, mode = index_tree(tree, args.length_mode, progress)
        node_colors, colors, entries = summarize_colors(
            tree, tip_ids, left, right, labels, counts, args.max_colors, args.purity, progress)
        if args.theme == "dark":
            # Lighten categorical colors for contrast; keep console and SVG
            # legend hex codes identical, including the mixed-branch color.
            def lighten(color):
                return "#" + "".join(f"{round(int(color[i:i + 2], 16) * 0.75 + 255 * 0.25):02X}"
                                     for i in (1, 3, 5))
            adjusted = {color: lighten(color) for color in PALETTE}
            adjusted[MIXED_COLOR] = "#A0A0A0"
            colors = [adjusted.get(color, color) for color in colors]
            entries = [(label, count, adjusted.get(color, color)) for label, count, color in entries]
        del metadata, tip_ids
        nodes, draw_parent, selected = reduce_tree(
            tree, distance, left, right, args.max_tips, progress)
        tip_scope = "retained" if args.pruned_tips or args.long_pruned_tips or args.relative_pruned_tips else "original"
        print(f"\nLegend ({rank}; ALL {len(tree.tips):,} {tip_scope} protein tips)", flush=True)
        for label, count, color in entries:
            print(f"{color}  {label} - {count:,} ({count / len(tree.tips):.1%})")
        mixed_color = "#A0A0A0" if args.theme == "dark" else MIXED_COLOR
        print(f"{mixed_color}  Mixed branches - <{args.purity:.0%} agreement", flush=True)
        render(tree, distance, left, right, nodes, draw_parent, selected,
               node_colors, colors, entries, rank, mode, args, progress)
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
