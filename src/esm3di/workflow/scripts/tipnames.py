"""Short placeholder tip names for QuickTree.

QuickTree truncates sequence names to 100 characters, so the distance matrix is
written with placeholders (seq000000, seq000001, ...) and the original names are
restored in the Newick output.
"""

import re

PLACEHOLDER = re.compile(r"(?<=[(,])(seq\d{6})(?=[:,);])")


def placeholder_names(identifiers):
    """Return the placeholder for each identifier, in the same order."""
    return [f"seq{i:06d}" for i in range(len(identifiers))]


def write_name_map(identifiers, path):
    """Write a two-column TSV: placeholder, original name."""
    with open(path, "w") as handle:
        for short, name in zip(placeholder_names(identifiers), identifiers):
            handle.write(f"{short}\t{name}\n")
    return path


def read_name_map(path):
    with open(path) as handle:
        return dict(line.rstrip("\n").split("\t", 1) for line in handle if line.strip())


def restore_names(newick, name_map):
    """Replace every placeholder tip name in a Newick string by its original name."""
    found = PLACEHOLDER.findall(newick)
    if sorted(found) != sorted(name_map):
        raise ValueError(
            f"Tree has {len(found)} placeholder tips, name map has {len(name_map)} entries."
        )
    return PLACEHOLDER.sub(lambda match: name_map[match.group(1)], newick)
