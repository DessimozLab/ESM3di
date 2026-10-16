"""Run Python MAD and validate its single-tree output before publishing it."""

from collections import Counter
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from Bio import Phylo


def run_mad_root(treefile, output=None, mad_script=None, log=None):
    """Root one Newick tree with the current interpreter, preserving tip labels."""
    source = Path(treefile).resolve()
    destination = Path(output).resolve() if output else Path(str(source) + ".rooted")
    script = (
        Path(mad_script).resolve() if mad_script is not None
        else Path(__file__).resolve().parents[1] / "madroot" / "mad.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Python MAD script not found: {script}")
    original = Phylo.read(source, "newick")
    tips = Counter(tip.name for tip in original.get_terminals())
    if sum(tips.values()) < 3 or None in tips:
        raise ValueError("MAD requires at least three named tips in one Newick tree")
    if destination == source:
        raise ValueError("MAD output must not replace the input tree")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # MAD writes beside its input, and -n can return success with only ';'.
    with tempfile.TemporaryDirectory(prefix=".mad-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "input.nwk"
        shutil.copyfile(source, staged)
        result = subprocess.run(
            [sys.executable, str(script), str(staged), "-n"],
            capture_output=True, text=True,
        )
        diagnostics = result.stdout + result.stderr
        if log is not None:
            log_path = Path(log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(diagnostics)
        if result.returncode:
            raise RuntimeError(f"Python MAD failed (exit {result.returncode}):\n{diagnostics}")
        rooted_path = Path(str(staged) + ".rooted")
        if not rooted_path.is_file():
            raise RuntimeError(f"Python MAD produced no rooted tree:\n{diagnostics}")
        # MAD emits integer-valued lengths as '1.', which some Newick readers
        # misinterpret as a tip label. Keep the value, with an explicit decimal.
        newick = re.sub(r"(:[+-]?\d+)\.(?=[eE,);]|\s|$)", r"\g<1>.0", rooted_path.read_text())
        rooted_path.write_text(newick)
        try:
            rooted = Phylo.read(rooted_path, "newick")
        except ValueError as error:
            raise RuntimeError(f"Python MAD produced invalid single-tree Newick:\n{diagnostics}") from error
        if Counter(tip.name for tip in rooted.get_terminals()) != tips:
            raise RuntimeError(f"Python MAD produced an error marker or changed tip labels:\n{diagnostics}")
        if any(
            clade.branch_length is not None
            and (not math.isfinite(clade.branch_length) or clade.branch_length < 0)
            for clade in rooted.find_clades()
        ):
            raise RuntimeError(f"Python MAD produced invalid branch lengths:\n{diagnostics}")
        os.replace(rooted_path, destination)
    return str(destination)


if "snakemake" in globals():
    run_mad_root(
        snakemake.input[0], snakemake.output[0],
        mad_script=snakemake.params.mad, log=snakemake.log[0],
    )
