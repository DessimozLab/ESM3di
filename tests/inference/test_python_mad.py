from importlib.resources import files
import subprocess
import sys

from Bio import Phylo


def test_packaged_python_mad_roots_tree(tmp_path):
    workflow = files("esm3di.workflow")
    assert not workflow.joinpath("madroot/mad").is_file()
    assert not workflow.joinpath("madroot/mad_macos").is_file()
    source = tmp_path / "tree.nwk"
    source.write_text("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);\n")
    result = subprocess.run(
        [sys.executable, str(workflow.joinpath("madroot/mad.py")), str(source), "-n"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rooted = Phylo.read(str(source) + ".rooted", "newick")
    assert {leaf.name for leaf in rooted.get_terminals()} == {"A", "B", "C", "D"}
    assert all(clade.branch_length >= 0 for clade in rooted.find_clades() if clade.branch_length is not None)
    rules = workflow.joinpath("rules/tree_building.smk").read_text()
    assert 'python "{params.mad}"' in rules
