from importlib.resources import files
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Bio import Phylo
from esm3di.workflow.scripts import run_madroot


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
    assert '"../scripts/run_madroot.py"' in rules


@pytest.mark.parametrize("newick", [
    "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);",
    "(A:1,B:1,C:1,D:1);",
    "((A:1,B:1):1,(C:1,D:1):1);",
])
def test_wrapper_handles_ties_polytomies_and_paths_with_spaces(tmp_path, newick):
    source = tmp_path / "input tree.nwk"
    source.write_text(newick + "\n")
    log = tmp_path / "logs" / "root.log"
    output = run_madroot.run_mad_root(source, log=log)
    tree = Phylo.read(output, "newick")
    assert {tip.name for tip in tree.get_terminals()} == {"A", "B", "C", "D"}
    assert source.read_text() == newick + "\n"
    assert "Rooted tree" in log.read_text()
    assert not list(tmp_path.glob(".mad-*"))


@pytest.mark.parametrize("body, message", [
    ("Path(sys.argv[1] + '.rooted').write_text(';\\n')", "error marker"),
    ("Path(sys.argv[1] + '.rooted').write_text('(X:1,Y:1,Z:1);')", "changed tip labels"),
    ("Path(sys.argv[1] + '.rooted').write_text('(A:1,B:1,C:1,D:-1);')", "invalid branch lengths"),
    ("Path(sys.argv[1] + '.rooted').write_text('(A:1,B:1,C:1,D:1);\\n(A:1,B:1,C:1,D:1);')", "invalid single-tree"),
    ("pass", "no rooted tree"),
    ("sys.exit(3)", "exit 3"),
])
def test_wrapper_rejects_failed_outputs_without_replacing_existing_tree(tmp_path, body, message):
    source = tmp_path / "input.nwk"
    source.write_text("(A:1,B:1,C:1,D:1);")
    output = tmp_path / "saved.rooted"
    output.write_text("previous result")
    fake = tmp_path / "fake mad.py"
    fake.write_text("import sys\nfrom pathlib import Path\nprint('MAD diagnostics')\n" + body + "\n")
    log = tmp_path / "mad.log"
    with pytest.raises(RuntimeError, match=message):
        run_madroot.run_mad_root(source, output, mad_script=fake, log=log)
    assert output.read_text() == "previous result"
    assert "MAD diagnostics" in log.read_text()
    assert not list(tmp_path.glob(".mad-*"))


def test_real_mad_error_marker_is_rejected(tmp_path):
    source = tmp_path / "duplicate tips.nwk"
    source.write_text("(A:1,A:1,C:1,D:1);")
    with pytest.raises(RuntimeError, match="error marker"):
        run_madroot.run_mad_root(source)
    assert not Path(str(source) + ".rooted").exists()


def test_snakemake_script_entry_point_uses_configured_python_mad(tmp_path):
    source = tmp_path / "input.nwk"
    source.write_text("(A:1,B:1,C:1,D:1);")
    output = tmp_path / "workflow.rooted"
    log = tmp_path / "workflow.log"
    workflow = files("esm3di.workflow")
    script = str(workflow.joinpath("scripts/run_madroot.py"))
    runpy.run_path(script, init_globals={"snakemake": SimpleNamespace(
        input=[str(source)], output=[str(output)], log=[str(log)],
        params=SimpleNamespace(mad=str(workflow.joinpath("madroot/mad.py"))),
    )})
    assert len(Phylo.read(output, "newick").get_terminals()) == 4
    assert log.is_file()


def test_archived_notebook_uses_the_same_python_wrapper(tmp_path):
    notebook = Path(__file__).resolve().parents[2] / "experiments/analyses/flavi_redo/simple_flavi/folded_vsesm3di.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    code = next("".join(cell["source"]) for cell in cells if "def run_madroot" in "".join(cell["source"]))
    namespace = {}
    exec(code, namespace)
    source = tmp_path / "notebook tree.nwk"
    source.write_text("(A:1,B:1,C:1,D:1);")
    output = namespace["run_madroot"](str(source))
    assert len(Phylo.read(output, "newick").get_terminals()) == 4
