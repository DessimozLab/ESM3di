import gzip
import os
from pathlib import Path
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments/analyses/fasttree_NJ"


@pytest.fixture
def tools(tmp_path):
    binaries = tmp_path / "tools"
    binaries.mkdir()
    foldmason = binaries / "foldmason"
    foldmason.write_text(
        '#!/bin/bash\nset -eu\n'
        'if [[ $1 == version ]]; then echo test-version; exit 0; fi\n'
        '[[ $1 == structuremsa && $4 == --threads && $6 == --fast && $7 == 1 '
        '&& $8 == --refine-iters && $9 == 0 ]]\n'
        'printf ">a\\nACDE\\n>b\\nACDF\\n" > "${3}_aa.fa"\n'
    )
    fasttree = binaries / "FastTreeMP"
    fasttree.write_text(
        '#!/bin/bash\nset -eu\n'
        '[[ "$*" == "-fastest -noml -spr 0 -nosupport" ]]\n'
        '[[ $OMP_NUM_THREADS == 2 ]]\n'
        'data=$(cat)\n[[ $data == *ACDE* && $data == *ACDF* ]]\n'
        'echo "(a:0.1,b:0.2);"\n'
    )
    for binary in (foldmason, fasttree):
        binary.chmod(0o755)
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}")
    env.pop("FOLDMASON_BIN", None)
    env.pop("FASTTREE_BIN", None)
    return binaries, env


@pytest.mark.parametrize("override", [False, True])
def test_pipeline_path_and_explicit_overrides(tmp_path, tools, override):
    binaries, env = tools
    if override:
        for name, variable in (("foldmason", "FOLDMASON_BIN"), ("FastTreeMP", "FASTTREE_BIN")):
            path = binaries / name
            renamed = binaries / f"custom-{name}"
            path.rename(renamed)
            env[variable] = str(renamed)
    database = tmp_path / "db"
    for component in ("db", "db_ss", "db_h"):
        for suffix in ("", ".index", ".dbtype"):
            (tmp_path / f"{component}{suffix}").write_bytes(b"test")
    output = tmp_path / "run"
    command = ["bash", str(SCRIPTS / "run_foldmason_fasttree.sh"), str(database), str(output), "2"]
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert (output / "tree.nwk").read_text() == "(a:0.1,b:0.2);\n"
    settings = (output / "run.txt").read_text()
    assert env.get("FASTTREE_BIN", str(binaries / "FastTreeMP")) in settings
    result = subprocess.run(command, env=env, capture_output=True)
    assert result.returncode != 0
    assert (output / "tree.nwk").read_text() == "(a:0.1,b:0.2);\n"


def test_standalone_gzip_uses_path(tmp_path, tools):
    _, env = tools
    alignment = tmp_path / "alignment.fa.gz"
    alignment.write_bytes(gzip.compress(b">a\nACDE\n>b\nACDF\n"))
    output = tmp_path / "tree.nwk"
    subprocess.run(
        ["bash", str(SCRIPTS / "run_fasttree_nj_nni.sh"), str(alignment), str(output), "2"],
        env=env, check=True, capture_output=True,
    )
    assert output.read_text() == "(a:0.1,b:0.2);\n"


@pytest.mark.parametrize("variable", ["FOLDMASON_BIN", "FASTTREE_BIN"])
def test_invalid_override_fails_before_creating_run(tmp_path, tools, variable):
    _, env = tools
    env[variable] = str(tmp_path / "missing-binary")
    output = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_foldmason_fasttree.sh"), str(tmp_path / "db"), str(output), "2"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert variable in result.stderr
    assert not output.exists()
