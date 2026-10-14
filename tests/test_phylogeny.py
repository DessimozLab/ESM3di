import gzip
from pathlib import Path
import subprocess

import pytest

from esm3di.inference import cli, phylogeny


def make_database(tmp_path):
    database = tmp_path / "db"
    for component in ("db", "db_ss", "db_h"):
        for suffix in ("", ".index", ".dbtype"):
            (tmp_path / f"{component}{suffix}").write_bytes(b"test")
    return database


@pytest.fixture
def external_tools(monkeypatch):
    calls = []
    monkeypatch.setattr(phylogeny.shutil, "which", lambda name: f"/tools/{name}")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "structuremsa":
            Path(command[3] + "_aa.fa").write_text(">a\nACDE\n>b\nACDF\n")
        else:
            assert kwargs["stdin"].read() == b">a\nACDE\n>b\nACDF\n"
            kwargs["stdout"].write(b"(a:0.1,b:0.2);\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(phylogeny.subprocess, "run", run)
    monkeypatch.setattr(
        cli.ESM3DiPredictor, "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Phylogeny must not load model weights"),
    )
    return calls


def test_combined_cli_uses_fast_mode_and_aa_alignment(tmp_path, external_tools):
    database = make_database(tmp_path)
    output = tmp_path / "run"
    cli.main(["foldmason-fasttree", "--input-db", str(database),
              "--output-dir", str(output), "--threads", "3"])
    foldmason, fasttree = external_tools
    assert foldmason[0] == [
        "/tools/foldmason", "structuremsa", str(database), str(output / "alignment"),
        "--threads", "3", "--fast", "1", "--refine-iters", "0",
    ]
    assert fasttree[0] == ["/tools/FastTreeMP", "-fastest", "-noml", "-spr", "0", "-nosupport"]
    assert fasttree[1]["env"]["OMP_NUM_THREADS"] == "3"
    assert (output / "tree.nwk").read_text() == "(a:0.1,b:0.2);\n"
    assert "structuremsa" in (output / "run.txt").read_text()
    assert (output / "foldmason.log").is_file()


def test_foldmason_only_and_no_overwrite(tmp_path, external_tools):
    database = make_database(tmp_path)
    output = tmp_path / "run"
    args = ["foldmason", "--input-db", str(database), "--output-dir", str(output)]
    cli.main(args)
    assert len(external_tools) == 1
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 1
    assert len(external_tools) == 1


@pytest.mark.parametrize("compressed", [False, True])
def test_fasttree_cli_supports_gzip_and_no_overwrite(tmp_path, external_tools, compressed):
    alignment = tmp_path / ("alignment.fa.gz" if compressed else "alignment.fa")
    data = b">a\nACDE\n>b\nACDF\n"
    alignment.write_bytes(gzip.compress(data) if compressed else data)
    output = tmp_path / "tree.nwk"
    args = ["fasttree", "--input-alignment", str(alignment), "--output-tree", str(output)]
    cli.main(args)
    assert len(external_tools) == 1
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 1
    assert output.read_bytes() == b"(a:0.1,b:0.2);\n"


def test_missing_database_does_not_create_output(tmp_path, external_tools):
    with pytest.raises(FileNotFoundError, match="Missing database component"):
        phylogeny.run_foldmason(tmp_path / "missing", tmp_path / "run", 1, "foldmason")
    assert not (tmp_path / "run").exists()
    assert not external_tools


def test_binary_resolution_overrides_and_missing(monkeypatch):
    monkeypatch.setenv("FASTTREE_BIN", "/custom/FastTreeMP")
    monkeypatch.setattr(phylogeny.shutil, "which", lambda name: name)
    assert phylogeny.resolve_binary(None, "FASTTREE_BIN", "FastTreeMP") == "/custom/FastTreeMP"
    assert phylogeny.resolve_binary("/explicit", "FASTTREE_BIN", "FastTreeMP") == "/explicit"
    monkeypatch.setattr(phylogeny.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="conda install"):
        phylogeny.resolve_binary(None, "FASTTREE_BIN", "FastTreeMP")


@pytest.mark.parametrize("command", ["fasttree", "foldmason", "foldmason-fasttree"])
def test_invalid_threads_rejected(command):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--threads", "0"])
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["exit", "empty"])
def test_failed_fasttree_preserves_log_not_partial_tree(tmp_path, monkeypatch, failure):
    alignment = tmp_path / "alignment.fa"
    alignment.write_text(">a\nACDE\n>b\nACDF\n")
    output = tmp_path / "tree.nwk"

    def run(command, **kwargs):
        if failure == "exit":
            kwargs["stdout"].write(b"partial")
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)

    monkeypatch.setattr(phylogeny.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="see"):
        phylogeny.run_fasttree(alignment, output, 1, "FastTreeMP")
    assert not output.exists()
    assert Path(f"{output}.log").exists()
    assert set(tmp_path.iterdir()) == {alignment, Path(f"{output}.log")}


@pytest.mark.parametrize("failure", ["exit", "empty"])
def test_failed_foldmason_keeps_diagnostics(tmp_path, monkeypatch, failure):
    database = make_database(tmp_path)
    monkeypatch.setattr(
        phylogeny.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1 if failure == "exit" else 0),
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="see"):
        phylogeny.run_foldmason(database, output, 1, "foldmason")
    assert (output / "foldmason.log").exists()
    assert (output / "run.txt").exists()
