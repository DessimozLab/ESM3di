import logging

import pytest

from esm3di.inference.cli import setup_logging


@pytest.mark.parametrize(
    ("verbosity", "expected_level"),
    [
        (0, logging.CRITICAL + 1),
        (1, logging.ERROR),
        (2, logging.WARNING),
        (3, logging.INFO),
    ],
)
def test_setup_logging_maps_verbosity_levels(verbosity, expected_level):
    setup_logging(verbosity)

    logger = logging.getLogger("esm3di")
    assert logger.level == expected_level
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_setup_logging_rejects_invalid_verbosity():
    with pytest.raises(ValueError, match="between 0 and 3"):
        setup_logging(4)


def test_foldtree_stages_packaged_workflow_and_uses_inference_cli(tmp_path, monkeypatch):
    from esm3di.inference import cli

    input_path = tmp_path / "input.fasta"
    input_path.write_text(">protein\nACDE\n")
    output_dir = tmp_path / "results"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        cli.main(["foldtree", "-i", str(input_path), "-o", str(output_dir), "--dry-run"])
    assert result.value.code == 0
    command, options = calls[0]
    assert command[0] == "snakemake"
    assert "--dry-run" in command
    assert options["cwd"] == output_dir / ".workflow"
    rules = (output_dir / ".workflow" / "rules" / "database_ops.smk").read_text()
    assert "esm3di-infer foldseek-db" in rules
