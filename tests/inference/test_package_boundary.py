import os
from pathlib import Path
import subprocess
import sys

import esm3di


def test_inference_help_without_training_or_analysis_dependencies():
    script = """
import builtins
import importlib.util
import sys

blocked = ('esm3di_training', 'esm3di.training', 'esm3di.experimental',
           'pandas', 'scipy', 'sklearn', 'tensorboard', 'sentencepiece')
def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
original_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    return None if forbidden(name) else original_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if forbidden(name):
        raise AssertionError('Inference imported optional module: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from esm3di.cli import main
from esm3di.inference.checkpoint_predictor import Predictor
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(forbidden(name) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(esm3di.__file__).parent.parent)},
    )
    assert result.returncode == 0, result.stderr
    assert "predict" in result.stdout and "cut" in result.stdout
    assert "pipeline" not in result.stdout and "mlm" not in result.stdout


def test_legacy_checkpoint_cli_does_not_require_training(monkeypatch):
    from esm3di import cli
    from esm3di.inference import checkpoint_predictor

    calls = []
    monkeypatch.setattr(checkpoint_predictor, "main", calls.append)
    cli.main(["predict", "--checkpoint", "best.pt", "-i", "in.fa", "-o", "out.fa"])
    assert calls == [["--checkpoint", "best.pt", "-i", "in.fa", "-o", "out.fa"]]
