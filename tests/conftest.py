"""Make the repository root importable for the tests that need it.

Most tests exercise the installed ``esm3di`` and ``esm3di_training`` packages, but
``test_polyprotein_cut_training`` also imports the analysis code under ``experiments/``,
which ships with the repository rather than with either wheel. The release workflow runs
pytest from a different working directory, so the repository root is not on ``sys.path``
there and those imports would fail.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
