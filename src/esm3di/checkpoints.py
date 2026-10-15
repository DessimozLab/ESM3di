"""Compatibility alias for shared checkpoints."""
import sys
from ._core import checkpoints as _implementation
sys.modules[__name__] = _implementation
