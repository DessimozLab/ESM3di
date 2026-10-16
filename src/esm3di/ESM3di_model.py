"""Compatibility alias for shared model architecture and training utilities."""
import sys
from ._core import model as _implementation
sys.modules[__name__] = _implementation
