"""Compatibility alias; requires the training distribution."""
import sys
from esm3di_training import losses as _implementation
sys.modules[__name__] = _implementation
