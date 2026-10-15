"""Compatibility alias for inference-release preprocessing."""

import sys
from .inference import preprocessing as _implementation

sys.modules[__name__] = _implementation
