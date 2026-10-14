"""Compatibility alias for the checkpoint inference database writer."""

import sys
from .inference import foldseek_db as _implementation

sys.modules[__name__] = _implementation
