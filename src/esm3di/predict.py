"""Compatibility alias for windowed training-checkpoint inference."""

import sys
from .inference import checkpoint_predictor as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
