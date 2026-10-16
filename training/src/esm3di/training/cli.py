"""Compatibility alias for the supported learning CLI."""
import sys
from esm3di_training import cli as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
