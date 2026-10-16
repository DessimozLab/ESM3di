"""Compatibility alias for structure preparation."""
import sys
from esm3di_training import prepare_data as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
