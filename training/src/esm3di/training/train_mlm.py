"""Compatibility alias for the MLM trainer."""
import sys
from esm3di_training import train_mlm as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
