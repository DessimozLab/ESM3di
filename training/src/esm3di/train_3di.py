"""Compatibility alias for the supported 3Di trainer."""

import sys
from esm3di_training import train_3di as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
