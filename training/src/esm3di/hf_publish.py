"""Compatibility alias for publishing; requires the training distribution."""
import sys
from esm3di_training import hf_publish as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
