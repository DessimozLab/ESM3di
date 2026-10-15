"""Compatibility alias for polyprotein cleavage inference."""

import sys
from .inference import cutting as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
