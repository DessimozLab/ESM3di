"""Backward-compatible entry point for 3Di training (``esm3di-train``).

Forwards to :mod:`esm3di.train_3di`, which accepts the old flag names and config
files used for the ESM++ MLM -> 3Di recipe. Invocations that use flags only the
original monolithic trainer understood (T5, iterative/transformer heads, focal
schedules, ...) are forwarded to :mod:`esm3di.experimental.esmretrain_full`.
"""

import argparse
import sys
from typing import Optional, Sequence

from . import train_3di


def _uses_unsupported_flags(argv: Sequence[str]) -> bool:
    parser = train_3di.add_train_3di_args(argparse.ArgumentParser(add_help=False))
    known = set(parser._option_string_actions)
    return any(a.startswith("--") and a.split("=", 1)[0] not in known for a in argv)


def main(argv: Optional[Sequence[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if _uses_unsupported_flags(argv):
        print("Note: options not supported by esm3di.train_3di; using esm3di.experimental.esmretrain_full")
        from .experimental import esmretrain_full

        sys.argv = [sys.argv[0]] + argv
        return esmretrain_full.main()
    return train_3di.main(argv)


if __name__ == "__main__":
    main()
