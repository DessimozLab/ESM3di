"""Backward-compatible entry point for MLM fine-tuning (``esm3di-train-mlm``).

Forwards to :mod:`esm3di.train_mlm`. Invocations using flags only the original
script understood are forwarded to :mod:`esm3di.experimental.esmretrain_mlm_full`.
"""

import argparse
import sys
from typing import Optional, Sequence

from . import train_mlm


def main(argv: Optional[Sequence[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = set(train_mlm.add_train_mlm_args(argparse.ArgumentParser(add_help=False))._option_string_actions)
    if any(a.startswith("--") and a.split("=", 1)[0] not in known for a in argv):
        print("Note: options not supported by esm3di.train_mlm; using esm3di.experimental.esmretrain_mlm_full")
        from .experimental import esmretrain_mlm_full

        sys.argv = [sys.argv[0]] + argv
        return esmretrain_mlm_full.main()
    return train_mlm.main(argv)


if __name__ == "__main__":
    main()
