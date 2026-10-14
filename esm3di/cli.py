"""Compatibility entry point for the original combined CLI."""

from .training.cli import build_parser as _build_parser, main as _main


def build_parser():
    return _build_parser(include_inference=True)


def main(argv=None):
    return _main(argv, include_inference=True)

if __name__ == "__main__":
    main()
