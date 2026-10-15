"""Inference-first command with lazy compatibility routing for full installs."""

import sys


def build_parser():
    from esm3di_training.cli import build_parser as training_parser

    return training_parser(include_inference=True)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "predict" and "--checkpoint" in args:
        from .inference.checkpoint_predictor import main as checkpoint_main

        return checkpoint_main(args[1:])
    legacy = args and (
        args[0] in {"prepare", "mlm", "train", "3di", "pipeline", "foldseekdb"}
    )
    if legacy:
        try:
            from esm3di_training.cli import main as training_main
        except ModuleNotFoundError as error:
            if error.name != "esm3di_training":
                raise
            raise SystemExit("Training commands require esm3di-training; install 'esm3di[training]'.") from error
        return training_main(args, include_inference=True)
    from .inference.cli import main as inference_main

    return inference_main(args)


if __name__ == "__main__":
    main()
