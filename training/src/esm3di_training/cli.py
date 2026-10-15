"""esm3di command line interface.

Subcommands:

    esm3di prepare    structures -> train/val(/test) AA, 3Di and pLDDT-bin FASTAs (needs FoldSeek)
    esm3di mlm        stage 1: unsupervised masked-LM fine-tuning of ESM++ on sequences
    esm3di train      stage 2: supervised 3Di tagging (LoRA + CNN head)          [alias: 3di]
    esm3di predict    3Di FASTA (+ optional FoldSeek DB) from amino-acid FASTA
    esm3di foldseekdb AA FASTA + 3Di FASTA -> FoldSeek DB (no model)
    esm3di pipeline   prepare -> [mlm] -> train in one command

Every training subcommand accepts --config file.json; explicit flags override it.
Run `esm3di <subcommand> --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from esm3di.checkpoints import DEFAULT_HF_MODEL


# ---------------------------------------------------------------------------
# Simple subcommands
# ---------------------------------------------------------------------------

def _cmd_prepare(ns):
    from .prepare_data import run_prepare

    run_prepare(ns)


def _cmd_mlm(ns):
    from . import train_mlm

    train_mlm.train(train_mlm.resolve_config(ns))


def _cmd_train(ns):
    from . import train_3di

    train_3di.train(train_3di.resolve_config(ns))


def _cmd_predict(ns):
    from esm3di.predict import run_predict

    run_predict(ns)


def _add_foldseekdb_args(p):
    p.add_argument("--aa-fasta", required=True)
    p.add_argument("--three-di-fasta", required=True)
    p.add_argument("--output-db", "-o", required=True, help="Output FoldSeek DB prefix.")


def _cmd_foldseekdb(ns):
    from esm3di.foldseek_db import fasta2foldseek

    fasta2foldseek(ns.aa_fasta, ns.three_di_fasta, ns.output_db)
    print(f"Wrote FoldSeek DB {ns.output_db}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _add_pipeline_args(p):
    g = p.add_argument_group("inputs")
    g.add_argument("--structures", "-s", nargs="+",
                   help="Structure dirs/files (.pdb/.cif[.gz]) or .txt path lists; converted with `esm3di prepare`.")
    g.add_argument("--data-dir", help="Already prepared data dir (output of `esm3di prepare`) instead of --structures.")
    g.add_argument("--sequences", nargs="+",
                   help="Unlabelled protein FASTA(s) for the MLM stage. Omit to skip MLM.")
    g.add_argument("--mlm-on-structure-sequences", action="store_true",
                   help="Also add the training-split sequences of the structures to the MLM corpus "
                        "(enables MLM when only structures are given).")
    g = p.add_argument_group("starting point")
    g.add_argument("--init-from", default=DEFAULT_HF_MODEL,
                   help="HF model id (e.g. Synthyra/ESMplusplus_small|large), an MLM checkpoint dir, "
                        f"or a 3Di .pt checkpoint (default {DEFAULT_HF_MODEL}). With a 3Di .pt the MLM stage is skipped.")
    g = p.add_argument_group("output / control")
    g.add_argument("--out-dir", "-o", required=True, help="Writes data/, mlm/ and 3di/ below this dir.")
    g.add_argument("--resume", action="store_true", help="Reuse prepared data and resume/skip finished stages.")
    g.add_argument("--device")
    g = p.add_argument_group("prepare options")
    g.add_argument("--foldseek-bin")
    g.add_argument("--val-fraction", type=float, default=0.02)
    g.add_argument("--test-fraction", type=float, default=0.0)
    g.add_argument("--cluster-split", action="store_true")
    g.add_argument("--no-plddt", action="store_true", help="Structures have no pLDDT in the B-factor column.")
    g.add_argument("--min-length", type=int, default=20)
    g.add_argument("--max-length", type=int)
    g = p.add_argument_group("MLM stage options")
    g.add_argument("--mlm-config", help="JSON config for the MLM stage (see configs/mlm_*.json).")
    g.add_argument("--mlm-epochs", type=int)
    g.add_argument("--mlm-batch-size", type=int)
    g.add_argument("--mlm-lr", type=float)
    g.add_argument("--mlm-max-train-samples", type=int)
    g = p.add_argument_group("3Di stage options")
    g.add_argument("--train-config", help="JSON config for the 3Di stage (see configs/3di_*.json).")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--max-train-samples", type=int)


def _is_3di_checkpoint(path: str) -> bool:
    return os.path.isfile(path) and path.endswith((".pt", ".pth", ".bin"))


def _flags(**kwargs) -> List[str]:
    argv: List[str] = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def _cmd_pipeline(ns):
    from . import train_3di, train_mlm
    from .prepare_data import data_files_from_dir, prepare

    if bool(ns.structures) == bool(ns.data_dir):
        raise SystemExit("pipeline: give exactly one of --structures or --data-dir")
    os.makedirs(ns.out_dir, exist_ok=True)

    # 1. data
    data_dir = ns.data_dir or os.path.join(ns.out_dir, "data")
    if ns.structures:
        if ns.resume and os.path.exists(os.path.join(data_dir, "manifest.json")):
            print(f"[pipeline] reusing prepared data in {data_dir}")
        else:
            print(f"[pipeline] preparing data from structures -> {data_dir}")
            prepare(
                ns.structures, data_dir, foldseek_bin=ns.foldseek_bin, use_plddt=not ns.no_plddt,
                val_fraction=ns.val_fraction, test_fraction=ns.test_fraction,
                min_length=ns.min_length, max_length=ns.max_length, cluster_split=ns.cluster_split,
            )
    files = data_files_from_dir(data_dir)

    # 2. MLM
    init_from = ns.init_from
    mlm_inputs = list(ns.sequences or [])
    if ns.mlm_on_structure_sequences:
        mlm_inputs.append(files["aa_fasta"])
    if mlm_inputs:
        if _is_3di_checkpoint(init_from):
            raise SystemExit("pipeline: --init-from is a 3Di checkpoint; MLM cannot start from it. "
                             "Drop --sequences or start from an HF model / MLM dir.")
        mlm_dir = os.path.join(ns.out_dir, "mlm")
        done = os.path.exists(os.path.join(mlm_dir, "final", "esm3di_mlm.json"))
        if ns.resume and done:
            print(f"[pipeline] MLM stage already finished in {mlm_dir}")
        else:
            print(f"[pipeline] MLM stage: {init_from} on {len(mlm_inputs)} FASTA(s) -> {mlm_dir}")
            argv = (["--config", ns.mlm_config] if ns.mlm_config else []) + _flags(
                input_fasta=mlm_inputs, model_name=init_from, output_dir=mlm_dir,
                epochs=ns.mlm_epochs, batch_size=ns.mlm_batch_size, learning_rate=ns.mlm_lr,
                max_train_samples=ns.mlm_max_train_samples, device=ns.device,
                resume="auto" if ns.resume else None,
            )
            parser = train_mlm.add_train_mlm_args(argparse.ArgumentParser())
            mlm_cfg = train_mlm.resolve_config(parser.parse_args(argv))
            # The pipeline owns the data: never pick up data paths from --mlm-config.
            mlm_cfg["input_txt"] = None
            mlm_cfg["val_fasta"] = [files["val_aa_fasta"]] if files.get("val_aa_fasta") else None
            mlm_cfg["test_fasta"] = None
            train_mlm.train(mlm_cfg)
        best = os.path.join(mlm_dir, "best")
        init_from = best if os.path.isdir(best) else os.path.join(mlm_dir, "final")

    # 3. 3Di
    tdi_dir = os.path.join(ns.out_dir, "3di")
    print(f"[pipeline] 3Di stage: init from {init_from} -> {tdi_dir}")
    argv = (["--config", ns.train_config] if ns.train_config else []) + _flags(
        data_dir=data_dir, init_from=init_from, out_dir=tdi_dir, epochs=ns.epochs,
        batch_size=ns.batch_size, lr=ns.lr, max_train_samples=ns.max_train_samples,
        device=ns.device, resume="auto" if ns.resume else None,
    )
    parser = train_3di.add_train_3di_args(argparse.ArgumentParser())
    tdi_cfg = train_3di.resolve_config(parser.parse_args(argv))
    # The pipeline owns data and output locations, whatever --train-config says.
    tdi_cfg.update(files)
    if tdi_cfg.get("tensorboard_log_dir"):
        tdi_cfg["tensorboard_log_dir"] = os.path.join(tdi_dir, "tensorboard")
    best_ckpt = train_3di.train(tdi_cfg)

    summary = {"data_dir": data_dir, "init_from_3di": init_from, "best_3di_checkpoint": best_ckpt}
    with open(os.path.join(ns.out_dir, "pipeline.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[pipeline] done. Best 3Di checkpoint: {best_ckpt}")
    print(f"  predict with: esm3di predict --checkpoint {best_ckpt} -i proteins.fasta -o proteins_3di.fasta")


# ---------------------------------------------------------------------------

def build_parser(include_inference: bool = False) -> argparse.ArgumentParser:
    from .prepare_data import add_prepare_args
    from .train_3di import add_train_3di_args
    from .train_mlm import add_train_mlm_args

    parser = argparse.ArgumentParser(
        prog="esm3di" if include_inference else "esm3di-learn",
        description="ESM++ learning: prepare structures, fine-tune MLM, and train 3Di.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name, adder, func, help_, aliases=()):
        sp = sub.add_parser(name, help=help_, description=help_, aliases=list(aliases))
        adder(sp)
        sp.set_defaults(func=func)

    add("prepare", add_prepare_args, _cmd_prepare, "Build training FASTAs from structures with FoldSeek.")
    add("mlm", add_train_mlm_args, _cmd_mlm, "Stage 1: masked-LM fine-tuning of ESM++ on sequences.")
    add("train", add_train_3di_args, _cmd_train, "Stage 2: supervised 3Di token training.", aliases=["3di"])
    if include_inference:
        from esm3di.predict import add_predict_args

        add("predict", add_predict_args, _cmd_predict, "Predict 3Di from amino-acid FASTA.")
        add("foldseekdb", _add_foldseekdb_args, _cmd_foldseekdb, "Write a FoldSeek DB from AA + 3Di FASTAs.")
    add("pipeline", _add_pipeline_args, _cmd_pipeline, "prepare -> [mlm] -> train in one command.")
    return parser


def main(argv: Optional[Sequence[str]] = None, include_inference: bool = False):
    ns = build_parser(include_inference=include_inference).parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main(sys.argv[1:])
