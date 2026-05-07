#!/usr/bin/env python
"""
Compute per-position perplexity from an ESM3Di checkpoint and AA FASTA input.

Example:
    python -m esm3di.perplexity_from_fasta \
        --model-ckpt checkpoints/epoch_3.pt \
        --input-fasta proteins_aa.fasta \
        --output-tsv proteins_perplexity.tsv
"""

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from .ESM3di_model import ESM3DiModel
from .model_outputs import PLDDT_BIN_VOCAB


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint at '{checkpoint_path}' is not a dict checkpoint. "
            "Expected keys like model_state_dict and args."
        )
    return checkpoint


def _build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[ESM3DiModel, str]:
    args_dict = checkpoint.get("args", {}) or {}

    hf_model_name = args_dict.get(
        "hf_model_name",
        args_dict.get("hf_model", "facebook/esm2_t33_650M_UR50D"),
    )
    label_vocab = checkpoint.get("label_vocab", []) or []
    if not label_vocab:
        raise ValueError(
            "Checkpoint is missing non-empty 'label_vocab', cannot infer num_labels."
        )

    plddt_label_vocab = checkpoint.get("plddt_label_vocab", PLDDT_BIN_VOCAB) or PLDDT_BIN_VOCAB
    aux_track_num_bins = args_dict.get("aux_track_num_bins", None) or checkpoint.get(
        "aux_track_num_bins", None
    )

    model = ESM3DiModel(
        hf_model_name=hf_model_name,
        num_labels=len(label_vocab),
        lora_r=args_dict.get("lora_r", 8),
        lora_alpha=args_dict.get("lora_alpha", 16),
        lora_dropout=args_dict.get("lora_dropout", 0.05),
        target_modules=checkpoint.get("lora_target_modules", None),
        use_cnn_head=args_dict.get("use_cnn_head", False),
        cnn_num_layers=args_dict.get("cnn_num_layers", 2),
        cnn_kernel_size=args_dict.get("cnn_kernel_size", 3),
        cnn_dropout=args_dict.get("cnn_dropout", 0.1),
        use_transformer_head=args_dict.get("use_transformer_head", False),
        transformer_head_dim=args_dict.get("transformer_head_dim", 256),
        transformer_head_layers=args_dict.get("transformer_head_layers", 2),
        transformer_head_dropout=args_dict.get("transformer_head_dropout", 0.1),
        transformer_head_num_heads=args_dict.get("transformer_head_num_heads", None),
        use_iterative_transformer_head=args_dict.get("use_iterative_transformer_head", False),
        iterative_head_max_iterations=args_dict.get("iterative_head_max_iterations", 5),
        iterative_head_halt_threshold=args_dict.get("iterative_head_halt_threshold", 0.95),
        iterative_head_lambda_p=args_dict.get("iterative_head_lambda_p", 0.01),
        iterative_head_prior_p=args_dict.get("iterative_head_prior_p", 0.5),
        use_positional_encoding=args_dict.get("use_positional_encoding", True),
        use_hidden_state_feedback=args_dict.get("use_hidden_state_feedback", True),
        use_gru_gate=args_dict.get("use_gru_gate", False),
        use_plddt_prediction_head=args_dict.get("use_plddt_prediction_head", False),
        plddt_num_bins=args_dict.get("plddt_num_bins", len(plddt_label_vocab) or 10),
        plddt_prediction_mode=args_dict.get("plddt_prediction_mode", "classification"),
        aux_track_num_bins=aux_track_num_bins,
    )

    return model, hf_model_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-position perplexity from AA FASTA using an ESM3Di checkpoint."
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--input-fasta",
        type=str,
        required=True,
        help="Input amino-acid FASTA file.",
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        required=True,
        help="Output TSV path for per-position perplexity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device override. Default: auto-detect.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print overall mean perplexity summary after export.",
    )
    return parser


def _compute_mean_perplexity(records: list) -> Optional[float]:
    total = 0.0
    count = 0
    for rec in records:
        values = rec.get("perplexity_per_position", [])
        total += float(sum(values))
        count += len(values)
    if count == 0:
        return None
    return total / count


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        checkpoint = _load_checkpoint(args.model_ckpt)
        model, hf_model_name = _build_model_from_checkpoint(checkpoint)

        print(f"Loaded checkpoint model config: {hf_model_name}")
        print(f"Input FASTA: {args.input_fasta}")
        print(f"Output TSV: {args.output_tsv}")

        records = model.output_per_position_perplexity_from_fasta(
            input_fasta_path=args.input_fasta,
            output_tsv_path=args.output_tsv,
            model_checkpoint_path=args.model_ckpt,
            batch_size=args.batch_size,
            device=args.device,
        )

        if args.summary:
            mean_ppl = _compute_mean_perplexity(records)
            if mean_ppl is None:
                print("Summary: no residues were exported.")
            else:
                print(f"Summary: mean per-position perplexity = {mean_ppl:.6f}")

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
