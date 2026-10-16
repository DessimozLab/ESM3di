"""CLI for cleavage prediction and polyprotein cutting using ESM3Di embeddings.

This tool:
1. Loads an ESM3Di checkpoint to compute per-residue embeddings.
2. Loads trained small classifier checkpoints (FFN and/or Transformer).
3. Predicts cleavage probability per residue with configurable cutoff.
4. Writes residue-level predictions and cleaved product FASTA outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import csv
import torch
import torch.nn as nn
from Bio import SeqIO

from .._core.model import ESM3DiModel


class SmallCutSiteFFN(nn.Module):
    """Per-position feed-forward classifier used in notebook training."""

    def __init__(self, d_in: int, d_hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCutSiteTransformer(nn.Module):
    """Sequence-aware transformer classifier used in notebook training."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        key_padding_mask = attn_mask == 0
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.head(h).squeeze(-1)
        return logits


def _forward_for_hidden_states(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Robust hidden-state extraction across wrapped models."""
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs
    except TypeError:
        # Some wrapped models reject wrapper-level kwargs; fall back to backbone below.
        pass

    # Wrapper outputs may omit hidden states; call the underlying backbone directly.
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        backbone = model.base_model.model
        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    raise RuntimeError("Could not retrieve hidden states from model")


def load_esm3di_embedding_model(
    checkpoint_path: str,
    device: str,
    requested_revision: Optional[str] = None,
):
    """Load ESM3Di model for embedding extraction based on a training checkpoint."""
    if Path(checkpoint_path).is_dir():
        from .cutting_bundle import load_bundle_checkpoint, read_bundle

        config = read_bundle(checkpoint_path)
        if config["format"] == "esm3di-cutting-v2":
            from huggingface_hub import snapshot_download
            from .predictor import ESM3DiPredictor

            source = config["embedding"]
            root = snapshot_download(
                repo_id=source["repo_id"], revision=source["repo_revision"],
                allow_patterns=["hf_compatible/*"],
            )
            predictor = ESM3DiPredictor(
                str(Path(root) / "hf_compatible"),
                revision=source["backbone_revision"], device=device,
            )
            return predictor.model, predictor.tokenizer, False

        ckpt = load_bundle_checkpoint(Path(checkpoint_path), "embedding")
    else:
        from .._core.checkpoints import read_3di_checkpoint

        ckpt = read_3di_checkpoint(checkpoint_path)
    from .._core.checkpoints import (
        get_tokenizer, load_state_into_model, lora_scaling_from_checkpoint,
        model_kwargs_from_checkpoint, strip_module_prefix,
    )

    kwargs = model_kwargs_from_checkpoint(ckpt)
    if requested_revision:
        kwargs["revision"] = requested_revision
    esm3di = ESM3DiModel(**kwargs)
    model = esm3di.get_model()
    load_state_into_model(
        model, strip_module_prefix(ckpt["model_state_dict"]),
        lora_scaling=lora_scaling_from_checkpoint(ckpt),
    )
    model.to(device).eval()
    tokenizer = get_tokenizer(esm3di, kwargs["hf_model_name"], kwargs.get("revision"))
    return model, tokenizer, False


def embed_sequences(
    model: nn.Module,
    tokenizer,
    sequences: Sequence[str],
    is_t5: bool,
    device: str,
    batch_size: int = 8,
) -> List[np.ndarray]:
    """Create per-sequence embedding matrices with shape (n_aa, d_embedding)."""
    if is_t5:
        raise ValueError("T5 embedding models are archived and unsupported")
    mats: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_raw = sequences[i:i + batch_size]
            batch_for_tok = batch_raw

            enc = tokenizer(
                batch_for_tok,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            special_mask = enc["special_tokens_mask"].to(device)

            outputs = _forward_for_hidden_states(model, input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError("Model output has no hidden_states")
            last_hidden = outputs.hidden_states[-1]

            for b_idx, seq in enumerate(batch_raw):
                aa_mask = (attention_mask[b_idx] == 1) & (special_mask[b_idx] == 0)
                emb = last_hidden[b_idx][aa_mask]
                if emb.shape[0] != len(seq):
                    n = min(emb.shape[0], len(seq))
                    emb = emb[:n]
                mats.append(emb.detach().cpu().numpy().astype(np.float32))

    return mats


def load_ffn_classifier(path: str, device: str) -> Tuple[nn.Module, dict]:
    if Path(path).is_dir():
        from .cutting_bundle import load_bundle_checkpoint

        ckpt = load_bundle_checkpoint(Path(path), "ffn")
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCutSiteFFN(
        d_in=int(ckpt["d_in"]),
        d_hidden=int(ckpt.get("d_hidden", 256)),
        dropout=float(ckpt.get("dropout", 0.25)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_transformer_classifier(path: str, device: str) -> Tuple[nn.Module, dict]:
    if Path(path).is_dir():
        from .cutting_bundle import load_bundle_checkpoint

        ckpt = load_bundle_checkpoint(Path(path), "transformer")
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCutSiteTransformer(
        d_in=int(ckpt["d_in"]),
        d_model=int(ckpt.get("d_model", 256)),
        nhead=int(ckpt.get("nhead", 8)),
        num_layers=int(ckpt.get("num_layers", 2)),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def predict_ffn_probs(model: nn.Module, emb: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(emb).float().to(device)).cpu().numpy().reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_transformer_probs(model: nn.Module, emb: np.ndarray, device: str) -> np.ndarray:
    n, d = emb.shape
    x_pad = np.zeros((1, n, d), dtype=np.float32)
    x_pad[0, :n, :] = emb
    attn_mask = np.ones((1, n), dtype=np.float32)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(x_pad).float().to(device),
            torch.from_numpy(attn_mask).float().to(device),
        ).cpu().numpy()[0]
    return 1.0 / (1.0 + np.exp(-logits))


def cleave_sequence(sequence: str, cut_positions_1based: Sequence[int]) -> List[Tuple[int, int, str]]:
    """Split sequence by cleavage positions and return (start, end, segment) tuples."""
    n = len(sequence)
    valid_cuts = sorted({int(p) for p in cut_positions_1based if 1 <= int(p) < n})

    out: List[Tuple[int, int, str]] = []
    start = 1
    for cut in valid_cuts:
        end = cut
        seg = sequence[start - 1:end]
        if seg:
            out.append((start, end, seg))
        start = cut + 1

    if start <= n:
        out.append((start, n, sequence[start - 1:n]))
    return out


def write_fasta(records: Sequence[Tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + "\n")


def run_cutting_pipeline(
    input_fasta: str,
    esm3di_checkpoint: str,
    ffn_checkpoint: Optional[str],
    transformer_checkpoint: Optional[str],
    output_prefix: str,
    cutoff: float = 0.5,
    classifier_mode: str = "ensemble",
    device: Optional[str] = None,
    batch_size: int = 8,
    revision: Optional[str] = None,
    output_format: str = "both",
) -> Dict[str, Path]:
    """Run full cleavage prediction pipeline and write output artifacts."""
    if output_format not in {"fasta", "csv", "both"}:
        raise ValueError("output_format must be fasta, csv, or both")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_records = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(input_fasta, "fasta")]
    if not input_records:
        raise ValueError(f"No sequences found in {input_fasta}")

    ids = [r[0] for r in input_records]
    seqs = [r[1] for r in input_records]

    esm_model, tokenizer, is_t5 = load_esm3di_embedding_model(
        checkpoint_path=esm3di_checkpoint,
        device=device,
        requested_revision=revision,
    )
    emb_mats = embed_sequences(
        model=esm_model,
        tokenizer=tokenizer,
        sequences=seqs,
        is_t5=is_t5,
        device=device,
        batch_size=batch_size,
    )

    ffn_model = None
    tr_model = None

    if classifier_mode in {"ffn", "ensemble"}:
        if not ffn_checkpoint:
            raise ValueError("FFN checkpoint is required for --classifier ffn/ensemble")
        ffn_model, _ = load_ffn_classifier(ffn_checkpoint, device)

    if classifier_mode in {"transformer", "ensemble"}:
        if not transformer_checkpoint:
            raise ValueError("Transformer checkpoint is required for --classifier transformer/ensemble")
        tr_model, _ = load_transformer_classifier(transformer_checkpoint, device)

    residue_rows: List[dict] = []
    summary_rows: List[dict] = []
    cleaved_records: List[Tuple[str, str]] = []

    for seq_id, seq, emb in zip(ids, seqs, emb_mats):
        probs_ffn = predict_ffn_probs(ffn_model, emb, device) if ffn_model is not None else None
        probs_tr = predict_transformer_probs(tr_model, emb, device) if tr_model is not None else None

        if classifier_mode == "ffn":
            probs_selected = probs_ffn
        elif classifier_mode == "transformer":
            probs_selected = probs_tr
        else:
            probs_selected = 0.5 * (probs_ffn + probs_tr)

        cut_mask = probs_selected >= cutoff
        cut_positions = [int(i + 1) for i, flag in enumerate(cut_mask) if flag]

        segments = cleave_sequence(seq, cut_positions)
        for start, end, segment in segments:
            header = f"{seq_id}_cut[{start}:{end}]"
            cleaved_records.append((header, segment))

        summary_rows.append(
            {
                "identifier": seq_id,
                "length": len(seq),
                "n_cleavage_sites": int(len(cut_positions)),
                "cleavage_positions": ";".join(map(str, cut_positions)) if cut_positions else "",
                "n_products": int(len(segments)),
            }
        )

        for idx in range(len(seq)):
            residue_rows.append(
                {
                    "identifier": seq_id,
                    "residue_index_1based": idx + 1,
                    "aa": seq[idx],
                    "prob_ffn": None if probs_ffn is None else float(probs_ffn[idx]),
                    "prob_transformer": None if probs_tr is None else float(probs_tr[idx]),
                    "prob_selected": float(probs_selected[idx]),
                    "is_cleavage": int(cut_mask[idx]),
                }
            )

    out_prefix = Path(output_prefix)
    if out_prefix.suffix:
        out_prefix = out_prefix.with_suffix("")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    residue_csv = out_prefix.parent / f"{out_prefix.name}_residue_predictions.csv"
    summary_csv = out_prefix.parent / f"{out_prefix.name}_sequence_summary.csv"
    products_fasta = out_prefix.parent / f"{out_prefix.name}_cleaved_products.fasta"

    outputs: Dict[str, Path] = {}
    if output_format in {"csv", "both"}:
        for path, rows in ((residue_csv, residue_rows), (summary_csv, summary_rows)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved residue predictions: {residue_csv}")
        print(f"Saved sequence summary: {summary_csv}")
        outputs.update(residue_csv=residue_csv, summary_csv=summary_csv)
    if output_format in {"fasta", "both"}:
        write_fasta(cleaved_records, products_fasta)
        print(f"Saved cleaved products FASTA: {products_fasta}")
        outputs["products_fasta"] = products_fasta
    return outputs


def add_cutting_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-fasta", required=True, help="Input amino acid FASTA path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--esm3di-checkpoint", help="Trusted local ESM3Di .pt checkpoint used for embeddings")
    source.add_argument("--cutter-model", help="Local exported cutter directory or Hugging Face repository ID")
    parser.add_argument("--cutter-subfolder", default="cutter", help="Cutter subfolder in the Hub repository (default: cutter)")
    parser.add_argument("--cutter-revision", default=None, help="Hub revision for the cutter bundle (not the backbone)")
    parser.add_argument("--ffn-checkpoint", default=None, help="Saved FFN classifier checkpoint")
    parser.add_argument("--transformer-checkpoint", default=None, help="Saved transformer classifier checkpoint")
    parser.add_argument(
        "--classifier",
        choices=["ffn", "transformer", "ensemble"],
        default="ensemble",
        help="Which classifier probabilities to use for cleavage calling",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Cleavage certainty cutoff on selected probability",
    )
    parser.add_argument("--output-prefix", required=True, help="Output prefix for CSV/FASTA artifacts")
    parser.add_argument(
        "--output-format", choices=["fasta", "csv", "both"], default="both",
        help="Write cleaved-product FASTA, cut-information CSVs, or both (default)",
    )
    parser.add_argument("--device", default=None, help="Compute device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--revision", default=None, help="Optional model revision fallback for ESM checkpoint")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict cleavage sites and write cleaved products using ESM3Di embeddings + small classifiers.",
    )
    add_cutting_args(parser)
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> Dict[str, Path]:
    if args.cutter_model:
        from .cutting_bundle import resolve_bundle

        if args.ffn_checkpoint or args.transformer_checkpoint or args.revision:
            raise ValueError("A cutter bundle supplies its own classifiers and backbone revision")
        bundle = str(resolve_bundle(args.cutter_model, args.cutter_subfolder, args.cutter_revision))
        embedding = ffn = transformer = bundle
    else:
        embedding, ffn, transformer = args.esm3di_checkpoint, args.ffn_checkpoint, args.transformer_checkpoint
    return run_cutting_pipeline(
        input_fasta=args.input_fasta,
        esm3di_checkpoint=embedding,
        ffn_checkpoint=ffn,
        transformer_checkpoint=transformer,
        output_prefix=args.output_prefix,
        cutoff=args.cutoff,
        classifier_mode=args.classifier,
        device=args.device,
        batch_size=args.batch_size,
        revision=args.revision,
        output_format=args.output_format,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
