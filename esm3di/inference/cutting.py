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
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from transformers import AutoTokenizer, T5Tokenizer

from .. import ESM3DiModel


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


def _get_tokenizer(esm3di_model: ESM3DiModel, hf_model_name: str):
    if hasattr(esm3di_model, "tokenizer") and esm3di_model.tokenizer is not None:
        return esm3di_model.tokenizer
    if esm3di_model.is_t5:
        try:
            return T5Tokenizer.from_pretrained(hf_model_name, legacy=True, trust_remote_code=True)
        except Exception:
            return AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(hf_model_name, do_lower_case=False, trust_remote_code=True)


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
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args_dict = ckpt.get("args", {})

    hf_model_name = args_dict.get("hf_model_name", args_dict.get("hf_model", "facebook/esm2_t33_650M_UR50D"))
    resolved_revision = args_dict.get("revision", requested_revision)
    label_vocab = ckpt.get("label_vocab") or []
    num_labels = len(label_vocab) or int(args_dict.get("num_labels", 20))

    use_cnn_head = bool(args_dict.get("use_cnn_head", False))
    use_transformer_head = bool(args_dict.get("use_transformer_head", False))
    use_plddt_prediction_head = bool(args_dict.get("use_plddt_prediction_head", False))
    plddt_prediction_mode = args_dict.get("plddt_prediction_mode", "classification")

    target_modules = ckpt.get("lora_target_modules", args_dict.get("target_modules"))

    build_kwargs = dict(
        hf_model_name=hf_model_name,
        revision=resolved_revision,
        num_labels=num_labels,
        lora_r=args_dict.get("lora_r", 8),
        lora_alpha=args_dict.get("lora_alpha", 16),
        lora_dropout=args_dict.get("lora_dropout", 0.05),
        target_modules=target_modules,
        use_cnn_head=use_cnn_head,
        cnn_num_layers=args_dict.get("cnn_num_layers", 2),
        cnn_kernel_size=args_dict.get("cnn_kernel_size", 3),
        cnn_dropout=args_dict.get("cnn_dropout", 0.1),
        use_transformer_head=use_transformer_head,
        use_plddt_prediction_head=use_plddt_prediction_head,
        plddt_num_bins=args_dict.get("plddt_num_bins", len(ckpt.get("plddt_label_vocab") or []) or 10),
        plddt_prediction_mode=plddt_prediction_mode,
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
        aux_track_num_bins=ckpt.get("aux_track_num_bins", args_dict.get("aux_track_num_bins", None)),
    )

    try:
        esm3di = ESM3DiModel(**build_kwargs)
    except ValueError as exc:
        if "Target modules" not in str(exc):
            raise
        build_kwargs["target_modules"] = None
        esm3di = ESM3DiModel(**build_kwargs)

    model = esm3di.get_model()
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = _get_tokenizer(esm3di, hf_model_name)
    return model, tokenizer, esm3di.is_t5


def embed_sequences(
    model: nn.Module,
    tokenizer,
    sequences: Sequence[str],
    is_t5: bool,
    device: str,
    batch_size: int = 8,
) -> List[np.ndarray]:
    """Create per-sequence embedding matrices with shape (n_aa, d_embedding)."""
    mats: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_raw = sequences[i:i + batch_size]
            batch_for_tok = [" ".join(list(s)) for s in batch_raw] if is_t5 else batch_raw

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
    ckpt = torch.load(path, map_location="cpu")
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
    ckpt = torch.load(path, map_location="cpu")
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
) -> Dict[str, Path]:
    """Run full cleavage prediction pipeline and write output artifacts."""
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

    residue_df = pd.DataFrame(residue_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_prefix = Path(output_prefix)
    if out_prefix.suffix:
        out_prefix = out_prefix.with_suffix("")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    residue_csv = out_prefix.parent / f"{out_prefix.name}_residue_predictions.csv"
    summary_csv = out_prefix.parent / f"{out_prefix.name}_sequence_summary.csv"
    products_fasta = out_prefix.parent / f"{out_prefix.name}_cleaved_products.fasta"

    residue_df.to_csv(residue_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    write_fasta(cleaved_records, products_fasta)

    print(f"Saved residue predictions: {residue_csv}")
    print(f"Saved sequence summary: {summary_csv}")
    print(f"Saved cleaved products FASTA: {products_fasta}")

    return {
        "residue_csv": residue_csv,
        "summary_csv": summary_csv,
        "products_fasta": products_fasta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict cleavage sites and write cleaved products using ESM3Di embeddings + small classifiers.",
    )
    parser.add_argument("--input-fasta", required=True, help="Input amino acid FASTA path")
    parser.add_argument("--esm3di-checkpoint", required=True, help="ESM3Di checkpoint used for embedding extraction")
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
    parser.add_argument("--device", default=None, help="Compute device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--revision", default=None, help="Optional model revision fallback for ESM checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cutting_pipeline(
        input_fasta=args.input_fasta,
        esm3di_checkpoint=args.esm3di_checkpoint,
        ffn_checkpoint=args.ffn_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        output_prefix=args.output_prefix,
        cutoff=args.cutoff,
        classifier_mode=args.classifier,
        device=args.device,
        batch_size=args.batch_size,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
