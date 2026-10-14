"""Supervised 3Di token training on top of ESM++ (stage 2 of the pipeline).

This is the streamlined trainer for the recipe that worked best:

* backbone: ESM++ (``Synthyra/ESMplusplus_small`` by default), optionally
  initialised from an MLM fine-tuned checkpoint (``esm3di mlm`` output),
* LoRA adapters (r=16, alpha=32, dropout=0.01) on auto-discovered modules,
* 2-layer CNN token-classification head,
* pLDDT-weighted cross-entropy (bins < 5 ignored) when a pLDDT-bin FASTA is
  given, plain cross-entropy (with ``X`` = masked) otherwise,
* AdamW, lr 1e-4, cosine schedule with 10% warmup, batch 4 x accumulation 8,
  fp16 autocast.

Starting points (``--init-from``):

* nothing / a HuggingFace id -> fresh LoRA + head on the stock backbone,
* an MLM checkpoint directory -> backbone weights copied from it,
* a 3Di ``.pt`` checkpoint -> continue fine-tuning that model (new optimizer).

``--resume`` continues an interrupted run exactly (optimizer, scheduler, epoch).

Every epoch writes ``epoch_N.pt`` (plus ``best.pt`` / ``last.pt``) in the
checkpoint format consumed by :func:`esm3di.checkpoints.load_3di_checkpoint`,
``esm3di predict`` and ``esm3di.fastas2foldseekdb``.

Usage::

    python -m esm3di.train_3di --config configs/3di_esmpp_small_from_mlm.json
    python -m esm3di.train_3di --aa-fasta train_aa.fasta --three-di-fasta train_3di.fasta \\
        --plddt-bins-fasta train_plddt_bins.fasta --init-from runs/mlm/best --out-dir runs/3di
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .checkpoints import (
    DEFAULT_HF_MODEL,
    THREEDI_VOCAB,
    detect_init_source,
    load_state_into_model,
    lora_scaling_from_checkpoint,
    mlm_base_model_name,
    model_kwargs_from_checkpoint,
    read_3di_checkpoint,
    resolve_hf_revision,
)
from .losses import DEFAULT_PLDDT_BIN_WEIGHTS, PLDDTWeightedFocalLoss

PLDDT_BIN_VOCAB = [str(i) for i in range(10)]

# Defaults reproduce the best-performing run (configs/3di_esmpp_small_from_mlm.json).
DEFAULTS = dict(
    config=None,
    aa_fasta=None,
    three_di_fasta=None,
    plddt_bins_fasta=None,
    data_dir=None,
    val_aa_fasta=None,
    val_three_di_fasta=None,
    val_plddt_bins_fasta=None,
    val_split=0.0,
    hf_model=None,
    hf_revision=None,
    init_from=None,
    resume=None,
    out_dir="runs/3di",
    tensorboard_log_dir=None,
    epochs=5,
    batch_size=4,
    gradient_accumulation_steps=8,
    lr=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=0.0,
    max_seq_length=1024,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.01,
    lora_target_modules=None,
    use_cnn_head=True,
    cnn_num_layers=2,
    cnn_kernel_size=3,
    cnn_dropout=0.1,
    mask_label_chars="X",
    plddt_min_bin=5,
    plddt_bin_weights=None,
    plddt_weight_exponent=1.0,
    mixed_precision=True,
    amp_dtype="float16",
    device=None,
    multi_gpu=False,
    num_workers=4,
    log_every=50,
    seed=42,
    max_train_samples=None,
    max_val_samples=None,
    keep_last_n=0,
)

# Old esmretrain.py config keys that map onto new names.
LEGACY_KEY_ALIASES = {
    "mlm_checkpoint": "init_from",
    "resume_from_checkpoint": "resume",
    "output_dir": "out_dir",
    "learning_rate": "lr",
    "grad_accum_steps": "gradient_accumulation_steps",
    "model_revision": "hf_revision",
}

# Old keys that are accepted (they match the recipe) but have no effect here.
IGNORED_LEGACY_KEYS = {
    "scheduler_type", "label_smoothing", "optimizer", "samples_per_epoch",
    "use_focal_loss", "focal_gamma", "focal_alpha", "push_to_hub", "hub_repo_id",
    "hub_private", "hub_revision", "hub_token_env_var", "cpu", "warmup_steps",
    "plateau_patience", "plateau_factor", "plateau_threshold", "plateau_min_lr",
    "gamma_scheduler", "comment", "description",
    # Sub-parameters of experimental features (inert unless the feature is enabled).
    "aux_loss_weight", "aux_track_num_bins", "cyclical_factor", "gamma_hc", "gamma_increase_factor",
    "gamma_max", "gamma_neg", "gamma_patience", "gamma_pos", "gamma_threshold",
    "iterative_backbone_k_layers", "iterative_head_halt_threshold", "iterative_head_lambda_p",
    "iterative_head_max_iterations", "iterative_head_prior_p", "lion_beta1", "lion_beta2",
    "plddt_loss_weight", "plddt_prediction_mode", "plddt_regression_loss", "transformer_head_dim",
    "transformer_head_dropout", "transformer_head_layers", "transformer_head_num_heads",
    "use_gru_gate", "use_hidden_state_feedback", "use_positional_encoding",
}

# Old keys enabling experimental features; refuse unless they are falsy.
EXPERIMENTAL_KEYS = {
    "use_transformer_head", "use_iterative_transformer_head", "use_iterative_backbone_head",
    "use_plddt_prediction_head", "aux_fastas", "val_aux_fastas", "use_cyclical_focal",
    "half_precision", "gradient_checkpointing",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read a FASTA file into ``[(id, sequence)]``; ``id`` is the first header token."""
    records: List[Tuple[str, str]] = []
    header, chunks = None, []
    opener = open
    if path.endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def load_paired_records(
    aa_fasta: str,
    three_di_fasta: str,
    plddt_bins_fasta: Optional[str] = None,
) -> List[Tuple[str, str, str, Optional[str]]]:
    """Pair AA, 3Di (and pLDDT-bin) FASTAs by record id.

    Sequences are upper-cased. Records with mismatched lengths are dropped with
    a warning; ids missing from one of the files raise an error.
    """
    aa = read_fasta(aa_fasta)
    tdi = {h: s.upper() for h, s in read_fasta(three_di_fasta)}
    plddt = {h: s for h, s in read_fasta(plddt_bins_fasta)} if plddt_bins_fasta else None

    missing = [h for h, _ in aa if h not in tdi or (plddt is not None and h not in plddt)]
    if missing:
        raise ValueError(
            f"{len(missing)} AA records have no matching 3Di/pLDDT record (e.g. {missing[:3]}). "
            "AA, 3Di and pLDDT FASTAs must share record ids."
        )

    records, dropped = [], 0
    for h, seq in aa:
        seq = seq.upper()
        lab = tdi[h]
        pl = plddt[h] if plddt is not None else None
        if len(seq) != len(lab) or (pl is not None and len(pl) != len(seq)):
            dropped += 1
            continue
        records.append((h, seq, lab, pl))
    if dropped:
        print(f"  warning: dropped {dropped} records with AA/3Di/pLDDT length mismatch")
    if not records:
        raise ValueError(f"No usable records in {aa_fasta}")
    return records


class Paired3DiDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate(tokenizer, char2idx: Dict[str, int], mask_chars: str, max_seq_length: Optional[int]):
    """Tokenise AA sequences and align per-residue labels/pLDDT bins to tokens.

    Special tokens (CLS/EOS/PAD) get label ``-100``. With pLDDT bins, every residue
    keeps its 3Di label and the loss weights it by confidence; without pLDDT,
    residues whose 3Di label is in ``mask_chars`` are ignored.
    """
    mask_set = set(mask_chars or "")

    def collate(batch):
        ids, aa_seqs, lab_seqs, plddt_seqs = zip(*batch)
        enc = tokenizer(
            list(aa_seqs),
            return_tensors="pt",
            padding=True,
            truncation=max_seq_length is not None,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special = enc["special_tokens_mask"].bool()
        bsz, length = enc["input_ids"].shape
        labels = torch.full((bsz, length), -100, dtype=torch.long)
        has_plddt = plddt_seqs[0] is not None
        bins = torch.zeros((bsz, length), dtype=torch.long) if has_plddt else None

        for i in range(bsz):
            positions = (~special[i]).nonzero(as_tuple=True)[0]
            n = min(len(positions), len(lab_seqs[i]))
            positions = positions[:n]
            lab = lab_seqs[i][:n]
            idx = []
            for ch in lab:
                if ch in char2idx:
                    idx.append(char2idx[ch])
                elif ch in mask_set:
                    idx.append(-100)
                else:
                    raise ValueError(
                        f"Record {ids[i]!r}: 3Di character {ch!r} is not in the 3Di alphabet "
                        f"({''.join(char2idx)}) or mask chars ({mask_chars!r})."
                    )
            labels[i, positions] = torch.tensor(idx, dtype=torch.long)
            if has_plddt:
                bins[i, positions] = torch.tensor(
                    [min(int(c), 9) for c in plddt_seqs[i][:n]], dtype=torch.long
                )

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
        if has_plddt:
            out["plddt_bins"] = bins
        return out

    return collate


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def add_train_3di_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", help="JSON config; explicit CLI flags override its values.")

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", help="Directory written by `esm3di prepare` (fills the train/val/pLDDT FASTAs below).")
    g.add_argument("--aa-fasta", help="Training amino-acid FASTA.")
    g.add_argument("--three-di-fasta", help="Training 3Di FASTA (same ids/lengths as --aa-fasta).")
    g.add_argument("--plddt-bins-fasta", help="Optional per-residue pLDDT bins (0-9) FASTA; enables pLDDT-weighted loss.")
    g.add_argument("--val-aa-fasta")
    g.add_argument("--val-three-di-fasta")
    g.add_argument("--val-plddt-bins-fasta")
    g.add_argument("--val-split", type=float, help="Hold out this fraction of training data for validation when no val FASTAs are given (default 0).")
    g.add_argument("--max-seq-length", type=int, help="Truncate sequences to this many tokens incl. special tokens (default 1024).")
    g.add_argument("--mask-label-chars", help="3Di characters treated as 'no label' when no pLDDT bins are given (default 'X').")
    g.add_argument("--max-train-samples", type=int, help="Use only the first N training records (smoke tests).")
    g.add_argument("--max-val-samples", type=int, help="Use only the first N validation records.")

    g = p.add_argument_group("model / initialisation")
    g.add_argument("--hf-model", "--model-name", dest="hf_model", help=f"Base HF model (default {DEFAULT_HF_MODEL}, or the MLM checkpoint's base).")
    g.add_argument("--hf-revision", help="HF revision; defaults to the pinned known-good revision.")
    g.add_argument("--init-from", "--mlm-checkpoint", dest="init_from", help="HF id, MLM checkpoint dir (esm3di mlm output) or 3Di .pt checkpoint to start from.")
    g.add_argument("--resume", "--resume-from-checkpoint", dest="resume", help="3Di .pt to resume (optimizer/scheduler/epoch), or 'auto' for the latest in --out-dir.")
    g.add_argument("--lora-r", type=int)
    g.add_argument("--lora-alpha", type=float)
    g.add_argument("--lora-dropout", type=float)
    g.add_argument("--lora-target-modules", help="Comma-separated module names; default auto-discovery.")
    g.add_argument("--use-cnn-head", type=_str2bool, help="Use the CNN head (default true).")
    g.add_argument("--cnn-num-layers", type=int)
    g.add_argument("--cnn-kernel-size", type=int)
    g.add_argument("--cnn-dropout", type=float)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--gradient-accumulation-steps", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--warmup-ratio", type=float)
    g.add_argument("--max-grad-norm", type=float, help="Gradient clipping norm (0 = off, default).")
    g.add_argument("--plddt-min-bin", type=int, help="Ignore residues with pLDDT bin below this (default 5 = pLDDT<50).")
    g.add_argument("--plddt-weight-exponent", type=float)
    g.add_argument("--mixed-precision", type=_str2bool, help="Autocast on CUDA (default true).")
    g.add_argument("--amp-dtype", choices=["float16", "bfloat16"])
    g.add_argument("--seed", type=int)

    g = p.add_argument_group("runtime / output")
    g.add_argument("--out-dir", "--output-dir", dest="out_dir", help="Checkpoint directory (default runs/3di).")
    g.add_argument("--tensorboard-log-dir")
    g.add_argument("--device", help="e.g. cuda, cuda:1, cpu (default: cuda if available).")
    g.add_argument("--multi-gpu", type=_str2bool, help="Wrap the model in DataParallel over visible GPUs.")
    g.add_argument("--num-workers", type=int)
    g.add_argument("--log-every", type=int)
    g.add_argument("--keep-last-n", type=int, help="Keep only the N most recent epoch_*.pt (0 = keep all).")
    return p


def normalise_config(raw: dict, source: str = "config") -> dict:
    cfg = {}
    ignored = []
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        key = key.replace("-", "_")
        key = LEGACY_KEY_ALIASES.get(key, key)
        if key in EXPERIMENTAL_KEYS:
            if value:
                raise ValueError(
                    f"{source}: '{key}' enables an experimental feature that the streamlined trainer "
                    "does not support. Use `python -m esm3di.experimental.esmretrain_full` instead."
                )
            continue
        if key in IGNORED_LEGACY_KEYS:
            ignored.append(key)
            continue
        if key not in DEFAULTS:
            raise ValueError(f"{source}: unknown option '{key}'")
        cfg[key] = value
    if ignored:
        print(f"Note: {source}: ignoring legacy options with no effect here: {', '.join(sorted(ignored))}")
    return cfg


def resolve_config(ns: argparse.Namespace) -> dict:
    """Merge defaults < config file < explicit CLI flags."""
    cfg = dict(DEFAULTS)
    if getattr(ns, "config", None):
        with open(ns.config) as fh:
            cfg.update(normalise_config(json.load(fh), ns.config))
        cfg["config"] = ns.config
    explicit = {k: v for k, v in vars(ns).items() if v is not None and k in DEFAULTS and k != "config"}
    if explicit.get("data_dir"):
        cfg["data_dir"] = explicit["data_dir"]
    if cfg.get("data_dir"):
        from .prepare_data import data_files_from_dir

        for key, value in data_files_from_dir(cfg["data_dir"]).items():
            if key not in explicit:
                cfg[key] = value
    cfg.update(explicit)
    if isinstance(cfg.get("lora_target_modules"), str):
        cfg["lora_target_modules"] = [m.strip() for m in cfg["lora_target_modules"].split(",") if m.strip()]
    if not cfg.get("aa_fasta") or not cfg.get("three_di_fasta"):
        raise ValueError("--aa-fasta and --three-di-fasta (or --data-dir) are required, directly or via --config.")
    if bool(cfg.get("val_aa_fasta")) != bool(cfg.get("val_three_di_fasta")):
        raise ValueError("Provide both --val-aa-fasta and --val-three-di-fasta, or neither.")
    if cfg.get("val_plddt_bins_fasta") and not cfg.get("plddt_bins_fasta"):
        raise ValueError("--val-plddt-bins-fasta requires --plddt-bins-fasta.")
    if cfg.get("plddt_bins_fasta") and cfg.get("val_aa_fasta") and not cfg.get("val_plddt_bins_fasta"):
        raise ValueError("--plddt-bins-fasta is set; also provide --val-plddt-bins-fasta.")
    return cfg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(out_dir: str) -> Optional[str]:
    found = []
    for path in glob.glob(os.path.join(out_dir, "epoch_*.pt")):
        m = re.search(r"epoch_(\d+)\.pt$", path)
        if m:
            found.append((int(m.group(1)), path))
    return max(found)[1] if found else None


def build_model(cfg: dict, resume_ckpt: Optional[dict]):
    """Return ``(esm3di_model, label_vocab, provenance)`` according to cfg."""
    from .ESM3di_model import ESM3DiModel

    if resume_ckpt is not None:
        kwargs = model_kwargs_from_checkpoint(resume_ckpt)
        esm = ESM3DiModel(**kwargs)
        load_state_into_model(esm.model, resume_ckpt["model_state_dict"])
        prov = dict(resume_ckpt["args"].get("provenance") or {})
        return esm, list(resume_ckpt["label_vocab"]), prov

    source = detect_init_source(cfg.get("init_from"))
    if source == "3di":
        ckpt = read_3di_checkpoint(cfg["init_from"])
        if list(ckpt["label_vocab"]) != THREEDI_VOCAB:
            print(f"  note: checkpoint label vocab {''.join(ckpt['label_vocab'])} differs from canonical order")
        kwargs = model_kwargs_from_checkpoint(ckpt)
        if cfg.get("hf_revision"):
            kwargs["revision"] = cfg["hf_revision"]
        esm = ESM3DiModel(**kwargs)
        load_state_into_model(esm.model, ckpt["model_state_dict"], lora_scaling=lora_scaling_from_checkpoint(ckpt))
        print(f"✓ Initialised from 3Di checkpoint {cfg['init_from']} (epoch {ckpt.get('epoch')})")
        prov = dict(ckpt["args"].get("provenance") or {})
        prov.setdefault("history", []).append(os.path.abspath(cfg["init_from"]))
        return esm, list(ckpt["label_vocab"]), prov

    backbone_init = None
    hf_model = cfg.get("hf_model")
    if source == "mlm":
        backbone_init = cfg["init_from"]
        mlm_base = mlm_base_model_name(backbone_init)
        if hf_model and mlm_base and hf_model != mlm_base:
            print(
                f"  warning: hf_model {hf_model} differs from the MLM checkpoint's base {mlm_base}; "
                f"using {mlm_base} (the backbone architecture must match the MLM weights)"
            )
        hf_model = mlm_base or hf_model
    elif cfg.get("init_from"):
        hf_model = hf_model or cfg["init_from"]
    hf_model = hf_model or DEFAULT_HF_MODEL

    esm = ESM3DiModel(
        hf_model_name=hf_model,
        num_labels=len(THREEDI_VOCAB),
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg.get("lora_target_modules"),
        use_cnn_head=cfg["use_cnn_head"],
        cnn_num_layers=cfg["cnn_num_layers"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        cnn_dropout=cfg["cnn_dropout"],
        revision=cfg.get("hf_revision"),
        backbone_init=backbone_init,
    )
    prov = {"base_model": hf_model, "hf_revision": esm.revision}
    if backbone_init:
        prov["mlm_checkpoint"] = os.path.abspath(backbone_init)
        prov["mlm_transfer"] = esm.backbone_transfer_report.summary()
    return esm, list(THREEDI_VOCAB), prov


def checkpoint_args(cfg: dict, esm) -> dict:
    """Args dict stored in checkpoints; contains everything needed to rebuild the model."""
    args = {k: v for k, v in cfg.items()}
    args.update(
        hf_model=esm.hf_model_name,
        hf_revision=esm.revision,
        lora_r=esm.lora_r,
        lora_alpha=esm.lora_alpha,
        lora_dropout=esm.lora_dropout,
        use_cnn_head=esm.use_cnn_head,
        cnn_num_layers=esm.cnn_num_layers,
        cnn_kernel_size=esm.cnn_kernel_size,
        cnn_dropout=esm.cnn_dropout,
        use_transformer_head=esm.use_transformer_head,
        use_iterative_transformer_head=esm.use_iterative_transformer_head,
        use_plddt_prediction_head=esm.use_plddt_prediction_head,
        lora_target_modules=list(esm.target_modules),
    )
    # Never rebuild from MLM weights at load time: the .pt holds the full model.
    args.pop("init_from", None)
    args.pop("resume", None)
    args["mlm_checkpoint"] = None
    return args


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class LossComputer:
    def __init__(self, cfg: dict, use_plddt: bool, device):
        self.use_plddt = use_plddt
        if use_plddt:
            self.fn = PLDDTWeightedFocalLoss(
                gamma=0.0,
                bin_weights=cfg.get("plddt_bin_weights") or DEFAULT_PLDDT_BIN_WEIGHTS,
                min_bin=cfg["plddt_min_bin"],
                weight_exponent=cfg["plddt_weight_exponent"],
            ).to(device)
        else:
            self.fn = nn.CrossEntropyLoss(ignore_index=-100)

    def __call__(self, logits, labels, plddt_bins=None):
        c = logits.shape[-1]
        logits = logits.float().reshape(-1, c)
        labels = labels.reshape(-1)
        if self.use_plddt:
            return self.fn(logits, labels, plddt_bins.reshape(-1))
        return self.fn(logits, labels)


@torch.no_grad()
def evaluate(model, loader, loss_computer, device, amp_ctx) -> Dict[str, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct = total = 0
    hc_correct = hc_total = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp_ctx():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits
        loss = loss_computer(logits, batch["labels"], batch.get("plddt_bins"))
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        preds = logits.argmax(-1)
        valid = batch["labels"] != -100
        hit = (preds == batch["labels"]) & valid
        correct += hit.sum().item()
        total += valid.sum().item()
        if "plddt_bins" in batch:
            hc = valid & (batch["plddt_bins"] >= 7)
            hc_correct += (hit & hc).sum().item()
            hc_total += hc.sum().item()
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
    }
    if hc_total:
        metrics["accuracy_plddt70"] = hc_correct / hc_total
    model.train()
    return metrics


def train(cfg: dict) -> str:
    """Run 3Di training; returns the path of the best checkpoint."""
    _set_seed(cfg["seed"])
    if not cfg.get("resume") and any(
        glob.glob(os.path.join(cfg["out_dir"], pat)) for pat in ("epoch_*.pt", "best.pt", "last.pt")
    ):
        raise FileExistsError(
            f"{cfg['out_dir']} already contains 3Di checkpoints; use --resume auto or choose a new --out-dir "
            "(existing checkpoints are never overwritten by a fresh run)."
        )
    os.makedirs(cfg["out_dir"], exist_ok=True)

    if cfg.get("device"):
        device = torch.device(cfg["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_path = cfg.get("resume")
    if resume_path == "auto":
        resume_path = _find_latest_checkpoint(cfg["out_dir"])
        print(f"Auto-resume: {resume_path or 'no checkpoint found, starting fresh'}")
    resume_ckpt = read_3di_checkpoint(resume_path) if resume_path else None

    esm, label_vocab, provenance = build_model(cfg, resume_ckpt)
    tokenizer = esm.tokenizer
    char2idx = {c: i for i, c in enumerate(label_vocab)}

    # ---- data
    use_plddt = bool(cfg.get("plddt_bins_fasta"))
    print("Loading training data...")
    train_records = load_paired_records(cfg["aa_fasta"], cfg["three_di_fasta"], cfg.get("plddt_bins_fasta"))
    if cfg.get("val_aa_fasta"):
        val_records = load_paired_records(
            cfg["val_aa_fasta"], cfg["val_three_di_fasta"], cfg.get("val_plddt_bins_fasta")
        )
    elif cfg.get("val_split"):
        rng = random.Random(cfg["seed"])
        shuffled = train_records[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * cfg["val_split"]))
        val_records, train_records = shuffled[:n_val], shuffled[n_val:]
    else:
        val_records = []
    if cfg.get("max_train_samples"):
        train_records = train_records[: cfg["max_train_samples"]]
    if cfg.get("max_val_samples"):
        val_records = val_records[: cfg["max_val_samples"]]
    print(f"  train: {len(train_records)} sequences, val: {len(val_records)} sequences, "
          f"loss: {'pLDDT-weighted CE' if use_plddt else 'CE'}")

    collate = make_collate(tokenizer, char2idx, cfg["mask_label_chars"], cfg.get("max_seq_length"))
    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        collate_fn=collate,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(Paired3DiDataset(train_records), shuffle=True, generator=g, **loader_kwargs)
    val_loader = DataLoader(Paired3DiDataset(val_records), shuffle=False, **loader_kwargs) if val_records else None

    # ---- model / optimisation
    model = esm.model.to(device)
    if cfg.get("multi_gpu") and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    accum = max(1, cfg["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_ratio"] * total_steps), num_training_steps=total_steps
    )

    use_amp = cfg["mixed_precision"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def amp_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    loss_computer = LossComputer(cfg, use_plddt, device)

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if resume_ckpt is not None:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if resume_ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            if resume_ckpt.get("scaler_state_dict") and scaler.is_enabled():
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        except Exception as e:  # e.g. param groups changed
            print(f"  warning: could not restore optimizer/scheduler state ({e}); continuing with fresh state")
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_val = float(resume_ckpt.get("best_val_loss", resume_ckpt.get("loss", float("inf"))) or float("inf"))
        print(f"Resumed from {resume_path} at epoch {start_epoch - 1}, step {global_step}")
        if start_epoch > cfg["epochs"]:
            print(f"Checkpoint already reached epoch {start_epoch - 1} >= --epochs {cfg['epochs']}; nothing to do.")

    writer = None
    if cfg.get("tensorboard_log_dir"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(cfg["tensorboard_log_dir"])
        except ImportError:
            print("  tensorboard not installed; skipping TensorBoard logging")

    args_dict = checkpoint_args(cfg, esm)
    args_dict["provenance"] = provenance
    with open(os.path.join(cfg["out_dir"], "train_config.json"), "w") as fh:
        json.dump(args_dict, fh, indent=2, default=str)

    best_path = os.path.join(cfg["out_dir"], "best.pt")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        running, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with amp_ctx():
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_computer(out.logits, batch["labels"], batch.get("plddt_bins"))
            if not torch.isfinite(loss):
                # e.g. a batch where every residue has pLDDT < 50 -> no supervised positions
                continue
            scaler.scale(loss / accum).backward()
            running += loss.item()
            n_seen += 1

            if step % accum == 0 or step == len(train_loader):
                if cfg.get("max_grad_norm"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg["log_every"] == 0:
                    avg = running / max(n_seen, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} loss {avg:.4f} lr {lr:.2e}", flush=True)
                    if writer:
                        writer.add_scalar("train/loss", avg, global_step)
                        writer.add_scalar("train/lr", lr, global_step)

        train_loss = running / max(n_seen, 1)
        metrics = {"train_loss": train_loss}
        if val_loader is not None:
            val = evaluate(model, val_loader, loss_computer, device, amp_ctx)
            metrics.update({f"val_{k}": v for k, v in val.items()})
            if writer:
                for k, v in val.items():
                    writer.add_scalar(f"val/{k}", v, epoch)
        if writer:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
        score = metrics.get("val_loss", train_loss)
        is_best = score < best_val
        best_val = min(best_val, score)
        print(f"== epoch {epoch} done in {(time.time() - t0) / 60:.1f} min: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
              + (" (best)" if is_best else ""), flush=True)

        ckpt = {
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "plateau_scheduler_state_dict": None,
            "gamma_scheduler_state_dict": None,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "label_vocab": label_vocab,
            "plddt_label_vocab": PLDDT_BIN_VOCAB if use_plddt else None,
            "mask_label_chars": cfg["mask_label_chars"],
            "lora_target_modules": list(esm.target_modules),
            "args": args_dict,
            "epoch": epoch,
            "loss": score,
            "best_val_loss": best_val,
            "metrics": metrics,
            "global_step": global_step,
            "esm3di_format_version": 2,
        }
        epoch_path = os.path.join(cfg["out_dir"], f"epoch_{epoch}.pt")
        _atomic_save(ckpt, epoch_path)
        _atomic_save(ckpt, os.path.join(cfg["out_dir"], "last.pt"))
        if is_best:
            _atomic_save(ckpt, best_path)
        with open(os.path.join(cfg["out_dir"], "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps({"epoch": epoch, "global_step": global_step, **metrics}) + "\n")
        print(f"  saved {epoch_path}")

        if cfg.get("keep_last_n"):
            olds = sorted(
                glob.glob(os.path.join(cfg["out_dir"], "epoch_*.pt")),
                key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p).group(1)),
            )
            for old in olds[: -cfg["keep_last_n"]]:
                os.remove(old)

    if writer:
        writer.close()
    if not os.path.exists(best_path):
        last = _find_latest_checkpoint(cfg["out_dir"])
        if last:
            best_path = last
    print(f"Training finished. Best checkpoint: {best_path}")
    return best_path


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Train an ESM++ 3Di token tagger (LoRA + CNN head).",
    )
    add_train_3di_args(parser)
    ns = parser.parse_args(argv)
    cfg = resolve_config(ns)
    train(cfg)


if __name__ == "__main__":
    main()
