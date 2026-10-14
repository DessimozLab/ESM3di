"""Unsupervised masked-language-model fine-tuning of ESM++ (stage 1 of the pipeline).

Adapts an ESM++ backbone to a protein family/domain (e.g. viral proteins) with
the standard BERT masking objective, before supervised 3Di training
(:mod:`esm3di.train_3di`) starts from the adapted weights.

Output layout (``--output-dir``)::

    epoch_N/   HF model dir (config.json, model.safetensors, tokenizer files,
               esm3di_mlm.json metadata) + training_state.pt for --resume
    best/      copy of the epoch with the lowest validation loss
    final/     weights after the last epoch
    metrics.jsonl

Any of these directories can be passed to ``esm3di train --init-from``. When
LoRA is used, merged full weights are saved (plus the adapter in ``adapter/``),
so the 3Di stage always sees a plain backbone.

Usage::

    python -m esm3di.train_mlm --config configs/mlm_esmpp_small.json
    python -m esm3di.train_mlm --input-fasta seqs.fasta --model-name Synthyra/ESMplusplus_small \\
        --output-dir runs/mlm --epochs 3
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..checkpoints import (
    DEFAULT_HF_MODEL,
    THREEDI_VOCAB,
    get_tokenizer,
    load_mlm_model,
    resolve_hf_revision,
)

AA_TOKENS = THREEDI_VOCAB  # the 20 canonical amino acids

DEFAULTS = dict(
    config=None,
    input_fasta=None,
    input_txt=None,
    val_fasta=None,
    test_fasta=None,
    val_split=0.05,
    min_length=1,
    max_length=1024,
    max_train_samples=None,
    max_val_samples=None,
    model_name=DEFAULT_HF_MODEL,
    hf_revision=None,
    output_dir="runs/mlm",
    resume=None,
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0,
    warmup_ratio=0.0,
    grad_accum_steps=1,
    max_grad_norm=1.0,
    mlm_probability=0.15,
    precision="bf16",
    device=None,
    seed=1337,
    num_workers=2,
    log_every=100,
    use_lora=False,
    lora_r=8,
    lora_alpha=16.0,
    lora_dropout=0.05,
    lora_targets=None,
    save_every_epoch=True,
    keep_last_n=0,
    push_to_hub=False,
    hub_repo_id=None,
    hub_private=False,
    hub_revision="main",
    hub_token_env_var="HF_TOKEN",
)

LEGACY_KEY_ALIASES = {
    "dtype": "precision",
    "lr": "learning_rate",
    "gradient_accumulation_steps": "grad_accum_steps",
    "out_dir": "output_dir",
    "hf_model": "model_name",
    "resume_from_checkpoint": "resume",
}
IGNORED_LEGACY_KEYS = {"merge_lora_on_save", "_comment", "comment", "description"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    return [v for v in str(value).split(",") if v]


def load_sequences(fastas=None, txts=None, min_length: int = 1) -> List[str]:
    from .train_3di import read_fasta

    seqs: List[str] = []
    for path in _as_list(fastas):
        seqs.extend(s.upper() for _, s in read_fasta(path))
    for path in _as_list(txts):
        with open(path) as fh:
            seqs.extend(line.strip().upper() for line in fh if line.strip())
    return [s for s in seqs if len(s) >= min_length]


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_mlm_collate(tokenizer, mlm_probability: float, max_length: Optional[int], generator=None):
    """BERT-style masking: of the selected residues 80% -> <mask>, 10% random AA, 10% unchanged.

    Only real residues are candidates (never CLS/EOS/PAD). At least one residue per
    sequence is selected so every sequence contributes to the loss.
    """
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token")
    aa_ids = torch.tensor(tokenizer.convert_tokens_to_ids(AA_TOKENS), dtype=torch.long)

    def collate(batch):
        enc = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"].clone()
        candidates = (enc["special_tokens_mask"] == 0) & (enc["attention_mask"] == 1)
        probs = torch.full(input_ids.shape, mlm_probability)
        selected = (torch.rand(input_ids.shape, generator=generator) < probs) & candidates
        # guarantee >=1 masked residue per sequence
        for i in range(input_ids.shape[0]):
            if not selected[i].any() and candidates[i].any():
                pos = candidates[i].nonzero(as_tuple=True)[0]
                selected[i, pos[torch.randint(len(pos), (1,), generator=generator)]] = True
        labels = torch.where(selected, input_ids, torch.full_like(input_ids, -100))
        roll = torch.rand(input_ids.shape, generator=generator)
        to_mask = selected & (roll < 0.8)
        to_random = selected & (roll >= 0.8) & (roll < 0.9)
        input_ids[to_mask] = mask_id
        input_ids[to_random] = aa_ids[torch.randint(len(aa_ids), (int(to_random.sum()),), generator=generator)]
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}

    return collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def add_train_mlm_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", help="JSON config; explicit CLI flags override its values.")
    g = p.add_argument_group("data")
    g.add_argument("--input-fasta", nargs="+", help="Training FASTA file(s).")
    g.add_argument("--input-txt", nargs="+", help="Text file(s) with one sequence per line.")
    g.add_argument("--val-fasta", nargs="+", help="Validation FASTA(s). If omitted, --val-split of training data is held out.")
    g.add_argument("--test-fasta", nargs="+", help="Optional test FASTA(s), evaluated once at the end.")
    g.add_argument("--val-split", type=float, help="Validation fraction when no --val-fasta (default 0.05).")
    g.add_argument("--min-length", type=int, help="Drop sequences shorter than this (default 1).")
    g.add_argument("--max-length", type=int, help="Truncate to this many tokens incl. special tokens (default 1024).")
    g.add_argument("--max-train-samples", type=int, help="Use only the first N training sequences (smoke tests).")
    g.add_argument("--max-val-samples", type=int)
    g.add_argument("--mlm-probability", type=float, help="Fraction of residues masked (default 0.15).")

    g = p.add_argument_group("model")
    g.add_argument("--model-name", "--hf-model", "--init-from", dest="model_name", help=f"HF id or a previous MLM checkpoint dir to continue from (default {DEFAULT_HF_MODEL}).")
    g.add_argument("--hf-revision", help="HF revision; defaults to the pinned known-good revision.")
    g.add_argument("--use-lora", type=_str2bool, help="Train LoRA adapters instead of all weights (default false).")
    g.add_argument("--lora-r", type=int)
    g.add_argument("--lora-alpha", type=float)
    g.add_argument("--lora-dropout", type=float)
    g.add_argument("--lora-targets", help="Comma-separated LoRA target modules (default auto-discovery).")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, help="Default 2e-5.")
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--warmup-steps", type=int)
    g.add_argument("--warmup-ratio", type=float, help="Used when --warmup-steps is 0.")
    g.add_argument("--grad-accum-steps", type=int)
    g.add_argument("--max-grad-norm", type=float, help="Gradient clipping (default 1.0, 0 = off).")
    g.add_argument("--precision", "--dtype", dest="precision", choices=["fp32", "bf16", "fp16"], help="Autocast precision; weights stay fp32 (default bf16).")
    g.add_argument("--seed", type=int)

    g = p.add_argument_group("runtime / output")
    g.add_argument("--output-dir", "--out-dir", dest="output_dir", help="Default runs/mlm.")
    g.add_argument("--resume", help="Resume from an epoch_N dir, or 'auto' for the latest one in --output-dir.")
    g.add_argument("--device")
    g.add_argument("--num-workers", type=int)
    g.add_argument("--log-every", type=int)
    g.add_argument("--keep-last-n", type=int, help="Keep only the N most recent epoch_* dirs (0 = keep all).")
    g.add_argument("--push-to-hub", type=_str2bool)
    g.add_argument("--hub-repo-id")
    g.add_argument("--hub-private", type=_str2bool)
    g.add_argument("--hub-revision")
    g.add_argument("--hub-token-env-var")
    return p


def normalise_config(raw: dict, source: str = "config") -> dict:
    cfg = {}
    for key, value in raw.items():
        key = LEGACY_KEY_ALIASES.get(key.replace("-", "_"), key.replace("-", "_"))
        if key in IGNORED_LEGACY_KEYS:
            continue
        if key not in DEFAULTS:
            raise ValueError(f"{source}: unknown option '{key}'")
        cfg[key] = value
    return cfg


def resolve_config(ns: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if getattr(ns, "config", None):
        with open(ns.config) as fh:
            cfg.update(normalise_config(json.load(fh), ns.config))
        cfg["config"] = ns.config
    cfg.update({k: v for k, v in vars(ns).items() if v is not None and k in DEFAULTS and k != "config"})
    if not cfg.get("input_fasta") and not cfg.get("input_txt"):
        raise ValueError("Provide --input-fasta and/or --input-txt (directly or via --config).")
    if cfg.get("push_to_hub") and not cfg.get("hub_repo_id"):
        raise ValueError("--push-to-hub requires --hub-repo-id")
    return cfg


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def _is_peft(model) -> bool:
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def _merged_state_dict(model) -> Dict[str, torch.Tensor]:
    """Full-weight state dict of a (possibly LoRA-wrapped) model, without mutating it."""
    if not _is_peft(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.merge_adapter()
    try:
        state = {}
        for k, v in model.get_base_model().state_dict().items():
            if "lora_" in k:
                continue
            state[k.replace(".base_layer.", ".")] = v.detach().cpu().clone()
    finally:
        model.unmerge_adapter()
    return state


def save_mlm_checkpoint(model, tokenizer, out_dir: str, meta: dict, training_state: Optional[dict] = None):
    """Write an HF-format directory loadable by esm3di train --init-from."""
    tmp_dir = out_dir.rstrip("/") + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    base = model.get_base_model() if _is_peft(model) else model
    state = _merged_state_dict(model)
    if _is_peft(model):
        # save_pretrained on the unwrapped class with explicit merged weights
        base.save_pretrained(tmp_dir, state_dict=state, safe_serialization=True)
        model.save_pretrained(os.path.join(tmp_dir, "adapter"))
    else:
        base.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    with open(os.path.join(tmp_dir, "esm3di_mlm.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    if training_state is not None:
        torch.save(training_state, os.path.join(tmp_dir, "training_state.pt"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)


def _find_latest_epoch_dir(output_dir: str) -> Optional[str]:
    found = []
    for path in glob.glob(os.path.join(output_dir, "epoch_*")):
        m = re.search(r"epoch_(\d+)$", path)
        if m and os.path.isfile(os.path.join(path, "training_state.pt")):
            found.append((int(m.group(1)), path))
    return max(found)[1] if found else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mlm_loss(logits, labels):
    return nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


@torch.no_grad()
def evaluate(model, sequences, tokenizer, cfg, device, amp_ctx) -> Dict[str, float]:
    """Masked-token loss/accuracy with a fixed masking seed (comparable across epochs)."""
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        SequenceDataset(sequences),
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=make_mlm_collate(tokenizer, cfg["mlm_probability"], cfg["max_length"], generator=g),
        num_workers=0,
    )
    model.eval()
    loss_sum, n_tok, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_ctx():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        labels = batch["labels"]
        valid = labels != -100
        n = int(valid.sum())
        loss_sum += _mlm_loss(logits, labels).item() * n
        correct += int(((logits.argmax(-1) == labels) & valid).sum())
        n_tok += n
    model.train()
    loss = loss_sum / max(n_tok, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50)), "accuracy": correct / max(n_tok, 1)}


def train(cfg: dict) -> str:
    """Run MLM fine-tuning; returns the path of the best checkpoint directory."""
    _set_seed(cfg["seed"])
    out = cfg["output_dir"]
    if not cfg.get("resume") and glob.glob(os.path.join(out, "epoch_*")) + glob.glob(os.path.join(out, "best")) + glob.glob(os.path.join(out, "final")):
        raise FileExistsError(
            f"{out} already contains MLM checkpoints; use --resume auto or choose a new --output-dir "
            "(existing checkpoints are never overwritten by a fresh run)."
        )
    os.makedirs(out, exist_ok=True)
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- data
    train_seqs = load_sequences(cfg.get("input_fasta"), cfg.get("input_txt"), cfg["min_length"])
    val_seqs = load_sequences(cfg.get("val_fasta"), None, cfg["min_length"])
    if not val_seqs and cfg["val_split"] > 0:
        rng = random.Random(cfg["seed"])
        rng.shuffle(train_seqs)
        n_val = max(1, int(len(train_seqs) * cfg["val_split"]))
        val_seqs, train_seqs = train_seqs[:n_val], train_seqs[n_val:]
    if cfg.get("max_train_samples"):
        train_seqs = train_seqs[: cfg["max_train_samples"]]
    if cfg.get("max_val_samples"):
        val_seqs = val_seqs[: cfg["max_val_samples"]]
    if not train_seqs:
        raise ValueError("No training sequences")
    print(f"Train sequences: {len(train_seqs)}, validation sequences: {len(val_seqs)}")

    # ---- model
    resume_dir = cfg.get("resume")
    if resume_dir == "auto":
        resume_dir = _find_latest_epoch_dir(out)
        print(f"Auto-resume: {resume_dir or 'no checkpoint found, starting fresh'}")
    state = None
    lora_resume = False
    if resume_dir:
        state = torch.load(os.path.join(resume_dir, "training_state.pt"), map_location="cpu", weights_only=False)
        lora_resume = os.path.isdir(os.path.join(resume_dir, "adapter"))
        if lora_resume:
            # Rebuild the adapter on top of the original base so optimizer state lines up.
            model, base_name, _ = load_mlm_model(state["init_model"], cfg.get("hf_revision"))
        else:
            model, base_name, _ = load_mlm_model(resume_dir, cfg.get("hf_revision"), base_model=state["base_model"])
        init_model = state["init_model"]
    else:
        model, base_name, report = load_mlm_model(cfg["model_name"], cfg.get("hf_revision"))
        init_model = cfg["model_name"]
        if report is not None:
            print(f"Continuing from MLM checkpoint {cfg['model_name']}: {report.summary()}")
    revision = resolve_hf_revision(base_name, cfg.get("hf_revision"))
    tokenizer = get_tokenizer(model, base_name, revision)

    if cfg["use_lora"]:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if lora_resume:
            model = PeftModel.from_pretrained(model, os.path.join(resume_dir, "adapter"), is_trainable=True)
        else:
            from ..ESM3di_model import discover_lora_target_modules

            targets = _as_list(cfg.get("lora_targets")) or [
                m for m in discover_lora_target_modules(model)
                if not any(x in m for x in ("head", "classifier", "embed"))
            ]
            model = get_peft_model(model, LoraConfig(
                r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                bias="none", task_type=TaskType.MASKED_LM, target_modules=targets,
            ))
        model.print_trainable_parameters()
    model.to(device).train()

    train_loader = DataLoader(
        SequenceDataset(train_seqs),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        collate_fn=make_mlm_collate(tokenizer, cfg["mlm_probability"], cfg["max_length"]),
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    from transformers import get_linear_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    accum = max(1, cfg["grad_accum_steps"])
    total_steps = math.ceil(len(train_loader) / accum) * cfg["epochs"]
    warmup = cfg["warmup_steps"] or int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    precision = cfg["precision"]
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")

    def amp_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_val = state.get("best_val", float("inf"))
        print(f"Resumed from {resume_dir} (epoch {state['epoch']}, step {global_step})")

    meta_base = {
        "format": "esm3di-mlm-v1",
        "base_model": base_name,
        "hf_revision": revision,
        "init_model": init_model,
        "config": {k: v for k, v in cfg.items() if not k.startswith("hub_")},
    }

    best_dir = os.path.join(out, "best")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with amp_ctx():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = _mlm_loss(logits, batch["labels"])
            scaler.scale(loss / accum).backward()
            running += loss.item()
            n_batches += 1
            if step % accum == 0 or step == len(train_loader):
                if cfg["max_grad_norm"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg["log_every"] == 0:
                    print(f"epoch {epoch} step {global_step}/{total_steps} loss {running / n_batches:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        metrics = {"train_loss": running / max(n_batches, 1)}
        if val_seqs:
            metrics.update({f"val_{k}": v for k, v in evaluate(model, val_seqs, tokenizer, cfg, device, amp_ctx).items()})
        score = metrics.get("val_loss", metrics["train_loss"])
        is_best = score < best_val
        best_val = min(best_val, score)
        print(f"== epoch {epoch} done in {(time.time() - t0) / 60:.1f} min: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + (" (best)" if is_best else ""), flush=True)

        meta = dict(meta_base, epoch=epoch, global_step=global_step, metrics=metrics)
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "base_model": base_name,
            "init_model": init_model,
        }
        epoch_dir = os.path.join(out, f"epoch_{epoch}")
        save_mlm_checkpoint(model, tokenizer, epoch_dir, meta, training_state)
        if is_best:
            save_mlm_checkpoint(model, tokenizer, best_dir, meta)
        with open(os.path.join(out, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps({"epoch": epoch, "global_step": global_step, **metrics}) + "\n")
        print(f"  saved {epoch_dir}")

        if cfg.get("keep_last_n"):
            olds = sorted(
                [d for d in glob.glob(os.path.join(out, "epoch_*")) if re.search(r"epoch_\d+$", d)],
                key=lambda d: int(re.search(r"epoch_(\d+)$", d).group(1)),
            )
            for old in olds[: -cfg["keep_last_n"]]:
                shutil.rmtree(old)

    final_meta = dict(meta_base, epoch=cfg["epochs"], global_step=global_step)
    test_seqs = load_sequences(cfg.get("test_fasta"), None, cfg["min_length"])
    if test_seqs:
        test = evaluate(model, test_seqs, tokenizer, cfg, device, amp_ctx)
        final_meta["test_metrics"] = test
        print("Test: " + ", ".join(f"{k}={v:.4f}" for k, v in test.items()))
    final_dir = os.path.join(out, "final")
    save_mlm_checkpoint(model, tokenizer, final_dir, final_meta)
    if not os.path.isdir(best_dir):
        best_dir = final_dir

    if cfg.get("push_to_hub"):
        from ..hf_publish import publish_directory_to_hub

        url = publish_directory_to_hub(
            local_dir=best_dir,
            repo_id=cfg["hub_repo_id"],
            private=cfg["hub_private"],
            revision=cfg["hub_revision"],
            commit_message="Upload ESM3Di MLM checkpoint",
            token_env_var=cfg["hub_token_env_var"],
        )
        print(f"Uploaded {best_dir} to {url}")

    print(f"MLM training finished. Best checkpoint: {best_dir}")
    return best_dir


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Masked-LM fine-tuning of ESM++ on protein sequences.")
    add_train_mlm_args(parser)
    cfg = resolve_config(parser.parse_args(argv))
    train(cfg)


if __name__ == "__main__":
    main()
