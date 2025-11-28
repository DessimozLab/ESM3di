#!/usr/bin/env python
import argparse, os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, TaskType


# -----------------------------
# FASTA utilities
# -----------------------------

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Very simple FASTA parser.
    Returns list of (header_without_>, sequence_string).
    """
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip().upper())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records


# -----------------------------
# Dataset
# -----------------------------

class Seq3DiDataset(Dataset):
    """
    Holds (amino_acid_sequence, 3Di_label_sequence) pairs.
    Assumes 1:1 correspondence and equal lengths.

    mask_label_chars: characters in 3Di sequences that indicate "masked" positions
                      (e.g. low pLDDT). Those chars:
                      - are NOT part of label_vocab / model outputs
                      - are ignored in the loss (labels = -100).
    """
    def __init__(
        self,
        aa_fasta: str,
        three_di_fasta: str,
        mask_label_chars: str = ""
    ):
        aa_records = read_fasta(aa_fasta)
        lab_records = read_fasta(three_di_fasta)

        assert len(aa_records) == len(lab_records), "Mismatched number of sequences"

        self.items = []
        all_chars = set()
        self.mask_label_chars = set(mask_label_chars) if mask_label_chars else set()

        for (h_aa, seq_aa), (h_lab, seq_lab) in zip(aa_records, lab_records):
            if len(seq_aa) != len(seq_lab):
                raise ValueError(
                    f"Length mismatch {h_aa}/{h_lab}: {len(seq_aa)} vs {len(seq_lab)}"
                )
            self.items.append((h_aa, seq_aa, seq_lab))
            all_chars.update(seq_lab)

        # Build vocab from all non-masked characters
        label_chars = sorted(ch for ch in all_chars if ch not in self.mask_label_chars)
        self.label_vocab = label_chars
        self.char2idx = {c: i for i, c in enumerate(self.label_vocab)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # (header, aa_seq, 3di_seq)
        return self.items[idx]


# -----------------------------
# Collate with HF tokenizer
# -----------------------------

def make_collate_fn(tokenizer, char2idx, mask_label_chars: str = ""):
    """
    - Tokenizes AA sequences with HF ESM tokenizer.
    - Uses special_tokens_mask to align per-residue 3Di labels
      to non-special tokens.
    - Positions whose 3Di label is in mask_label_chars are set to -100
      (ignored in the loss) and do NOT belong to label_vocab.
    """
    mask_set = set(mask_label_chars) if mask_label_chars else set()

    def collate(batch):
        headers, aa_seqs, label_seqs = zip(*batch)

        enc = tokenizer(
            list(aa_seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"]              # [B, T]
        attention_mask = enc["attention_mask"]    # [B, T]
        special_mask = enc["special_tokens_mask"] # [B, T]

        batch_size, max_len = input_ids.shape
        labels = torch.full(
            (batch_size, max_len),
            -100,  # ignore_index for CE
            dtype=torch.long,
        )

        for i, lab_seq in enumerate(label_seqs):
            k = 0  # index into label sequence
            for j in range(max_len):
                if special_mask[i, j] == 1:
                    # CLS/EOS/PAD etc.
                    labels[i, j] = -100
                else:
                    if k < len(lab_seq):
                        ch = lab_seq[k]
                        if ch in mask_set:
                            # masked (e.g. low pLDDT) -> ignore in loss
                            labels[i, j] = -100
                        else:
                            try:
                                labels[i, j] = char2idx[ch]
                            except KeyError:
                                raise ValueError(
                                    f"Label char '{ch}' not in vocabulary and not in "
                                    f"mask_label_chars; check your data."
                                )
                        k += 1
                    else:
                        labels[i, j] = -100
            if k != len(lab_seq):
                # Safety check: we should have consumed all labels
                raise ValueError(
                    f"Did not consume all labels for sequence {headers[i]}: "
                    f"used {k}, have {len(lab_seq)}"
                )

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch_out

    return collate


# -----------------------------
# LoRA / PEFT helpers
# -----------------------------

def freeze_all_but_lora_and_classifier(model):
    """
    Explicitly freeze everything except:
      - LoRA parameters (names contain 'lora_')
      - classifier head (names contain 'classifier')
    """
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            p.requires_grad = True


# -----------------------------
# Training
# -----------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # 1) Data (with mask_label_chars)
    dataset = Seq3DiDataset(
        args.aa_fasta,
        args.three_di_fasta,
        mask_label_chars=args.mask_label_chars,
    )
    print(f"Loaded {len(dataset)} sequences")
    print(f"3Di vocab ({len(dataset.label_vocab)}): {dataset.label_vocab}")
    if args.mask_label_chars:
        print(f"Masked 3Di chars (ignored in loss): {list(set(args.mask_label_chars))}")

    # 2) HF tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    base_model = AutoModelForTokenClassification.from_pretrained(
        args.hf_model,
        num_labels=len(dataset.label_vocab),
    )

    # 3) PEFT LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=(
            args.lora_target_modules.split(",")
            if args.lora_target_modules
            else None
        ),
    )

    # 4) Wrap with PEFT and freeze base weights
    model = get_peft_model(base_model, lora_config)
    freeze_all_but_lora_and_classifier(model)
    model.to(device)

    # Optional sanity check
    try:
        model.print_trainable_parameters()
    except AttributeError:
        pass

    # 5) DataLoader (pass mask_label_chars into collate)
    collate_fn = make_collate_fn(
        tokenizer,
        dataset.char2idx,
        mask_label_chars=args.mask_label_chars,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # 6) Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 7) Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_tokens = 0

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            token_count = (batch["labels"] != -100).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * max(token_count, 1)
            running_tokens += max(token_count, 1)

            if step % args.log_every == 0:
                avg_loss = running_loss / max(running_tokens, 1)
                print(f"Epoch {epoch} Step {step} | Loss/residue: {avg_loss:.4f}")
                running_loss = 0.0
                running_tokens = 0

        # Save checkpoint each epoch
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_vocab": dataset.label_vocab,
                "mask_label_chars": args.mask_label_chars,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


# -----------------------------
# Inference helper
# -----------------------------

@torch.no_grad()
def predict_3di_for_fasta(model_ckpt: str, aa_fasta: str, device: str = None):
    """
    Load LoRA-ESM 3Di tagger from a checkpoint and predict 3Di
    sequences for an AA FASTA.

    Note: predictions are only over the non-masked 3Di vocabulary.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(model_ckpt, map_location=device)
    label_vocab = ckpt["label_vocab"]
    idx2char = {i: c for i, c in enumerate(label_vocab)}
    args = ckpt["args"]
    mask_label_chars = ckpt.get("mask_label_chars", "")

    tokenizer = AutoTokenizer.from_pretrained(args["hf_model"])
    base_model = AutoModelForTokenClassification.from_pretrained(
        args["hf_model"],
        num_labels=len(label_vocab),
    )

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        target_modules=(
            args["lora_target_modules"].split(",")
            if args["lora_target_modules"]
            else None
        ),
    )

    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    records = read_fasta(aa_fasta)
    results = []

    for header, seq in records:
        enc = tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits[0]  # [T, num_labels]
        special_mask = enc["special_tokens_mask"][0]  # [T]

        pred_indices = logits.argmax(dim=-1)
        pred_chars = []
        k = 0
        for j in range(pred_indices.shape[0]):
            if special_mask[j] == 1:
                continue
            pred_chars.append(idx2char[int(pred_indices[j])])
            k += 1

        if k != len(seq):
            print(
                f"Warning: predicted length {k} != seq length {len(seq)} for {header}"
            )

        results.append((header, seq, "".join(pred_chars)))

    return results


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ESM (HuggingFace) + PEFT LoRA 3Di per-residue classifier"
    )
    p.add_argument("--aa-fasta", type=str, required=True,
                   help="FASTA with amino-acid sequences.")
    p.add_argument("--three-di-fasta", type=str, required=True,
                   help="FASTA with matching 3Di sequences (same lengths).")

    p.add_argument("--hf-model", type=str, default="facebook/esm2_t33_650M_UR50D",
                   help="HuggingFace model name for ESM-2.")

    # Masking
    p.add_argument(
        "--mask-label-chars",
        type=str,
        default="X",
        help=(
            "Characters in 3Di FASTA to treat as masked (e.g. low pLDDT). "
            "These positions are ignored during training and are NOT part "
            "of the model output alphabet. You can pass multiple chars, e.g. 'X?'."
        ),
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,fc1,fc2",
        help="Comma-separated list of module names to apply LoRA to.",
    )

    # Training
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="esm_hf_peft_3di_ckpts")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
