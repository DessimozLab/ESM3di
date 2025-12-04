#!/usr/bin/env python
import argparse
import json
import os
import socket
from typing import List, Tuple
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ESM3di_model import ESM3DiModel, read_fasta, Seq3DiDataset, make_collate_fn

# -----------------------------
# Training
# -----------------------------

def train(args):
    # Setup device with explicit GPU selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Setup TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.tensorboard_log_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n{'='*60}")
    print(f"TensorBoard Logging Setup")
    print(f"{'='*60}")
    print(f"Log directory: {log_dir}")
    print(f"\nTo view training progress in real-time, run:")
    print(f"  tensorboard --logdir={args.tensorboard_log_dir}")
    print(f"\nTensorBoard will be available at:")
    print(f"  http://localhost:6006")
    print(f"\nTo use a different port (e.g., 6007):")
    print(f"  tensorboard --logdir={args.tensorboard_log_dir} --port=6007")
    print(f"\nTo access from a remote machine, use SSH tunneling:")
    print(f"  ssh -L 6006:localhost:6006 user@{socket.gethostname()}")
    print(f"{'='*60}\n")

    # 1) Data (with mask_label_chars)
    dataset = Seq3DiDataset(
        args.aa_fasta,
        args.three_di_fasta,
        mask_label_chars=args.mask_label_chars,
    )
    print(f"Loaded {len(dataset)} sequences")
    print(f"3Di vocab ({len(dataset.label_vocab)}): {dataset.label_vocab}")
    if args.mask_label_chars:
        print(f"Masked 3Di chars (ignored in loss): "
              f"{list(set(args.mask_label_chars))}")

    # 2) HF tokenizer + model
    print(f"\nLoading tokenizer: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    print("✓ Tokenizer loaded")

    # Create ESM3Di model wrapper
    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    esm_model = ESM3DiModel(
        hf_model_name=args.hf_model,
        num_labels=len(dataset.label_vocab),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    model = esm_model.get_model()
    model.to(device)
    print("✓ Model with LoRA loaded")

    # Print trainable parameters
    try:
        model.print_trainable_parameters()
    except AttributeError:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} || Total params: {total:,} || "
              f"Trainable %: {100 * trainable / total:.2f}")

    # 6) DataLoader (pass mask_label_chars into collate)
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
    # 7) Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 8) Learning rate scheduler setup
    # Calculate total training steps
    total_steps = len(loader) * args.epochs
    
    # Determine warmup steps (default: 10% of total steps, or user-specified)
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.warmup_ratio is not None:
        warmup_steps = int(total_steps * args.warmup_ratio)
    else:
        warmup_steps = int(total_steps * 0.1)  # Default: 10% warmup
    
    # Create scheduler based on specified type
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler_name = "Cosine with warmup"
    elif args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler_name = "Linear with warmup"
    elif args.scheduler_type == 'constant':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # Keeps LR constant after warmup
        )
        scheduler_name = "Constant with warmup"
    else:
        # No scheduler
        scheduler = None
        scheduler_name = "None (constant LR)"
    
    # 8) Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"Learning rate scheduling: {scheduler_name}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"  Initial LR: {args.lr:.2e}")
    model.train()
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        running_loss = 0.0
        running_tokens = 0
        epoch_loss = 0.0
        epoch_tokens = 0

        # Create progress bar for batches
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")

        for step, batch in enumerate(progress_bar, start=1):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            token_count = (batch["labels"] != -100).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Calculate per-residue loss for this batch
            per_residue_loss = loss.item() / max(token_count, 1)
            
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/learning_rate', current_lr, global_step)

            running_loss += loss.item() * max(token_count, 1)
            running_tokens += max(token_count, 1)
            epoch_loss += loss.item() * max(token_count, 1)
            epoch_tokens += max(token_count, 1)

            # Update progress bar with current loss and LR
            avg_loss = running_loss / max(running_tokens, 1)
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loss/residue": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Log to TensorBoard every step
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/train_per_residue_step', per_residue_loss, global_step)

            if step % args.log_every == 0:
                # Log running average to TensorBoard
                writer.add_scalar('Loss/train_running_avg', avg_loss, global_step)
                writer.add_scalar('Training/tokens_processed', running_tokens, global_step)
                running_loss = 0.0
                running_tokens = 0

        # Log epoch-level metrics to TensorBoard
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        writer.add_scalar('Loss/train_epoch', epoch_avg_loss, epoch)
        writer.add_scalar('Training/epoch_tokens', epoch_tokens, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} Average Loss (per residue): {epoch_avg_loss:.4f}")
        print(f"Current Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint each epoch
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "label_vocab": dataset.label_vocab,
            "mask_label_chars": args.mask_label_chars,
            "lora_target_modules": target_modules,
            "args": vars(args),
            "epoch": epoch,
            "loss": epoch_avg_loss,
            "global_step": global_step,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")
        print()
    
    # Close TensorBoard writer and print final instructions
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"\nTo view training results:")
    print(f"  tensorboard --logdir={args.tensorboard_log_dir}")
    print(f"\nCheckpoints saved to: {args.out_dir}")
    print(f"{'='*60}\n")


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
    
    # Get target modules from checkpoint (with fallback)
    target_modules = ckpt.get("lora_target_modules")
    if not target_modules:
        # Fallback: try to get from args or use default
        if "lora_target_modules" in args and args["lora_target_modules"]:
            target_modules = args["lora_target_modules"].split(",")
        else:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                              'fc1', 'fc2']

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
        target_modules=target_modules,
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
                f"Warning: predicted length {k} != seq length {len(seq)} "
                f"for {header}"
            )

        results.append((header, seq, "".join(pred_chars)))

    return results


# -----------------------------
# CLI
# -----------------------------

def load_config_file(config_path: str) -> dict:
    """Load training configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_args():
    p = argparse.ArgumentParser(
        description="ESM (HuggingFace) + PEFT LoRA 3Di per-residue classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file. If provided, overrides "
                        "other arguments.")
    
    # Data
    p.add_argument("--aa-fasta", type=str, required=False,
                   help="FASTA with amino-acid sequences.")
    p.add_argument("--three-di-fasta", type=str, required=False,
                   help="FASTA with matching 3Di sequences (same lengths).")

    # Model
    p.add_argument("--hf-model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="HuggingFace model name for ESM-2. Options: "
                        "esm2_t12_35M_UR50D (35M params, fast), "
                        "esm2_t30_150M_UR50D (150M params), "
                        "esm2_t33_650M_UR50D (650M params, best quality)")

    # Masking
    p.add_argument(
        "--mask-label-chars",
        type=str,
        default="X",
        help="Characters in 3Di FASTA to treat as masked (e.g. low pLDDT). "
             "These positions are ignored during training and are NOT part "
             "of the model output alphabet. You can pass multiple chars, "
             "e.g. 'X?'.",
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=8,
                   help="LoRA rank (r)")
    p.add_argument("--lora-alpha", type=float, default=16.0,
                   help="LoRA alpha parameter")
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="LoRA dropout rate")
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="",
        help="Comma-separated list of module names to apply LoRA to. "
             "If empty, will auto-discover attention modules. "
             "Example: 'q_proj,k_proj,v_proj,o_proj,fc1,fc2'",
    )

    # Training
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Weight decay for optimizer")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of DataLoader workers")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log training progress every N steps")
    
    # Learning rate scheduler
    p.add_argument("--scheduler-type", type=str, default='cosine',
                   choices=['cosine', 'linear', 'constant', 'none'],
                   help="Learning rate scheduler type. 'cosine' uses cosine decay after warmup, "
                        "'linear' uses linear decay, 'constant' keeps LR constant after warmup, "
                        "'none' uses no scheduler (constant LR throughout)")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Number of warmup steps. If not specified, uses warmup-ratio")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="Warmup ratio (fraction of total steps). Default: 0.1 (10%%)")
    
    # Device
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu'). "
                        "If not specified, uses CUDA if available.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available "
                        "(ignored if --device is specified)")
    
    # Output
    p.add_argument("--out-dir", type=str, default="esm_hf_peft_3di_ckpts",
                   help="Directory to save model checkpoints")
    p.add_argument("--tensorboard-log-dir", type=str, default="tensorboard_logs",
                   help="Directory to save TensorBoard logs")
    
    args = p.parse_args()
    
    # Load config file if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
        
        # Update args with config values (config file takes precedence)
        for key, value in config.items():
            # Convert config keys to arg format (e.g., 'aa_fasta' -> 'aa_fasta')
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                setattr(args, arg_key, value)
                print(f"  {arg_key}: {value}")
    
    # Validate required arguments
    if not args.aa_fasta or not args.three_di_fasta:
        p.error("--aa-fasta and --three-di-fasta are required "
                "(or must be in config file)")
    
    return args


def main():
    """Main entry point for training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
