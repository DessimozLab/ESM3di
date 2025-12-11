#!/usr/bin/env python
import argparse
import json
import os
import socket
from typing import List, Tuple
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ESM3di_model import ESM3DiModel, read_fasta, Seq3DiDataset, make_collate_fn
from esm.models.esmc import ESMC

# -----------------------------
# Distributed Training Setup
# -----------------------------

def setup_distributed():
    """Initialize distributed training environment."""
    # Get rank and world_size from environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


# -----------------------------
# Training
# -----------------------------

def train(args):
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Only print from main process
    def print_main(*args_print, **kwargs):
        if is_main_process():
            print(*args_print, **kwargs)
    
    # Setup device
    if world_size > 1:
        device = f"cuda:{local_rank}"
    elif args.device:
        device = args.device
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    
    print_main(f"World size: {world_size}")
    print_main(f"Rank: {rank}, Local rank: {local_rank}")
    print_main(f"Using device: {device}")
    
    # Setup TensorBoard logging (only on main process)
    writer = None
    if is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.tensorboard_log_dir, f"run_{timestamp}")
        writer = SummaryWriter(log_dir=log_dir)
        
        print_main(f"\n{'='*60}")
        print_main(f"TensorBoard Logging Setup")
        print_main(f"{'='*60}")
        print_main(f"Log directory: {log_dir}")
        print_main(f"\nTo view training progress in real-time, run:")
        print_main(f"  tensorboard --logdir={args.tensorboard_log_dir}")
        print_main(f"\nTensorBoard will be available at:")
        print_main(f"  http://localhost:6006")
        print_main(f"\nTo use a different port (e.g., 6007):")
        print_main(f"  tensorboard --logdir={args.tensorboard_log_dir} --port=6007")
        print_main(f"\nTo access from a remote machine, use SSH tunneling:")
        print_main(f"  ssh -L 6006:localhost:6006 user@{socket.gethostname()}")
        print_main(f"{'='*60}\n")

    # 1) Data (with mask_label_chars)
    dataset = Seq3DiDataset(
        args.aa_fasta,
        args.three_di_fasta,
        mask_label_chars=args.mask_label_chars,
    )
    print_main(f"Loaded {len(dataset)} sequences")
    print_main(f"3Di vocab ({len(dataset.label_vocab)}): {dataset.label_vocab}")
    if args.mask_label_chars:
        print_main(f"Masked 3Di chars (ignored in loss): "
              f"{list(set(args.mask_label_chars))}")

    # 2) HF tokenizer + model
    print_main(f"\nLoading tokenizer: {args.hf_model}")
    if args.hf_model.__contains__('esmc'):
        # For ESMC models, use the ESM library tokenizer
        esm_model = ESMC.from_pretrained("esmc_300m")  # This loads both model and tokenizer
        tokenizer = esm_model.tokenizer
    else:
        # For ESM-2 and other HuggingFace models
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    print_main("✓ Tokenizer loaded")

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
        target_modules=target_modules,
        use_cnn_head=args.use_cnn_head,
        cnn_num_layers=args.cnn_num_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_dropout=args.cnn_dropout
    )
    model = esm_model.get_model()
    model.to(device)
    
    # Wrap model with DDP if using multi-GPU
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set to True if you have unused parameters
        )
        print_main("✓ Model wrapped with DistributedDataParallel")
    
    print_main("✓ Model with LoRA loaded")

    # Print trainable parameters (access underlying model if wrapped in DDP)
    model_to_print = model.module if isinstance(model, DDP) else model
    try:
        model_to_print.print_trainable_parameters()
    except AttributeError:
        trainable = sum(p.numel() for p in model_to_print.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_to_print.parameters())
        print_main(f"Trainable params: {trainable:,} || Total params: {total:,} || "
              f"Trainable %: {100 * trainable / total:.2f}")

    # 3) DataLoader with DistributedSampler for multi-GPU
    collate_fn = make_collate_fn(
        tokenizer,
        dataset.char2idx,
        mask_label_chars=args.mask_label_chars,
    )
    
    # Use DistributedSampler if multi-GPU
    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        shuffle = False  # Sampler handles shuffling
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.startswith("cuda") else False,
    )
    
    # 4) Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 5) Learning rate scheduler setup
    # Calculate total training steps (accounting for gradient accumulation)
    steps_per_epoch = len(loader) // args.gradient_accumulation_steps
    if len(loader) % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    
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
    
    # Load checkpoint if resuming training
    start_epoch = 1
    global_step = 0
    accumulation_step = 0
    
    if args.resume_from_checkpoint:
        print_main(f"\n{'='*60}")
        print_main(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        print_main(f"{'='*60}")
        
        # Load checkpoint to CPU first to avoid GPU memory issues
        map_location = {"cuda:0": device} if device.startswith("cuda") else device
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=map_location)
        
        # Load model state (handle DDP wrapper)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        print_main("✓ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print_main("✓ Optimizer state loaded")
        
        # Load scheduler state if available
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print_main("✓ Scheduler state loaded")
        
        # Resume from next epoch
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        
        print_main(f"\nResuming from:")
        print_main(f"  Epoch: {checkpoint.get('epoch', 0)} (will start at {start_epoch})")
        print_main(f"  Global step: {global_step}")
        print_main(f"  Previous loss: {checkpoint.get('loss', 'N/A'):.4f}" if isinstance(checkpoint.get('loss'), (int, float)) else f"  Previous loss: N/A")
        print_main(f"{'='*60}\n")
    
    # 6) Training loop
    print_main(f"\nStarting training for {args.epochs} epochs...\n")
    print_main(f"Batch size per GPU: {args.batch_size}")
    print_main(f"Number of GPUs: {world_size}")
    print_main(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print_main(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    print_main(f"Learning rate scheduling: {scheduler_name}")
    print_main(f"  Total optimization steps: {total_steps}")
    print_main(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print_main(f"  Initial LR: {args.lr:.2e}")
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        print_main(f"{'='*60}")
        print_main(f"EPOCH {epoch}/{args.epochs}")
        print_main(f"{'='*60}")
        
        # Set epoch for DistributedSampler (ensures different shuffle each epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_tokens = 0
        epoch_loss = 0.0
        epoch_tokens = 0

        # Create progress bar for batches (only on main process)
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", disable=not is_main_process())

        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            token_count = (batch["labels"] != -100).sum().item()

            # Scale loss by accumulation steps for proper gradient averaging
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulation_step += 1

            # Calculate per-residue loss for this batch
            per_residue_loss = loss.item() / max(token_count, 1)

            running_loss += loss.item() * max(token_count, 1)
            running_tokens += max(token_count, 1)
            epoch_loss += loss.item() * max(token_count, 1)
            epoch_tokens += max(token_count, 1)

            # Perform optimization step after accumulating enough gradients
            if accumulation_step % args.gradient_accumulation_steps == 0:
                global_step += 1
                
                # Gradient clipping (optional but recommended for multi-GPU)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Step the learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Log current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                if writer is not None:
                    writer.add_scalar('Training/learning_rate', current_lr, global_step)
                
                # Update progress bar with current loss and LR
                avg_loss = running_loss / max(running_tokens, 1)
                if is_main_process():
                    progress_bar.set_postfix({
                        "loss/residue": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step
                    })
                
                # Log to TensorBoard
                if writer is not None and global_step % args.log_every == 0:
                    writer.add_scalar('Loss/train_step', avg_loss, global_step)
                    writer.add_scalar('Loss/train_running_avg', avg_loss, global_step)
                    writer.add_scalar('Training/tokens_processed', running_tokens, global_step)
                    running_loss = 0.0
                    running_tokens = 0
            else:
                # Just update progress bar during accumulation
                avg_loss = running_loss / max(running_tokens, 1)
                if is_main_process():
                    progress_bar.set_postfix({
                        "loss/residue": f"{avg_loss:.4f}",
                        "accum": f"{accumulation_step % args.gradient_accumulation_steps}/{args.gradient_accumulation_steps}"
                    })
            
            # Always log per-batch loss (before accumulation)
            if writer is not None:
                writer.add_scalar('Loss/train_batch', loss.item(), step + (epoch - 1) * len(loader))
                writer.add_scalar('Loss/train_per_residue_batch', per_residue_loss, step + (epoch - 1) * len(loader))

        # Synchronize metrics across all processes
        if world_size > 1:
            # Sum epoch_loss and epoch_tokens across all processes
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            epoch_tokens_tensor = torch.tensor(epoch_tokens, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item()
            epoch_tokens = epoch_tokens_tensor.item()
        
        # Log epoch-level metrics to TensorBoard
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', epoch_avg_loss, epoch)
            writer.add_scalar('Training/epoch_tokens', epoch_tokens, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        print_main(f"\nEpoch {epoch} Average Loss (per residue): {epoch_avg_loss:.4f}")
        print_main(f"Current Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint each epoch (only from main process)
        if is_main_process():
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            
            # Get underlying model if wrapped in DDP
            model_to_save = model.module if isinstance(model, DDP) else model
            
            torch.save(
                {
                "model_state_dict": model_to_save.state_dict(),
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
            print_main(f"Saved checkpoint to {ckpt_path}")
        
        # Wait for all processes to reach this point
        if world_size > 1:
            dist.barrier()
        
        print_main()
    
    # Close TensorBoard writer and print final instructions
    if writer is not None:
        writer.close()
        print_main(f"\n{'='*60}")
        print_main(f"Training Complete!")
        print_main(f"{'='*60}")
        print_main(f"\nTensorBoard logs saved to: {log_dir}")
        print_main(f"\nTo view training results:")
        print_main(f"  tensorboard --logdir={args.tensorboard_log_dir}")
        print_main(f"\nCheckpoints saved to: {args.out_dir}")
        print_main(f"{'='*60}\n")
    
    # Cleanup distributed training
    cleanup_distributed()


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
        description="ESM (HuggingFace) + PEFT LoRA 3Di per-residue classifier - Multi-GPU Training",
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
                   help="HuggingFace model name or ESMC model identifier. "
                        "ESM-2 options: esm2_t12_35M_UR50D (35M params), "
                        "esm2_t30_150M_UR50D (150M), esm2_t33_650M_UR50D (650M). "
                        "ESMC options: esmc-300m-2024-12 (300M params), "
                        "esmc-600m-2024-12 (600M params). "
                        "Note: ESMC models require 'pip install esm'")

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

    # CNN Classification Head
    p.add_argument("--use-cnn-head", action="store_true",
                   help="Use multi-layer CNN classification head instead of "
                        "linear classifier. Adds learnable convolutional layers "
                        "before final classification for better local context.")
    p.add_argument("--cnn-num-layers", type=int, default=2,
                   help="Number of CNN layers in classification head "
                        "(only used if --use-cnn-head is set)")
    p.add_argument("--cnn-kernel-size", type=int, default=3,
                   help="Kernel size for CNN layers "
                        "(only used if --use-cnn-head is set)")
    p.add_argument("--cnn-dropout", type=float, default=0.1,
                   help="Dropout rate for CNN layers "
                        "(only used if --use-cnn-head is set)")

    # Training
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size per GPU")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Number of gradient accumulation steps. Effective batch size = "
                        "batch_size * num_gpus * gradient_accumulation_steps. Use to simulate "
                        "larger batch sizes when GPU memory is limited.")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Weight decay for optimizer")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Maximum gradient norm for clipping. Set to 0 to disable.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of DataLoader workers per GPU")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log training progress every N optimization steps")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt file) to resume training from. ")
    
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
    
    # Device (note: multi-GPU uses torchrun, so --device is typically not needed)
    p.add_argument("--device", type=str, default=None,
                   help="Device to use for single-GPU training (e.g., 'cuda:0', 'cuda:1', 'cpu'). "
                        "For multi-GPU training, use torchrun instead and this will be ignored.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available "
                        "(ignored if --device is specified or using torchrun)")
    
    # Output
    p.add_argument("--out-dir", type=str, default="esm_hf_peft_3di_ckpts",
                   help="Directory to save model checkpoints")
    p.add_argument("--tensorboard-log-dir", type=str, default="tensorboard_logs",
                   help="Directory to save TensorBoard logs")
    
    args = p.parse_args()
    
    # Load config file if provided
    if args.config:
        # Only print from rank 0 in distributed setting
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
        
        # Update args with config values (config file takes precedence)
        for key, value in config.items():
            # Convert config keys to arg format (e.g., 'aa_fasta' -> 'aa_fasta')
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                setattr(args, arg_key, value)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"  {arg_key}: {value}")
    
    # Validate required arguments
    if not args.aa_fasta or not args.three_di_fasta:
        p.error("--aa-fasta and --three-di-fasta are required "
                "(or must be in config file)")
    
    return args


def main():
    """
    Main entry point for multi-GPU training.
    
    Usage:
        Single GPU:
            python esmretrain_multigpu.py --config config.json
        
        Multi-GPU (recommended):
            torchrun --nproc_per_node=NUM_GPUS esmretrain_multigpu.py --config config.json
            
        Example with 4 GPUs:
            torchrun --nproc_per_node=4 esmretrain_multigpu.py --config config.json
            
        Example with specific GPUs (e.g., GPU 0 and 1):
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 esmretrain_multigpu.py --config config.json
    """
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
