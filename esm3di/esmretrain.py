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
from transformers import T5Tokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .ESM3di_model import ESM3DiModel, read_fasta, Seq3DiDataset, make_collate_fn, Lion, is_t5_model
from .losses import (
    FocalLoss, 
    CyclicalFocalLoss,
    PLDDTWeightedFocalLoss, 
    PLDDTWeightedCyclicalFocalLoss, 
    DEFAULT_PLDDT_BIN_WEIGHTS, 
    GammaSchedulerOnPlateau
)


# -----------------------------
# Validation
# -----------------------------

@torch.no_grad()
def validate(model, val_loader, loss_fn, device, use_amp=False, use_plddt=False, epoch=0):
    """
    Run validation on a held-out dataset with model in eval mode.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        loss_fn: Loss function to compute validation loss
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        use_plddt: Whether to track pLDDT-stratified accuracy
        epoch: Current epoch (needed for CyclicalFocalLoss)
    
    Returns:
        Tuple of (avg_loss, accuracy, total_tokens, plddt_accuracies)
        plddt_accuracies is a dict {threshold: accuracy} or empty dict if not tracking
    """
    model.eval()
    
    # Check if loss function is cyclical (needs epoch parameter)
    is_cyclical = isinstance(loss_fn, (CyclicalFocalLoss, PLDDTWeightedCyclicalFocalLoss))
    
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    # pLDDT-stratified accuracy tracking
    plddt_thresholds = [2, 4, 6, 8]  # bin thresholds (pLDDT >= 20, 40, 60, 80)
    correct_by_plddt = {t: 0 for t in plddt_thresholds}
    tokens_by_plddt = {t: 0 for t in plddt_thresholds}
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        plddt_bins = batch.get("plddt_bins")
        if plddt_bins is not None:
            plddt_bins = plddt_bins.to(device)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Flatten for loss computation
            if use_plddt and plddt_bins is not None:
                if is_cyclical:
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        epoch=epoch,
                        plddt_bins=plddt_bins.view(-1)
                    )
                else:
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        plddt_bins.view(-1)
                    )
            else:
                if is_cyclical:
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        epoch=epoch
                    )
                else:
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
        
        # Count non-masked tokens
        valid_mask = labels != -100
        token_count = valid_mask.sum().item()
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = ((preds == labels) & valid_mask).sum().item()
        
        # pLDDT-stratified accuracy
        if use_plddt and plddt_bins is not None:
            for thresh in plddt_thresholds:
                plddt_mask = (plddt_bins >= thresh) & valid_mask
                thresh_tokens = plddt_mask.sum().item()
                thresh_correct = ((preds == labels) & plddt_mask).sum().item()
                correct_by_plddt[thresh] += thresh_correct
                tokens_by_plddt[thresh] += thresh_tokens
        
        total_loss += loss.item() * max(token_count, 1)
        total_tokens += max(token_count, 1)
        total_correct += correct
    
    # Restore training mode
    model.train()
    
    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    
    # Compute pLDDT-stratified accuracies
    plddt_accuracies = {}
    if use_plddt:
        for thresh in plddt_thresholds:
            if tokens_by_plddt[thresh] > 0:
                plddt_accuracies[thresh] = correct_by_plddt[thresh] / tokens_by_plddt[thresh]
    
    return avg_loss, accuracy, total_tokens, plddt_accuracies

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
    
    # Multi-GPU setup
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        multi_gpu = True
    else:
        multi_gpu = False
    
    #set device
    torch.device(device)
    print(f"Using device: {device}")
    if multi_gpu:
        print(f"Multi-GPU training enabled with {torch.cuda.device_count()} GPUs")
    
    # Setup mixed precision training
    use_amp = args.mixed_precision and device != "cpu"
    if use_amp:
        print("Mixed precision training enabled (FP16)")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        if args.mixed_precision and device == "cpu":
            print("Warning: Mixed precision requested but not available on CPU")
    
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

    # 1) Data (with mask_label_chars and optional pLDDT bins)
    use_plddt = args.plddt_bins_fasta is not None
    dataset = Seq3DiDataset(
        args.aa_fasta,
        args.three_di_fasta,
        mask_label_chars=args.mask_label_chars,
        plddt_bins_fasta=args.plddt_bins_fasta,
    )
    print(f"Loaded {len(dataset)} sequences")
    print(f"3Di vocab ({len(dataset.label_vocab)}): {dataset.label_vocab}")
    if args.mask_label_chars:
        print(f"Masked 3Di chars (ignored in loss): "
                f"{list(set(args.mask_label_chars))}")
    if use_plddt:
        print(f"pLDDT bins loaded from: {args.plddt_bins_fasta}")
    
    
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
        cnn_dropout=args.cnn_dropout,
        use_transformer_head=args.use_transformer_head,
        transformer_head_dim=args.transformer_head_dim,
        transformer_head_layers=args.transformer_head_layers,
        transformer_head_dropout=args.transformer_head_dropout,
        transformer_head_num_heads=args.transformer_head_num_heads,
    )
    
    # Setup loss function (pLDDT-weighted, Focal Loss, or Cross-Entropy)
    if use_plddt:
        if args.use_cyclical_focal:
            # pLDDT-weighted Cyclical Focal Loss
            loss_fn = PLDDTWeightedCyclicalFocalLoss(
                gamma_pos=args.gamma_pos,
                gamma_neg=args.gamma_neg,
                gamma_hc=args.gamma_hc,
                eps=args.label_smoothing,
                epochs=args.epochs,  # Total epochs for eta schedule
                factor=args.cyclical_factor,  # 2.0 = cyclical, 1.0 = one-way
                min_bin=args.plddt_min_bin,
                weight_exponent=args.plddt_weight_exponent,
                ignore_index=-100
            )
            print(f"Using pLDDT-Weighted Cyclical Focal Loss:")
            print(f"  gamma_pos={args.gamma_pos}, gamma_neg={args.gamma_neg}, gamma_hc={args.gamma_hc}")
            print(f"  epochs={args.epochs}, factor={args.cyclical_factor}, label_smoothing={args.label_smoothing}")
            print(f"  min_bin={args.plddt_min_bin}, weight_exponent={args.plddt_weight_exponent}")
            print(f"  Bin weights: {DEFAULT_PLDDT_BIN_WEIGHTS}")
        else:
            # Regular pLDDT-weighted Focal Loss
            loss_fn = PLDDTWeightedFocalLoss(
                gamma=args.focal_gamma if args.use_focal_loss else 0.0,
                bin_weights=DEFAULT_PLDDT_BIN_WEIGHTS,
                min_bin=args.plddt_min_bin,
                weight_exponent=args.plddt_weight_exponent,
                ignore_index=-100
            )
            print(f"Using pLDDT-Weighted Focal Loss (gamma={loss_fn.gamma}, "
                  f"min_bin={args.plddt_min_bin}, weight_exponent={args.plddt_weight_exponent})")
            print(f"  Bin weights: {DEFAULT_PLDDT_BIN_WEIGHTS}")
    elif args.use_cyclical_focal:
        # Regular Cyclical Focal Loss (no pLDDT)
        loss_fn = CyclicalFocalLoss(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            gamma_hc=args.gamma_hc,
            eps=args.label_smoothing,
            epochs=args.epochs,
            factor=args.cyclical_factor,
            ignore_index=-100
        )
        print(f"Using Cyclical Focal Loss:")
        print(f"  gamma_pos={args.gamma_pos}, gamma_neg={args.gamma_neg}, gamma_hc={args.gamma_hc}")
        print(f"  epochs={args.epochs}, factor={args.cyclical_factor}, label_smoothing={args.label_smoothing}")
    elif args.use_focal_loss:
        loss_fn = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            ignore_index=-100
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        print("Using Cross-Entropy Loss")
    
    # Setup gamma scheduler for focal loss (increases gamma on accuracy plateau)
    # Note: Gamma scheduler only works with regular Focal Loss, not Cyclical Focal Loss
    gamma_scheduler = None
    if args.gamma_scheduler and args.use_cyclical_focal:
        print("Warning: --gamma-scheduler is not compatible with --use-cyclical-focal, ignoring")
    elif args.gamma_scheduler and (args.use_focal_loss or use_plddt):
        gamma_scheduler = GammaSchedulerOnPlateau(
            loss_fn=loss_fn,
            mode='max',  # Monitor accuracy (want it to increase)
            factor=args.gamma_increase_factor,
            patience=args.gamma_patience,
            threshold=args.gamma_threshold,
            max_gamma=args.gamma_max,
            verbose=True
        )
        print(f"Gamma scheduler enabled (patience={args.gamma_patience}, "
              f"increase={args.gamma_increase_factor}, max_gamma={args.gamma_max})")
    elif args.gamma_scheduler and not args.use_focal_loss and not use_plddt:
        print("Warning: --gamma-scheduler requires --use-focal-loss or --plddt-bins-fasta, ignoring")

    # Detect if using T5-based model and initialize appropriate tokenizer
    is_t5 = is_t5_model(args.hf_model)
    
    if is_t5:
        print("\nUsing T5Tokenizer for T5-based model")
        # Try AutoTokenizer first (works for Ankh), fall back to T5Tokenizer (ProtT5)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.hf_model,
                trust_remote_code=True
            )
        except Exception:
            tokenizer = T5Tokenizer.from_pretrained(
                args.hf_model,
                legacy=True,
                trust_remote_code=True
            )
    elif "esm2" in args.hf_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            trust_remote_code=True  # Required for ESM++ custom code
        )
    else:
        tokenizer = esm_model.base_model.tokenizer

    model = esm_model.get_model()
    model.to(device)
    
    # Move loss function to device (for any tensor parameters like alpha)
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)
    
    # Wrap with DataParallel for multi-GPU training
    if multi_gpu:
        model = torch.nn.DataParallel(model)
        print(f"✓ Model with LoRA loaded and wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")
    else:
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
        include_plddt=use_plddt,
        is_t5=is_t5,
    )
    # Determine steps per epoch based on samples_per_epoch or full dataset
    if args.samples_per_epoch is not None:
        steps_per_epoch_target = args.samples_per_epoch // args.batch_size
        if steps_per_epoch_target < 1:
            steps_per_epoch_target = 1
        print(f"Using custom epoch size: {args.samples_per_epoch} samples "
              f"({steps_per_epoch_target} batches per epoch)")
        use_custom_epoch = True
    else:
        steps_per_epoch_target = None
        use_custom_epoch = False
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Setup validation dataloader if validation data provided
    val_loader = None
    if args.val_aa_fasta and args.val_three_di_fasta:
        print(f"\nSetting up validation data...")
        val_dataset = Seq3DiDataset(
            args.val_aa_fasta,
            args.val_three_di_fasta,
            mask_label_chars=args.mask_label_chars,
            plddt_bins_fasta=args.val_plddt_bins_fasta,
        )
        # Use same collate_fn (same tokenizer and label vocab)
        # But we need to create a new one with the training dataset's char2idx
        val_use_plddt = use_plddt and args.val_plddt_bins_fasta is not None
        val_collate_fn = make_collate_fn(
            tokenizer,
            dataset.char2idx,  # Use training vocab to ensure consistency
            mask_label_chars=args.mask_label_chars,
            include_plddt=val_use_plddt,
            is_t5=is_t5,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_collate_fn,
        )
        print(f"✓ Validation set: {len(val_dataset)} sequences")
        if val_use_plddt:
            print(f"  pLDDT bins: {args.val_plddt_bins_fasta}")
    elif args.val_aa_fasta or args.val_three_di_fasta:
        print("Warning: Both --val-aa-fasta and --val-three-di-fasta required for validation")
    
    # 7) Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'lion':
        optimizer = Lion(
            trainable_params,
            lr=args.lr,
            betas=(args.lion_beta1, args.lion_beta2),
            weight_decay=args.weight_decay,
        )
        print(f"Using Lion optimizer (lr={args.lr:.2e}, betas=({args.lion_beta1}, {args.lion_beta2}), wd={args.weight_decay})")
    else:  # adamw
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"Using AdamW optimizer (lr={args.lr:.2e}, wd={args.weight_decay})")

    # 8) Learning rate scheduler setup
    # Calculate total training steps (accounting for gradient accumulation)
    if use_custom_epoch:
        steps_per_epoch = steps_per_epoch_target // args.gradient_accumulation_steps
        if steps_per_epoch_target % args.gradient_accumulation_steps != 0:
            steps_per_epoch += 1
    else:
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
    plateau_scheduler = None  # Separate variable for plateau scheduler (epoch-based)
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
    elif args.scheduler_type == 'plateau':
        # ReduceLROnPlateau is epoch-based, not step-based
        scheduler = None  # No per-step scheduler
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            min_lr=args.plateau_min_lr,
        )
        scheduler_name = (f"ReduceLROnPlateau (patience={args.plateau_patience}, "
                          f"factor={args.plateau_factor}, min_lr={args.plateau_min_lr:.2e})")
    else:
        # No scheduler
        scheduler = None
        scheduler_name = "None (constant LR)"
    
    # Load checkpoint if resuming training
    start_epoch = 1
    global_step = 0
    accumulation_step = 0
    
    if args.resume_from_checkpoint:
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        # Load model state (handle DataParallel wrapper)
        model_state_dict = checkpoint["model_state_dict"]
        
        # Check if state dict keys have "module." prefix (saved with DataParallel)
        saved_with_dp = any(k.startswith("module.") for k in model_state_dict.keys())
        
        if multi_gpu and not saved_with_dp:
            # Loading into DataParallel model, but checkpoint was saved without it
            # Add "module." prefix to all keys
            model_state_dict = {"module." + k: v for k, v in model_state_dict.items()}
        elif not multi_gpu and saved_with_dp:
            # Loading into non-DataParallel model, but checkpoint was saved with it
            # Remove "module." prefix from all keys
            model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
        
        model.load_state_dict(model_state_dict)
        print("✓ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✓ Optimizer state loaded")
        
        # Load scheduler state if available
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✓ Scheduler state loaded")
        
        # Load plateau scheduler state if available
        if plateau_scheduler is not None and checkpoint.get("plateau_scheduler_state_dict") is not None:
            plateau_scheduler.load_state_dict(checkpoint["plateau_scheduler_state_dict"])
            print("✓ Plateau scheduler state loaded")
        
        # Load gamma scheduler state if available
        if gamma_scheduler is not None and checkpoint.get("gamma_scheduler_state_dict") is not None:
            gamma_scheduler.load_state_dict(checkpoint["gamma_scheduler_state_dict"])
            print(f"✓ Gamma scheduler state loaded (gamma={gamma_scheduler.gamma:.2f})")
        
        # Load AMP scaler state if available
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("✓ AMP scaler state loaded")
        
        # Resume from next epoch
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        
        print(f"\nResuming from:")
        print(f"  Epoch: {checkpoint.get('epoch', 0)} (will start at {start_epoch})")
        print(f"  Global step: {global_step}")
        print(f"  Previous loss: {checkpoint.get('loss', 'N/A'):.4f}" if isinstance(checkpoint.get('loss'), (int, float)) else f"  Previous loss: N/A")
        print(f"{'='*60}\n")
    
    # 8) Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"Dataset size: {len(dataset)} sequences")
    if use_custom_epoch:
        print(f"Epoch definition: {args.samples_per_epoch} samples ({steps_per_epoch_target} batches)")
        print(f"  Full dataset passes per {args.epochs} epochs: {(args.samples_per_epoch * args.epochs) / len(dataset):.2f}")
    else:
        print(f"Epoch definition: Full dataset ({len(dataset)} samples)")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate scheduling: {scheduler_name}")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"  Initial LR: {args.lr:.2e}")
    model.train()
    
    # Check if loss function is cyclical (needs epoch parameter)
    is_cyclical = isinstance(loss_fn, (CyclicalFocalLoss, PLDDTWeightedCyclicalFocalLoss))
    
    # For custom epoch sizes, create an infinite iterator over the dataset
    if use_custom_epoch:
        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch
        data_iterator = iter(infinite_loader())
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        if use_custom_epoch:
            print(f"(Custom epoch: {args.samples_per_epoch} samples, {steps_per_epoch_target} batches)")
        print(f"{'='*60}")
        
        running_loss = 0.0
        running_tokens = 0
        running_correct = 0
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_correct = 0
        
        # pLDDT-stratified accuracy tracking (bins 2, 4, 6, 8 = pLDDT >= 20, 40, 60, 80)
        plddt_thresholds = [2, 4, 6, 8]  # bin thresholds
        epoch_correct_by_plddt = {t: 0 for t in plddt_thresholds}
        epoch_tokens_by_plddt = {t: 0 for t in plddt_thresholds}

        # Create progress bar for batches
        if use_custom_epoch:
            # Use custom number of steps per epoch
            progress_bar = tqdm(range(steps_per_epoch_target), desc=f"Epoch {epoch}", unit="batch")
            batch_source = ((i + 1, next(data_iterator)) for i in progress_bar)
        else:
            progress_bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
            batch_source = enumerate(progress_bar, start=1)

        for step, batch in batch_source:
            # Update progress bar reference for custom epoch mode
            if use_custom_epoch:
                current_progress_bar = progress_bar
            else:
                current_progress_bar = progress_bar
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Extract labels and optional pLDDT bins for external loss computation
            labels = batch.pop("labels")
            plddt_bins = batch.pop("plddt_bins", None)
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    # Compute loss externally (supports pLDDT-weighted, Focal Loss, or CE)
                    logits = outputs.logits
                    if use_plddt and plddt_bins is not None:
                        if is_cyclical:
                            loss = loss_fn(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                epoch=epoch,
                                plddt_bins=plddt_bins.view(-1)
                            )
                        else:
                            loss = loss_fn(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                plddt_bins.view(-1)
                            )
                    else:
                        if is_cyclical:
                            loss = loss_fn(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                epoch=epoch
                            )
                        else:
                            loss = loss_fn(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1)
                            )
            else:
                outputs = model(**batch)
                # Compute loss externally (supports pLDDT-weighted, Focal Loss, or CE)
                logits = outputs.logits
                if use_plddt and plddt_bins is not None:
                    if is_cyclical:
                        loss = loss_fn(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            epoch=epoch,
                            plddt_bins=plddt_bins.view(-1)
                        )
                    else:
                        loss = loss_fn(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            plddt_bins.view(-1)
                        )
                else:
                    if is_cyclical:
                        loss = loss_fn(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            epoch=epoch
                        )
                    else:
                        loss = loss_fn(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1)
                        )
            
            # Restore labels to batch for metrics
            batch["labels"] = labels

            # DataParallel returns loss per GPU, need to take mean
            if multi_gpu and loss.dim() > 0:
                loss = loss.mean()

            token_count = (labels != -100).sum().item()

            # Calculate accuracy
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                valid_mask = labels != -100
                correct = ((preds == labels) & valid_mask).sum().item()
                batch_accuracy = correct / max(token_count, 1)
                
                # pLDDT-stratified accuracy (only when plddt_bins available)
                if use_plddt and plddt_bins is not None:
                    for thresh in plddt_thresholds:
                        # Positions with pLDDT bin >= threshold
                        plddt_mask = (plddt_bins >= thresh) & valid_mask
                        thresh_tokens = plddt_mask.sum().item()
                        thresh_correct = ((preds == labels) & plddt_mask).sum().item()
                        epoch_correct_by_plddt[thresh] += thresh_correct
                        epoch_tokens_by_plddt[thresh] += thresh_tokens

            # Scale loss by accumulation steps for proper gradient averaging
            scaled_loss = loss / args.gradient_accumulation_steps
            
            # Mixed precision backward pass
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            accumulation_step += 1

            # Calculate per-residue loss for this batch
            per_residue_loss = loss.item() / max(token_count, 1)

            running_loss += loss.item() * max(token_count, 1)
            running_tokens += max(token_count, 1)
            running_correct += correct
            epoch_loss += loss.item() * max(token_count, 1)
            epoch_tokens += max(token_count, 1)
            epoch_correct += correct

            # Perform optimization step after accumulating enough gradients
            if accumulation_step % args.gradient_accumulation_steps == 0:
                global_step += 1
                
                # Mixed precision optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Step the learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Log current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Training/learning_rate', current_lr, global_step)
                
                # Update progress bar with current loss, accuracy, and LR
                avg_loss = running_loss / max(running_tokens, 1)
                running_accuracy = running_correct / max(running_tokens, 1)
                current_progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{running_accuracy:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step
                })
                
                # Log to TensorBoard
                writer.add_scalar('Loss/train_step', avg_loss, global_step)
                writer.add_scalar('Accuracy/train_step', running_accuracy, global_step)
                
                if global_step % args.log_every == 0:
                    # Log running average to TensorBoard
                    writer.add_scalar('Loss/train_running_avg', avg_loss, global_step)
                    writer.add_scalar('Accuracy/train_running_avg', running_accuracy, global_step)
                    writer.add_scalar('Training/tokens_processed', running_tokens, global_step)
                    running_loss = 0.0
                    running_tokens = 0
                    running_correct = 0
            else:
                # Just update progress bar during accumulation
                avg_loss = running_loss / max(running_tokens, 1)
                running_accuracy = running_correct / max(running_tokens, 1)
                current_progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{running_accuracy:.4f}",
                    "accum": f"{accumulation_step % args.gradient_accumulation_steps}/{args.gradient_accumulation_steps}"
                })
            
            # Always log per-batch loss and accuracy (before accumulation)
            writer.add_scalar('Loss/train_batch', loss.item(), step + (epoch - 1) * len(loader))
            writer.add_scalar('Loss/train_per_residue_batch', per_residue_loss, step + (epoch - 1) * len(loader))
            writer.add_scalar('Accuracy/train_batch', batch_accuracy, step + (epoch - 1) * len(loader))

        # Log epoch-level metrics to TensorBoard
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        epoch_accuracy = epoch_correct / max(epoch_tokens, 1)
        writer.add_scalar('Loss/train_epoch', epoch_avg_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', epoch_accuracy, epoch)
        writer.add_scalar('Training/epoch_tokens', epoch_tokens, epoch)
        
        # Log pLDDT-stratified accuracy if using pLDDT weighting
        if use_plddt:
            plddt_accuracies = {}
            for thresh in plddt_thresholds:
                thresh_tokens = epoch_tokens_by_plddt[thresh]
                thresh_correct = epoch_correct_by_plddt[thresh]
                if thresh_tokens > 0:
                    thresh_acc = thresh_correct / thresh_tokens
                    plddt_accuracies[thresh] = thresh_acc
                    # Log to TensorBoard
                    plddt_val = thresh * 10  # Convert bin to pLDDT value (2 -> 20, etc.)
                    writer.add_scalar(f'Accuracy/plddt_ge{plddt_val}', thresh_acc, epoch)
        
        # Step plateau scheduler (epoch-based) if using it
        if plateau_scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            plateau_scheduler.step(epoch_avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"\n*** Plateau detected! Reducing LR: {old_lr:.2e} -> {new_lr:.2e} ***")
                writer.add_scalar('Training/lr_reduction_event', 1, epoch)
        
        # Step gamma scheduler (epoch-based) if using it - based on accuracy
        if gamma_scheduler is not None:
            old_gamma = gamma_scheduler.gamma
            gamma_increased = gamma_scheduler.step(epoch_accuracy, epoch)
            if gamma_increased:
                writer.add_scalar('Training/gamma_increase_event', 1, epoch)
            writer.add_scalar('Training/focal_gamma', gamma_scheduler.gamma, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} Average Loss (per residue): {epoch_avg_loss:.4f}")
        print(f"Epoch {epoch} Accuracy: {epoch_accuracy:.4f} ({epoch_correct}/{epoch_tokens} residues)")
        
        # Print pLDDT-stratified accuracy
        if use_plddt and plddt_accuracies:
            plddt_str = " | ".join([f"≥{t*10}: {plddt_accuracies.get(t, 0):.4f}" for t in plddt_thresholds if t in plddt_accuracies])
            print(f"Epoch {epoch} Accuracy by pLDDT: {plddt_str}")
        
        print(f"Current Learning Rate: {current_lr:.2e}")
        if gamma_scheduler is not None:
            print(f"Current Focal Gamma: {gamma_scheduler.gamma:.2f}")
        
        # Run validation if validation data is provided
        if val_loader is not None:
            print(f"\nRunning validation...")
            val_loss, val_accuracy, val_tokens, val_plddt_accuracies = validate(
                model, val_loader, loss_fn, device, use_amp=use_amp, use_plddt=use_plddt, epoch=epoch
            )
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f} ({int(val_accuracy * val_tokens)}/{val_tokens} residues)")
            
            # Print pLDDT-stratified validation accuracy
            if use_plddt and val_plddt_accuracies:
                val_plddt_str = " | ".join([f"≥{t*10}: {val_plddt_accuracies.get(t, 0):.4f}" for t in plddt_thresholds if t in val_plddt_accuracies])
                print(f"Validation Accuracy by pLDDT: {val_plddt_str}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
            writer.add_scalar('Validation/tokens', val_tokens, epoch)
            
            # Log pLDDT-stratified validation accuracy
            if use_plddt and val_plddt_accuracies:
                for thresh, acc in val_plddt_accuracies.items():
                    plddt_val = thresh * 10
                    writer.add_scalar(f'Accuracy/val_plddt_ge{plddt_val}', acc, epoch)
            
            # Log train vs val comparison
            writer.add_scalars('Loss/train_vs_val', {
                'train': epoch_avg_loss,
                'val': val_loss
            }, epoch)
            writer.add_scalars('Accuracy/train_vs_val', {
                'train': epoch_accuracy,
                'val': val_accuracy
            }, epoch)
        
        # Save checkpoint each epoch
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
        
        # Get model state dict (handle DataParallel wrapper)
        if multi_gpu:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        torch.save(
            {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "plateau_scheduler_state_dict": plateau_scheduler.state_dict() if plateau_scheduler is not None else None,
            "gamma_scheduler_state_dict": gamma_scheduler.state_dict() if gamma_scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
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
        
        torch.save( model if not multi_gpu else model.module, ckpt_path.replace(".pt", "_model.pt") )

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
            target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]

    esm_model = ESM3DiModel(
        hf_model_name=args["hf_model"],
        num_labels=len(label_vocab),
        lora_r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        target_modules=target_modules,
        use_cnn_head=args.get("use_cnn_head", False),
        cnn_num_layers=args.get("cnn_num_layers", 2),
        cnn_kernel_size=args.get("cnn_kernel_size", 3),
        cnn_dropout=args.get("cnn_dropout", 0.1),
        use_transformer_head=args.get("use_transformer_head", False),
        transformer_head_dim=args.get("transformer_head_dim", 256),
        transformer_head_layers=args.get("transformer_head_layers", 2),
        transformer_head_dropout=args.get("transformer_head_dropout", 0.1),
        transformer_head_num_heads=args.get("transformer_head_num_heads", None),
    )

    model = esm_model.get_model()
    
    # Handle potential DataParallel prefix in checkpoint
    model_state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in model_state_dict.keys()):
        model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    if "esm2" in args["hf_model"].lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args["hf_model"],
            trust_remote_code=True,
        )
    else:
        tokenizer = esm_model.base_model.tokenizer

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
    
    # Validation data (optional)
    p.add_argument("--val-aa-fasta", type=str, default=None,
                   help="FASTA with validation amino-acid sequences (optional). "
                        "If provided along with --val-three-di-fasta, validation "
                        "will be run at the end of each epoch with model in eval mode.")
    p.add_argument("--val-three-di-fasta", type=str, default=None,
                   help="FASTA with validation 3Di sequences (optional). "
                        "Must have matching sequences to --val-aa-fasta.")

    # Model
    p.add_argument("--hf-model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="HuggingFace model name. "
                        "ESM-2 options: facebook/esm2_t12_35M_UR50D (35M params), "
                        "facebook/esm2_t30_150M_UR50D (150M), facebook/esm2_t33_650M_UR50D (650M). "
                        "ESM++ options: Synthyra/ESMplusplus_small (333M params), "
                        "Synthyra/ESMplusplus_large (575M params).")

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

    # Transformer Classification Head
    p.add_argument("--use-transformer-head", action="store_true",
                   help="Use a small TransformerEncoder classification head instead "
                        "of the default linear classifier.")
    p.add_argument("--transformer-head-dim", type=int, default=256,
                   help="Hidden dimension of transformer head "
                        "(only used if --use-transformer-head is set)")
    p.add_argument("--transformer-head-layers", type=int, default=2,
                   help="Number of TransformerEncoder layers in head "
                        "(only used if --use-transformer-head is set)")
    p.add_argument("--transformer-head-dropout", type=float, default=0.1,
                   help="Dropout rate for transformer head "
                        "(only used if --use-transformer-head is set)")
    p.add_argument("--transformer-head-num-heads", type=int, default=None,
                   help="Attention head count for transformer head. If omitted, "
                        "auto-selects a divisor of --transformer-head-dim.")

    # Focal Loss
    p.add_argument("--use-focal-loss", action="store_true",
                   help="Use Focal Loss instead of Cross-Entropy. "
                        "Focal Loss down-weights easy examples, focusing training on hard ones. "
                        "Useful when accuracy is already high but you want to improve on difficult tokens.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (focusing) parameter. Higher values "
                        "focus more on hard examples. Common values: 0.5, 1.0, 2.0, 5.0. "
                        "(only used if --use-focal-loss is set)")
    p.add_argument("--focal-alpha", type=float, default=None,
                   help="Optional class weight for focal loss. "
                        "(only used if --use-focal-loss is set)")
    
    # Cyclical Focal Loss
    p.add_argument("--use-cyclical-focal", action="store_true",
                   help="Use Cyclical Focal Loss instead of regular Focal Loss. "
                        "CFL dynamically adjusts focus during training with epoch-dependent weighting. "
                        "Can be combined with --plddt-bins-fasta for pLDDT-weighted cyclical loss.")
    p.add_argument("--gamma-pos", type=float, default=0.0,
                   help="CFL: Focal exponent for positive (correct) class. "
                        "0.0 = no down-weighting of correct predictions (default: 0.0)")
    p.add_argument("--gamma-neg", type=float, default=4.0,
                   help="CFL: Focal exponent for negative (incorrect) classes. "
                        "Higher values suppress easy negatives more strongly (default: 4.0)")
    p.add_argument("--gamma-hc", type=float, default=0.0,
                   help="CFL: Hard-class positive weighting exponent. "
                        "Increases loss for uncertain correct predictions. "
                        "Use 0.5-2.0 for highly overlapping classes (default: 0.0)")
    p.add_argument("--cyclical-factor", type=float, default=2.0,
                   help="CFL: Scheduling factor. 2.0 = full cyclical behavior, "
                        "1.0 = one-way modified schedule (default: 2.0)")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="CFL: Label smoothing epsilon for regularization (default: 0.1)")
    
    # pLDDT-weighted loss
    p.add_argument("--plddt-bins-fasta", type=str, default=None,
                   help="FASTA with pLDDT bin values (0-9 per position). "
                        "If provided, uses PLDDTWeightedFocalLoss which weights loss "
                        "by pLDDT confidence. Generate with build_trainingset.py --output-plddt-bins.")
    p.add_argument("--val-plddt-bins-fasta", type=str, default=None,
                   help="FASTA with validation pLDDT bin values (optional).")
    p.add_argument("--plddt-min-bin", type=int, default=5,
                   help="Minimum pLDDT bin (0-9) to include in loss. Bins below this "
                        "get zero weight. Default 5 corresponds to pLDDT >= 50.")
    p.add_argument("--plddt-weight-exponent", type=float, default=1.0,
                   help="Exponent applied to pLDDT weights. Higher values sharpen "
                        "contrast between low and high confidence. Values >1 increase "
                        "focus on high-confidence regions.")
    
    # Gamma scheduler (for focal loss)
    p.add_argument("--gamma-scheduler", action="store_true",
                   help="Enable gamma scheduler to increase focal loss gamma on accuracy plateau. "
                        "Helps focus on hard examples when the model stops improving.")
    p.add_argument("--gamma-increase-factor", type=float, default=0.5,
                   help="Amount to add to gamma when plateau detected (default: 0.5)")
    p.add_argument("--gamma-patience", type=int, default=2,
                   help="Number of epochs with no accuracy improvement before increasing gamma")
    p.add_argument("--gamma-max", type=float, default=5.0,
                   help="Maximum gamma value (prevents over-focusing on hard examples)")
    p.add_argument("--gamma-threshold", type=float, default=1e-3,
                   help="Minimum accuracy improvement to be considered significant")

    # Training
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size per GPU")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Number of gradient accumulation steps. Effective batch size = "
                        "batch_size * gradient_accumulation_steps. Use to simulate larger "
                        "batch sizes when GPU memory is limited.")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--samples-per-epoch", type=int, default=None,
                   help="Number of samples per epoch. If set, an 'epoch' is defined as this many "
                        "samples rather than a full pass through the dataset. Useful for large "
                        "datasets where a full epoch takes too long. The dataset will cycle "
                        "continuously across epochs.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate. Note: if using Lion optimizer, use 3-10x lower "
                        "(e.g., 1e-5 to 3e-5) than AdamW.")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Weight decay for optimizer. For Lion, use higher values (0.1-0.3).")
    p.add_argument("--optimizer", type=str, default='adamw',
                   choices=['adamw', 'lion'],
                   help="Optimizer to use. 'adamw' is the default and well-tested. "
                        "'lion' uses sign-based updates with 50%% less memory - "
                        "requires lower LR (3-10x) and higher weight decay (0.1-0.3).")
    p.add_argument("--lion-beta1", type=float, default=0.9,
                   help="Lion beta1 parameter for update interpolation (only used with --optimizer=lion)")
    p.add_argument("--lion-beta2", type=float, default=0.99,
                   help="Lion beta2 parameter for momentum decay (only used with --optimizer=lion)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of DataLoader workers")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log training progress every N optimization steps")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt file) to resume training from. ")
    
    # Learning rate scheduler
    p.add_argument("--scheduler-type", type=str, default='cosine',
                   choices=['cosine', 'linear', 'constant', 'plateau', 'none'],
                   help="Learning rate scheduler type. 'cosine' uses cosine decay after warmup, "
                        "'linear' uses linear decay, 'constant' keeps LR constant after warmup, "
                        "'plateau' reduces LR when loss plateaus, "
                        "'none' uses no scheduler (constant LR throughout)")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Number of warmup steps. If not specified, uses warmup-ratio")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="Warmup ratio (fraction of total steps). Default: 0.1 (10%%)")
    
    # Plateau scheduler options
    p.add_argument("--plateau-patience", type=int, default=2,
                   help="Number of epochs with no improvement after which LR will be reduced "
                        "(only used with --scheduler-type=plateau)")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                   help="Factor by which LR will be reduced (new_lr = lr * factor). "
                        "(only used with --scheduler-type=plateau)")
    p.add_argument("--plateau-threshold", type=float, default=1e-4,
                   help="Threshold for measuring the new optimum. Only focus on significant changes. "
                        "(only used with --scheduler-type=plateau)")
    p.add_argument("--plateau-min-lr", type=float, default=1e-7,
                   help="Minimum learning rate. LR will not be reduced below this value. "
                        "(only used with --scheduler-type=plateau)")
    
    # Device and performance
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu'). "
                        "If not specified, uses CUDA if available.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available "
                        "(ignored if --device is specified)")
    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable mixed precision training (FP16) for faster training "
                        "and reduced memory usage. Only works with CUDA.")
    p.add_argument("--multi-gpu", action="store_true",
                   help="Enable multi-GPU training using DataParallel. "
                        "Automatically uses all available GPUs.")
    
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

    if args.use_cnn_head and args.use_transformer_head:
        p.error("--use-cnn-head and --use-transformer-head are mutually exclusive")
    
    return args


def main():
    """Main entry point for training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
