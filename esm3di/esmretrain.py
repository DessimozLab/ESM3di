#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import inspect
import json
import math
import os
import socket
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

# Set CUDA memory allocator config before importing torch
# This helps reduce memory fragmentation on large models
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, T5Tokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .ESM3di_model import ESM3DiModel, read_fasta, Seq3DiDataset, make_collate_fn, Lion
from .T5Model import T5ProteinModel, is_t5_model
from .model_outputs import PLDDT_BIN_VOCAB
from .losses import (
    FocalLoss, 
    CyclicalFocalLoss,
    PLDDTWeightedFocalLoss, 
    PLDDTWeightedCyclicalFocalLoss, 
    DEFAULT_PLDDT_BIN_WEIGHTS, 
    GammaSchedulerOnPlateau
)


def _load_esm3di_model_class(use_iterative_backbone_head: bool):
    if not use_iterative_backbone_head:
        return ESM3DiModel

    copy_path = Path(__file__).parent / "ESM3di_model copy.py"
    if not copy_path.exists():
        print("Warning: iterative backbone mode requested but ESM3di_model copy.py not found; using default ESM3DiModel")
        return ESM3DiModel

    module_name = "esm3di.ESM3di_model copy"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"Warning: could not import {module_name} ({exc}); using default ESM3DiModel")
        return ESM3DiModel
    if not hasattr(module, "ESM3DiModel"):
        print("Warning: ESM3di_model copy.py has no ESM3DiModel class; using default ESM3DiModel")
        return ESM3DiModel

    print("Using ESM3DiModel from ESM3di_model copy.py")
    return module.ESM3DiModel


def _filtered_model_kwargs(model_cls, kwargs):
    try:
        valid_keys = set(inspect.signature(model_cls.__init__).parameters.keys())
        valid_keys.discard("self")
        return {k: v for k, v in kwargs.items() if k in valid_keys}
    except Exception:
        return kwargs


def _compute_aux_track_loss(aux_logits, aux_bins, labels, aux_loss_weight: float = 1.0):
    """
    Compute summed cross-entropy loss for all auxiliary categorical tracks.

    Args:
        aux_logits: Dict[str, Tensor[B, T, n_bins]] from model output.
        aux_bins:   Dict[str, Tensor[B, T]] from collate; -100 = ignore.
        labels:     Primary labels tensor (used only to gate valid positions).
        aux_loss_weight: Scalar weight applied to the total aux loss.

    Returns:
        (total_aux_loss, per_track_losses_dict) or (None, {}) if nothing to compute.
    """
    if not aux_logits or not aux_bins:
        return None, {}

    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    per_track = {}
    total = None
    for track_name, logits_t in aux_logits.items():
        bins_t = aux_bins.get(track_name)
        if bins_t is None:
            continue
        # logits_t: [B, T, n_bins], bins_t: [B, T]
        loss_t = ce(logits_t.view(-1, logits_t.size(-1)), bins_t.view(-1))
        per_track[track_name] = loss_t
        total = loss_t if total is None else total + loss_t

    if total is None:
        return None, {}
    return aux_loss_weight * total, per_track


def _compute_aux_track_metrics(aux_logits, aux_bins):
    """
    Compute per-track accuracy for auxiliary categorical tracks.

    Returns:
        Dict[track_name → {"tokens": int, "correct": int}]
    """
    if not aux_logits or not aux_bins:
        return {}

    metrics = {}
    for track_name, logits_t in aux_logits.items():
        bins_t = aux_bins.get(track_name)
        if bins_t is None:
            continue
        preds = logits_t.argmax(dim=-1)          # [B, T]
        valid  = bins_t != -100                   # [B, T]
        tokens = valid.sum().item()
        correct = ((preds == bins_t) & valid).sum().item()
        metrics[track_name] = {"tokens": tokens, "correct": correct}
    return metrics


def _compute_primary_loss(loss_fn, logits, labels, use_plddt=False, plddt_bins=None, is_cyclical=False, epoch=0):
    if use_plddt and plddt_bins is not None:
        if is_cyclical:
            return loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                epoch=epoch,
                plddt_bins=plddt_bins.view(-1)
            )
        return loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            plddt_bins.view(-1)
        )

    if is_cyclical:
        return loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            epoch=epoch
        )
    return loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )


def _plddt_bins_to_scores(plddt_bins):
    return torch.clamp(plddt_bins.float() * 10.0 + 5.0, min=0.0, max=100.0)


def _compute_plddt_aux_loss(plddt_logits, plddt_bins, labels, mode="classification", regression_loss="huber"):
    if plddt_logits is None or plddt_bins is None:
        return None

    valid_mask = (labels != -100) & (plddt_bins >= 0)
    if valid_mask.sum().item() == 0:
        return None

    if mode == "classification":
        aux_labels = plddt_bins.clone()
        aux_labels[~valid_mask] = -100
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(plddt_logits.view(-1, plddt_logits.size(-1)), aux_labels.view(-1))

    pred_scores = plddt_logits.squeeze(-1)
    target_scores = _plddt_bins_to_scores(plddt_bins)
    pred_valid = pred_scores[valid_mask]
    target_valid = target_scores[valid_mask]

    if regression_loss == "mse":
        return torch.nn.functional.mse_loss(pred_valid, target_valid)
    return torch.nn.functional.smooth_l1_loss(pred_valid, target_valid)


def _compute_aux_metrics(plddt_logits, plddt_bins, valid_mask, mode="classification"):
    if plddt_logits is None or plddt_bins is None:
        return {"tokens": 0}

    aux_valid_mask = valid_mask & (plddt_bins >= 0)
    aux_tokens = aux_valid_mask.sum().item()
    if aux_tokens == 0:
        return {"tokens": 0}

    if mode == "classification":
        aux_preds = plddt_logits.argmax(dim=-1)
        aux_correct = ((aux_preds == plddt_bins) & aux_valid_mask).sum().item()
        return {
            "tokens": aux_tokens,
            "correct": aux_correct,
        }

    pred_scores = plddt_logits.squeeze(-1)
    target_scores = _plddt_bins_to_scores(plddt_bins)
    pred_valid = pred_scores[aux_valid_mask]
    target_valid = target_scores[aux_valid_mask]
    abs_error_sum = torch.abs(pred_valid - target_valid).sum().item()
    sq_error_sum = torch.square(pred_valid - target_valid).sum().item()
    pred_bins = torch.clamp((pred_valid / 10.0).floor().long(), min=0, max=9)
    target_bins = torch.clamp((target_valid / 10.0).floor().long(), min=0, max=9)
    bin_correct = (pred_bins == target_bins).sum().item()
    return {
        "tokens": aux_tokens,
        "abs_error_sum": abs_error_sum,
        "sq_error_sum": sq_error_sum,
        "bin_correct": bin_correct,
    }


# -----------------------------
# Validation
# -----------------------------

@torch.no_grad()
def validate(
    model,
    val_loader,
    loss_fn,
    device,
    use_amp=False,
    use_plddt=False,
    use_plddt_prediction_head=False,
    plddt_prediction_mode="classification",
    plddt_regression_loss="huber",
    plddt_loss_weight=1.0,
    aux_loss_weight=0.5,
    epoch=0,
    track_iterations=False,
):
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
        If track_iterations=False:
            Tuple of (avg_loss, accuracy, total_tokens, plddt_accuracies)
        If track_iterations=True:
            Tuple of (avg_loss, accuracy, total_tokens, plddt_accuracies, avg_n_iterations)
        plddt_accuracies is a dict {threshold: accuracy} or empty dict if not tracking
    """
    model.eval()
    
    # Check if loss function is cyclical (needs epoch parameter)
    is_cyclical = isinstance(loss_fn, (CyclicalFocalLoss, PLDDTWeightedCyclicalFocalLoss))
    
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_aux_correct = 0
    total_aux_tokens = 0
    total_aux_abs_error_sum = 0.0
    total_aux_sq_error_sum = 0.0
    total_aux_bin_correct = 0
    iter_weighted_sum = 0.0
    iter_weighted_tokens = 0
    
    # pLDDT-stratified accuracy tracking
    plddt_thresholds = [2, 4, 6, 8]  # bin thresholds (pLDDT >= 20, 40, 60, 80)
    correct_by_plddt = {t: 0 for t in plddt_thresholds}
    tokens_by_plddt = {t: 0 for t in plddt_thresholds}

    # Per-track aux accumulators (bend, torsion, ...)
    val_track_correct: dict = {}
    val_track_tokens: dict = {}
    val_track_loss_sum: dict = {}
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        plddt_bins = batch.get("plddt_bins")
        if plddt_bins is not None:
            plddt_bins = plddt_bins.to(device)
        # Aux track bins (dict of tensors or absent)
        aux_bins_raw = batch.get("aux_bins")
        if aux_bins_raw:
            batch["aux_bins"] = {k: v.to(device) for k, v in aux_bins_raw.items()}
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if track_iterations:
                model_kwargs["return_halt_info"] = True
            outputs = model(**model_kwargs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            plddt_logits = getattr(outputs, 'plddt_logits', None)
            model_aux_logits = getattr(outputs, 'aux_logits', None) or {}

            loss = _compute_primary_loss(
                loss_fn,
                logits,
                labels,
                use_plddt=use_plddt,
                plddt_bins=plddt_bins,
                is_cyclical=is_cyclical,
                epoch=epoch,
            )
            aux_loss = None
            if use_plddt_prediction_head:
                aux_loss = _compute_plddt_aux_loss(
                    plddt_logits,
                    plddt_bins,
                    labels,
                    mode=plddt_prediction_mode,
                    regression_loss=plddt_regression_loss,
                )
                if aux_loss is not None:
                    loss = loss + (plddt_loss_weight * aux_loss)

            # Aux track loss (bend, torsion, ...)
            aux_bins_batch = batch.get("aux_bins")
            if model_aux_logits and aux_bins_batch:
                track_loss, per_track_losses_val = _compute_aux_track_loss(
                    model_aux_logits, aux_bins_batch, labels, aux_loss_weight
                )
                if track_loss is not None:
                    loss = loss + track_loss
                    for tname, tloss in per_track_losses_val.items():
                        val_track_loss_sum[tname] = val_track_loss_sum.get(tname, 0.0) + tloss.item()
            else:
                aux_bins_batch = None
        
        # Count non-masked tokens
        valid_mask = labels != -100
        token_count = valid_mask.sum().item()
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = ((preds == labels) & valid_mask).sum().item()
        # Per-track accuracy
        track_metrics_val = _compute_aux_track_metrics(model_aux_logits, aux_bins_batch)
        for tname, tm in track_metrics_val.items():
            val_track_correct[tname] = val_track_correct.get(tname, 0) + tm["correct"]
            val_track_tokens[tname] = val_track_tokens.get(tname, 0) + tm["tokens"]
        if use_plddt_prediction_head:
            aux_stats = _compute_aux_metrics(plddt_logits, plddt_bins, valid_mask, mode=plddt_prediction_mode)
            total_aux_tokens += aux_stats.get("tokens", 0)
            total_aux_correct += aux_stats.get("correct", 0)
            total_aux_abs_error_sum += aux_stats.get("abs_error_sum", 0.0)
            total_aux_sq_error_sum += aux_stats.get("sq_error_sum", 0.0)
            total_aux_bin_correct += aux_stats.get("bin_correct", 0)
        
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

        if track_iterations:
            n_iterations = getattr(outputs, 'n_iterations', None)
            if n_iterations is not None:
                if isinstance(n_iterations, torch.Tensor):
                    n_iter_val = n_iterations.float().mean().item()
                elif isinstance(n_iterations, (list, tuple)):
                    n_iter_val = float(sum(n_iterations) / max(len(n_iterations), 1))
                else:
                    n_iter_val = float(n_iterations)
                iter_weighted_sum += n_iter_val * max(token_count, 1)
                iter_weighted_tokens += max(token_count, 1)
    
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

    aux_metrics = {}
    if use_plddt_prediction_head:
        aux_metrics['tokens'] = total_aux_tokens
        if plddt_prediction_mode == "classification":
            aux_metrics['accuracy'] = total_aux_correct / max(total_aux_tokens, 1)
        else:
            aux_metrics['mae'] = total_aux_abs_error_sum / max(total_aux_tokens, 1)
            aux_metrics['rmse'] = math.sqrt(total_aux_sq_error_sum / max(total_aux_tokens, 1))
            aux_metrics['bin_accuracy'] = total_aux_bin_correct / max(total_aux_tokens, 1)
    # Per-track aux accuracy and loss
    if val_track_correct:
        aux_metrics['aux_track_accuracy'] = {
            t: val_track_correct[t] / max(val_track_tokens.get(t, 1), 1)
            for t in val_track_correct if val_track_tokens.get(t, 0) > 0
        }
        aux_metrics['aux_track_tokens'] = val_track_tokens
    if val_track_loss_sum:
        aux_metrics['aux_track_loss'] = val_track_loss_sum

    if track_iterations:
        avg_n_iterations = iter_weighted_sum / max(iter_weighted_tokens, 1)
        return avg_loss, accuracy, total_tokens, plddt_accuracies, aux_metrics, avg_n_iterations

    return avg_loss, accuracy, total_tokens, plddt_accuracies, aux_metrics

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
    use_plddt_prediction_head = getattr(args, 'use_plddt_prediction_head', False)
    plddt_prediction_mode = getattr(args, 'plddt_prediction_mode', 'classification')
    plddt_regression_loss = getattr(args, 'plddt_regression_loss', 'huber')

    # Auxiliary categorical tracks (bend, torsion, …)
    aux_fastas: dict = getattr(args, 'aux_fastas', None) or {}
    aux_track_num_bins: dict = getattr(args, 'aux_track_num_bins', None) or {}
    aux_loss_weight: float = float(getattr(args, 'aux_loss_weight', 0.5))
    use_aux_tracks = bool(aux_fastas) and bool(aux_track_num_bins)

    dataset = Seq3DiDataset(
        args.aa_fasta,
        args.three_di_fasta,
        mask_label_chars=args.mask_label_chars,
        plddt_bins_fasta=args.plddt_bins_fasta,
        aux_fastas=aux_fastas if use_aux_tracks else None,
    )
    print(f"Loaded {len(dataset)} sequences")
    print(f"3Di vocab ({len(dataset.label_vocab)}): {dataset.label_vocab}")
    if args.mask_label_chars and not use_plddt:
        print(f"Masked 3Di chars (ignored in loss): "
                f"{list(set(args.mask_label_chars))}")
    if use_plddt:
        print(f"pLDDT bins loaded from: {args.plddt_bins_fasta}")
        print("Note: Using pLDDT weighting - use original ground truth 3Di (no X masking)")
    if use_plddt_prediction_head:
        if plddt_prediction_mode == "regression":
            print(
                f"Auxiliary pLDDT prediction head enabled (regression, "
                f"loss={plddt_regression_loss}, loss weight={args.plddt_loss_weight})"
            )
        else:
            print(f"Auxiliary pLDDT prediction head enabled ({len(PLDDT_BIN_VOCAB)} bins, loss weight={args.plddt_loss_weight})")
    if use_aux_tracks:
        for tname, nbins in aux_track_num_bins.items():
            print(f"Aux track '{tname}': {nbins} bins, loss weight={aux_loss_weight}")
    
    
    # Create model wrapper (T5ProteinModel for T5-based, ESM3DiModel otherwise)
    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    # Check if model is T5-based (ProstT5, ProtT5, Ankh)
    is_t5 = is_t5_model(args.hf_model)
    esm_model_cls = _load_esm3di_model_class(getattr(args, 'use_iterative_backbone_head', False))
    
    if is_t5:
        if getattr(args, 'use_iterative_backbone_head', False):
            # Use copy-class iterative backbone path for T5 models
            esm_kwargs = dict(
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
                use_iterative_transformer_head=getattr(args, 'use_iterative_transformer_head', False),
                iterative_head_max_iterations=getattr(args, 'iterative_head_max_iterations', 5),
                iterative_head_halt_threshold=getattr(args, 'iterative_head_halt_threshold', 0.95),
                iterative_head_lambda_p=getattr(args, 'iterative_head_lambda_p', 0.01),
                iterative_head_prior_p=getattr(args, 'iterative_head_prior_p', 0.5),
                use_positional_encoding=getattr(args, 'use_positional_encoding', True),
                use_hidden_state_feedback=getattr(args, 'use_hidden_state_feedback', True),
                use_gru_gate=getattr(args, 'use_gru_gate', False),
                use_plddt_prediction_head=use_plddt_prediction_head,
                plddt_num_bins=len(PLDDT_BIN_VOCAB),
                plddt_prediction_mode=plddt_prediction_mode,
                use_iterative_backbone_head=True,
                iterative_backbone_k_layers=getattr(args, 'iterative_backbone_k_layers', 2),
            )
            esm_model = esm_model_cls(**_filtered_model_kwargs(esm_model_cls, esm_kwargs))
        else:
            # Use T5ProteinModel for T5-based models
            esm_model = T5ProteinModel(
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
                use_iterative_transformer_head=getattr(args, 'use_iterative_transformer_head', False),
                iterative_head_max_iterations=getattr(args, 'iterative_head_max_iterations', 5),
                iterative_head_halt_threshold=getattr(args, 'iterative_head_halt_threshold', 0.95),
                iterative_head_lambda_p=getattr(args, 'iterative_head_lambda_p', 0.01),
                iterative_head_prior_p=getattr(args, 'iterative_head_prior_p', 0.5),
                use_positional_encoding=getattr(args, 'use_positional_encoding', True),
                use_hidden_state_feedback=getattr(args, 'use_hidden_state_feedback', True),
                use_gru_gate=getattr(args, 'use_gru_gate', False),
                use_plddt_prediction_head=use_plddt_prediction_head,
                plddt_num_bins=len(PLDDT_BIN_VOCAB),
                plddt_prediction_mode=plddt_prediction_mode,
                half_precision=getattr(args, 'half_precision', False),
                gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
                aux_track_num_bins=aux_track_num_bins if use_aux_tracks else None,
            )
    else:
        # Use ESM3DiModel for ESM2/ESM++ models
        esm_kwargs = dict(
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
            use_iterative_transformer_head=getattr(args, 'use_iterative_transformer_head', False),
            iterative_head_max_iterations=getattr(args, 'iterative_head_max_iterations', 5),
            iterative_head_halt_threshold=getattr(args, 'iterative_head_halt_threshold', 0.95),
            iterative_head_lambda_p=getattr(args, 'iterative_head_lambda_p', 0.01),
            iterative_head_prior_p=getattr(args, 'iterative_head_prior_p', 0.5),
            use_positional_encoding=getattr(args, 'use_positional_encoding', True),
            use_hidden_state_feedback=getattr(args, 'use_hidden_state_feedback', True),
            use_gru_gate=getattr(args, 'use_gru_gate', False),
            use_plddt_prediction_head=use_plddt_prediction_head,
            plddt_num_bins=len(PLDDT_BIN_VOCAB),
            plddt_prediction_mode=plddt_prediction_mode,
            use_iterative_backbone_head=getattr(args, 'use_iterative_backbone_head', False),
            iterative_backbone_k_layers=getattr(args, 'iterative_backbone_k_layers', 2),
            aux_track_num_bins=aux_track_num_bins if use_aux_tracks else None,
        )
        esm_model = esm_model_cls(**_filtered_model_kwargs(esm_model_cls, esm_kwargs))
    
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

    # Get tokenizer from model (T5ProteinModel and ESM3DiModel both have tokenizer attribute)
    if is_t5:
        if hasattr(esm_model, 'get_tokenizer'):
            tokenizer = esm_model.get_tokenizer()
        elif hasattr(esm_model, 'tokenizer') and esm_model.tokenizer is not None:
            tokenizer = esm_model.tokenizer
        else:
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
        print("\nUsing T5 tokenizer (space-separated amino acids)")
    elif hasattr(esm_model, 'tokenizer') and esm_model.tokenizer is not None:
        tokenizer = esm_model.tokenizer
    elif "esm2" in args.hf_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            trust_remote_code=True
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
        max_seq_length=args.max_seq_length,
        aux_track_names=dataset.aux_track_names if use_aux_tracks else None,
    )
    if args.max_seq_length:
        print(f"Truncating sequences to max length: {args.max_seq_length}")
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
    val_aux_fastas: dict = getattr(args, 'val_aux_fastas', None) or {}
    if args.val_aa_fasta and args.val_three_di_fasta:
        print(f"\nSetting up validation data...")
        val_dataset = Seq3DiDataset(
            args.val_aa_fasta,
            args.val_three_di_fasta,
            mask_label_chars=args.mask_label_chars,
            plddt_bins_fasta=args.val_plddt_bins_fasta,
            aux_fastas=val_aux_fastas if (use_aux_tracks and val_aux_fastas) else None,
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
            max_seq_length=args.max_seq_length,
            aux_track_names=val_dataset.aux_track_names if use_aux_tracks else None,
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
        running_aux_correct = 0
        running_aux_tokens = 0
        running_aux_abs_error_sum = 0.0
        running_aux_sq_error_sum = 0.0
        running_aux_bin_correct = 0
        running_iter_sum = 0.0
        running_iter_count = 0
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_correct = 0
        epoch_aux_correct = 0
        epoch_aux_tokens = 0
        epoch_aux_abs_error_sum = 0.0
        epoch_aux_sq_error_sum = 0.0
        epoch_aux_bin_correct = 0
        epoch_iter_sum = 0.0
        epoch_iter_count = 0
        # Per-track aux accumulators
        epoch_track_correct: dict = {}
        epoch_track_tokens: dict = {}
        epoch_track_loss_sum: dict = {}
        
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
            
            # Move non-dict tensors; aux_bins is a nested dict, handle separately
            aux_bins_batch = batch.pop("aux_bins", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if aux_bins_batch:
                aux_bins_batch = {k: v.to(device) for k, v in aux_bins_batch.items()}
            
            # Extract labels and optional pLDDT bins for external loss computation
            labels = batch.pop("labels")
            plddt_bins = batch.pop("plddt_bins", None)
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    logits = outputs.logits
                    plddt_logits = getattr(outputs, 'plddt_logits', None)
                    model_aux_logits = getattr(outputs, 'aux_logits', None) or {}
                    loss = _compute_primary_loss(
                        loss_fn,
                        logits,
                        labels,
                        use_plddt=use_plddt,
                        plddt_bins=plddt_bins,
                        is_cyclical=is_cyclical,
                        epoch=epoch,
                    )
                    aux_loss = None
                    if use_plddt_prediction_head:
                        aux_loss = _compute_plddt_aux_loss(
                            plddt_logits,
                            plddt_bins,
                            labels,
                            mode=plddt_prediction_mode,
                            regression_loss=plddt_regression_loss,
                        )
                        if aux_loss is not None:
                            loss = loss + (args.plddt_loss_weight * aux_loss)
                    # Aux track loss
                    track_total_loss, per_track_losses = _compute_aux_track_loss(
                        model_aux_logits, aux_bins_batch, labels, aux_loss_weight
                    )
                    if track_total_loss is not None:
                        loss = loss + track_total_loss
            else:
                outputs = model(**batch)
                logits = outputs.logits
                plddt_logits = getattr(outputs, 'plddt_logits', None)
                model_aux_logits = getattr(outputs, 'aux_logits', None) or {}
                loss = _compute_primary_loss(
                    loss_fn,
                    logits,
                    labels,
                    use_plddt=use_plddt,
                    plddt_bins=plddt_bins,
                    is_cyclical=is_cyclical,
                    epoch=epoch,
                )
                aux_loss = None
                if use_plddt_prediction_head:
                    aux_loss = _compute_plddt_aux_loss(
                        plddt_logits,
                        plddt_bins,
                        labels,
                        mode=plddt_prediction_mode,
                        regression_loss=plddt_regression_loss,
                    )
                    if aux_loss is not None:
                        loss = loss + (args.plddt_loss_weight * aux_loss)
                # Aux track loss
                track_total_loss, per_track_losses = _compute_aux_track_loss(
                    model_aux_logits, aux_bins_batch, labels, aux_loss_weight
                )
                if track_total_loss is not None:
                    loss = loss + track_total_loss
            
            # Add halt regularization loss for iterative transformer head
            halt_reg_loss = getattr(outputs, 'halt_reg_loss', None)
            if halt_reg_loss is not None:
                # DataParallel returns tensor per GPU, take mean
                if multi_gpu and halt_reg_loss.dim() > 0:
                    halt_reg_loss = halt_reg_loss.mean()
                loss = loss + halt_reg_loss
            
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
                # Aux track metrics
                aux_track_metrics_batch = _compute_aux_track_metrics(model_aux_logits, aux_bins_batch)
                for tname, tm in aux_track_metrics_batch.items():
                    epoch_track_correct[tname] = epoch_track_correct.get(tname, 0) + tm["correct"]
                    epoch_track_tokens[tname] = epoch_track_tokens.get(tname, 0) + tm["tokens"]
                if per_track_losses:
                    for tname, tloss in per_track_losses.items():
                        epoch_track_loss_sum[tname] = epoch_track_loss_sum.get(tname, 0.0) + tloss.item()
                aux_stats = _compute_aux_metrics(plddt_logits, plddt_bins, valid_mask, mode=plddt_prediction_mode)
                aux_tokens = aux_stats.get("tokens", 0)
                aux_correct = aux_stats.get("correct", 0)
                aux_abs_error_sum = aux_stats.get("abs_error_sum", 0.0)
                aux_sq_error_sum = aux_stats.get("sq_error_sum", 0.0)
                aux_bin_correct = aux_stats.get("bin_correct", 0)
                epoch_aux_correct += aux_correct
                epoch_aux_tokens += aux_tokens
                epoch_aux_abs_error_sum += aux_abs_error_sum
                epoch_aux_sq_error_sum += aux_sq_error_sum
                epoch_aux_bin_correct += aux_bin_correct
                running_aux_correct += aux_correct
                running_aux_tokens += aux_tokens
                running_aux_abs_error_sum += aux_abs_error_sum
                running_aux_sq_error_sum += aux_sq_error_sum
                running_aux_bin_correct += aux_bin_correct
                
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
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{running_accuracy:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                }
                if track_total_loss is not None:
                    postfix["t_loss"] = f"{track_total_loss.item():.4f}"
                current_progress_bar.set_postfix(postfix)
                
                # Log to TensorBoard
                writer.add_scalar('Loss/train_step', avg_loss, global_step)
                writer.add_scalar('Accuracy/train_step', running_accuracy, global_step)
                if use_plddt_prediction_head and running_aux_tokens > 0:
                    if plddt_prediction_mode == "classification":
                        writer.add_scalar('Accuracy/plddt_aux_train_step', running_aux_correct / running_aux_tokens, global_step)
                    else:
                        writer.add_scalar('PLDDT_Aux/mae_train_step', running_aux_abs_error_sum / running_aux_tokens, global_step)
                        writer.add_scalar('PLDDT_Aux/rmse_train_step', math.sqrt(running_aux_sq_error_sum / running_aux_tokens), global_step)
                        writer.add_scalar('PLDDT_Aux/bin_accuracy_train_step', running_aux_bin_correct / running_aux_tokens, global_step)
                if use_plddt_prediction_head and aux_loss is not None:
                    writer.add_scalar('Loss/plddt_aux_train_step', aux_loss.item(), global_step)
                # Log aux track step metrics
                for tname, tm in aux_track_metrics_batch.items():
                    if tm["tokens"] > 0:
                        acc = tm["correct"] / tm["tokens"]
                        writer.add_scalar(f'AuxTrack/{tname}_accuracy_step', acc, global_step)
                if track_total_loss is not None:
                    writer.add_scalar('Loss/aux_tracks_train_step', track_total_loss.item(), global_step)
                
                if global_step % args.log_every == 0:
                    # Log running average to TensorBoard
                    writer.add_scalar('Loss/train_running_avg', avg_loss, global_step)
                    writer.add_scalar('Accuracy/train_running_avg', running_accuracy, global_step)
                    writer.add_scalar('Training/tokens_processed', running_tokens, global_step)
                    if use_plddt_prediction_head and running_aux_tokens > 0:
                        if plddt_prediction_mode == "classification":
                            writer.add_scalar('Accuracy/plddt_aux_train_running_avg', running_aux_correct / running_aux_tokens, global_step)
                        else:
                            writer.add_scalar('PLDDT_Aux/mae_train_running_avg', running_aux_abs_error_sum / running_aux_tokens, global_step)
                            writer.add_scalar('PLDDT_Aux/rmse_train_running_avg', math.sqrt(running_aux_sq_error_sum / running_aux_tokens), global_step)
                            writer.add_scalar('PLDDT_Aux/bin_accuracy_train_running_avg', running_aux_bin_correct / running_aux_tokens, global_step)
                    if running_iter_count > 0:
                        writer.add_scalar('Halt/train_n_iterations_running_window', running_iter_sum / running_iter_count, global_step)
                        print(f"  Iterations (running avg): {running_iter_sum / running_iter_count:.3f}")
                    running_loss = 0.0
                    running_tokens = 0
                    running_correct = 0
                    running_aux_correct = 0
                    running_aux_tokens = 0
                    running_aux_abs_error_sum = 0.0
                    running_aux_sq_error_sum = 0.0
                    running_aux_bin_correct = 0
                    running_iter_sum = 0.0
                    running_iter_count = 0
                
                # Periodic GPU cache clearing to prevent memory fragmentation
                # Clear every 5 optimization steps
                if global_step % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
            if use_plddt_prediction_head and aux_loss is not None:
                writer.add_scalar('Loss/plddt_aux_train_batch', aux_loss.item(), step + (epoch - 1) * len(loader))
            if use_plddt_prediction_head and aux_tokens > 0:
                if plddt_prediction_mode == "classification":
                    writer.add_scalar('Accuracy/plddt_aux_train_batch', aux_correct / aux_tokens, step + (epoch - 1) * len(loader))
                else:
                    writer.add_scalar('PLDDT_Aux/mae_train_batch', aux_abs_error_sum / aux_tokens, step + (epoch - 1) * len(loader))
                    writer.add_scalar('PLDDT_Aux/rmse_train_batch', math.sqrt(aux_sq_error_sum / aux_tokens), step + (epoch - 1) * len(loader))
                    writer.add_scalar('PLDDT_Aux/bin_accuracy_train_batch', aux_bin_correct / aux_tokens, step + (epoch - 1) * len(loader))
            
            # Log halt statistics for iterative transformer head
            if halt_reg_loss is not None:
                # halt_reg_loss is already mean-reduced above
                writer.add_scalar('Halt/reg_loss', halt_reg_loss.item() if halt_reg_loss.dim() == 0 else halt_reg_loss.mean().item(), step + (epoch - 1) * len(loader))
            n_iterations = getattr(outputs, 'n_iterations', None)
            if n_iterations is not None:
                # n_iterations may be a tensor (DataParallel), take mean
                if isinstance(n_iterations, torch.Tensor):
                    n_iter_val = n_iterations.float().mean().item()
                elif isinstance(n_iterations, (list, tuple)):
                    n_iter_val = float(sum(n_iterations) / max(len(n_iterations), 1))
                else:
                    n_iter_val = float(n_iterations)
                writer.add_scalar('Halt/n_iterations', n_iter_val, step + (epoch - 1) * len(loader))
                running_iter_sum += n_iter_val
                running_iter_count += 1
                epoch_iter_sum += n_iter_val
                epoch_iter_count += 1
                writer.add_scalar('Halt/train_n_iterations_batch', n_iter_val, step + (epoch - 1) * len(loader))
                writer.add_scalar('Halt/train_n_iterations_running_avg', running_iter_sum / max(running_iter_count, 1), step + (epoch - 1) * len(loader))

        # Log epoch-level metrics to TensorBoard
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        epoch_accuracy = epoch_correct / max(epoch_tokens, 1)
        writer.add_scalar('Loss/train_epoch', epoch_avg_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', epoch_accuracy, epoch)
        writer.add_scalar('Training/epoch_tokens', epoch_tokens, epoch)
        # Per-track aux epoch metrics
        for tname, tcorrect in epoch_track_correct.items():
            ttokens = epoch_track_tokens.get(tname, 0)
            if ttokens > 0:
                writer.add_scalar(f'AuxTrack/{tname}_accuracy_epoch', tcorrect / ttokens, epoch)
        for tname, tloss_sum in epoch_track_loss_sum.items():
            writer.add_scalar(f'AuxTrack/{tname}_loss_epoch', tloss_sum, epoch)
        if use_plddt_prediction_head and epoch_aux_tokens > 0:
            if plddt_prediction_mode == "classification":
                writer.add_scalar('Accuracy/plddt_aux_train_epoch', epoch_aux_correct / epoch_aux_tokens, epoch)
            else:
                writer.add_scalar('PLDDT_Aux/mae_train_epoch', epoch_aux_abs_error_sum / epoch_aux_tokens, epoch)
                writer.add_scalar('PLDDT_Aux/rmse_train_epoch', math.sqrt(epoch_aux_sq_error_sum / epoch_aux_tokens), epoch)
                writer.add_scalar('PLDDT_Aux/bin_accuracy_train_epoch', epoch_aux_bin_correct / epoch_aux_tokens, epoch)
        if epoch_iter_count > 0:
            epoch_avg_n_iterations = epoch_iter_sum / epoch_iter_count
            writer.add_scalar('Halt/train_n_iterations_epoch', epoch_avg_n_iterations, epoch)
        else:
            epoch_avg_n_iterations = None
        
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
        # Per-track aux epoch print
        for tname in sorted(epoch_track_correct):
            tcorrect = epoch_track_correct[tname]
            ttokens = epoch_track_tokens.get(tname, 0)
            acc = tcorrect / max(ttokens, 1)
            tloss_str = f", loss={epoch_track_loss_sum[tname]:.4f}" if tname in epoch_track_loss_sum else ""
            print(f"Epoch {epoch} Aux '{tname}': acc={acc:.4f}{tloss_str} ({tcorrect}/{ttokens} residues)")
        if use_plddt_prediction_head and epoch_aux_tokens > 0:
            if plddt_prediction_mode == "classification":
                print(f"Epoch {epoch} pLDDT Aux Accuracy: {epoch_aux_correct / epoch_aux_tokens:.4f} ({epoch_aux_correct}/{epoch_aux_tokens} residues)")
            else:
                print(
                    f"Epoch {epoch} pLDDT Aux MAE: {epoch_aux_abs_error_sum / epoch_aux_tokens:.4f} | "
                    f"RMSE: {math.sqrt(epoch_aux_sq_error_sum / epoch_aux_tokens):.4f} | "
                    f"Bin-Acc: {epoch_aux_bin_correct / epoch_aux_tokens:.4f} "
                    f"({epoch_aux_tokens} residues)"
                )
        if epoch_avg_n_iterations is not None:
            print(f"Epoch {epoch} Avg Iterations: {epoch_avg_n_iterations:.3f}")
        
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
            if args.use_iterative_transformer_head:
                val_loss, val_accuracy, val_tokens, val_plddt_accuracies, val_aux_metrics, val_avg_n_iterations = validate(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    use_amp=use_amp,
                    use_plddt=use_plddt,
                    use_plddt_prediction_head=use_plddt_prediction_head,
                    plddt_prediction_mode=plddt_prediction_mode,
                    plddt_regression_loss=plddt_regression_loss,
                    plddt_loss_weight=args.plddt_loss_weight,
                    epoch=epoch,
                    track_iterations=True,
                )
            else:
                val_loss, val_accuracy, val_tokens, val_plddt_accuracies, val_aux_metrics = validate(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    use_amp=use_amp,
                    use_plddt=use_plddt,
                    use_plddt_prediction_head=use_plddt_prediction_head,
                    plddt_prediction_mode=plddt_prediction_mode,
                    plddt_regression_loss=plddt_regression_loss,
                    plddt_loss_weight=args.plddt_loss_weight,
                    epoch=epoch,
                )
                val_avg_n_iterations = None
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f} ({int(val_accuracy * val_tokens)}/{val_tokens} residues)")
            # Per-track aux validation metrics
            for tname in sorted(val_aux_metrics.get('aux_track_accuracy', {})):
                acc = val_aux_metrics['aux_track_accuracy'][tname]
                ttokens = val_aux_metrics.get('aux_track_tokens', {}).get(tname, 0)
                tloss_str = f", loss={val_aux_metrics['aux_track_loss'][tname]:.4f}" if tname in val_aux_metrics.get('aux_track_loss', {}) else ""
                print(f"Validation Aux '{tname}': acc={acc:.4f}{tloss_str} ({int(acc*ttokens)}/{ttokens} residues)")
            if use_plddt_prediction_head and val_aux_metrics.get('tokens', 0) > 0:
                if plddt_prediction_mode == "classification":
                    print(f"Validation pLDDT Aux Accuracy: {val_aux_metrics['accuracy']:.4f} ({int(val_aux_metrics['accuracy'] * val_aux_metrics['tokens'])}/{val_aux_metrics['tokens']} residues)")
                else:
                    print(
                        f"Validation pLDDT Aux MAE: {val_aux_metrics['mae']:.4f} | "
                        f"RMSE: {val_aux_metrics['rmse']:.4f} | "
                        f"Bin-Acc: {val_aux_metrics['bin_accuracy']:.4f} ({val_aux_metrics['tokens']} residues)"
                    )
            if val_avg_n_iterations is not None:
                print(f"Validation Avg Iterations: {val_avg_n_iterations:.3f}")
            
            # Print pLDDT-stratified validation accuracy
            if use_plddt and val_plddt_accuracies:
                val_plddt_str = " | ".join([f"≥{t*10}: {val_plddt_accuracies.get(t, 0):.4f}" for t in plddt_thresholds if t in val_plddt_accuracies])
                print(f"Validation Accuracy by pLDDT: {val_plddt_str}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
            writer.add_scalar('Validation/tokens', val_tokens, epoch)
            # Per-track aux validation logging
            for tname, acc in val_aux_metrics.get('aux_track_accuracy', {}).items():
                writer.add_scalar(f'AuxTrack/{tname}_accuracy_val', acc, epoch)
            for tname, tloss_sum in val_aux_metrics.get('aux_track_loss', {}).items():
                writer.add_scalar(f'AuxTrack/{tname}_loss_val', tloss_sum, epoch)
            if use_plddt_prediction_head and val_aux_metrics.get('tokens', 0) > 0:
                if plddt_prediction_mode == "classification":
                    writer.add_scalar('Accuracy/plddt_aux_val_epoch', val_aux_metrics['accuracy'], epoch)
                else:
                    writer.add_scalar('PLDDT_Aux/mae_val_epoch', val_aux_metrics['mae'], epoch)
                    writer.add_scalar('PLDDT_Aux/rmse_val_epoch', val_aux_metrics['rmse'], epoch)
                    writer.add_scalar('PLDDT_Aux/bin_accuracy_val_epoch', val_aux_metrics['bin_accuracy'], epoch)
            if val_avg_n_iterations is not None:
                writer.add_scalar('Halt/val_n_iterations_epoch', val_avg_n_iterations, epoch)
            
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
            
            # Clear GPU cache after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
            "plddt_label_vocab": PLDDT_BIN_VOCAB if use_plddt_prediction_head else None,
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
        
        # Clear GPU cache at end of epoch to free fragmented memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
def predict_3di_for_fasta(model_ckpt: str, aa_fasta: str, device: str = None, output_plddt_fasta: str = None):
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

    esm_model_cls = _load_esm3di_model_class(args.get("use_iterative_backbone_head", False))
    predict_kwargs = dict(
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
        use_iterative_transformer_head=args.get("use_iterative_transformer_head", False),
        iterative_head_max_iterations=args.get("iterative_head_max_iterations", 5),
        iterative_head_halt_threshold=args.get("iterative_head_halt_threshold", 0.95),
        iterative_head_lambda_p=args.get("iterative_head_lambda_p", 0.01),
        iterative_head_prior_p=args.get("iterative_head_prior_p", 0.5),
        use_positional_encoding=args.get("use_positional_encoding", True),
        use_hidden_state_feedback=args.get("use_hidden_state_feedback", True),
        use_gru_gate=args.get("use_gru_gate", False),
        use_plddt_prediction_head=args.get("use_plddt_prediction_head", False),
        plddt_num_bins=args.get("plddt_num_bins", len(ckpt.get("plddt_label_vocab", PLDDT_BIN_VOCAB))),
        plddt_prediction_mode=args.get("plddt_prediction_mode", "classification"),
        use_iterative_backbone_head=args.get("use_iterative_backbone_head", False),
        iterative_backbone_k_layers=args.get("iterative_backbone_k_layers", 2),
    )
    esm_model = esm_model_cls(**_filtered_model_kwargs(esm_model_cls, predict_kwargs))
    return esm_model.predict_from_fasta(
        input_fasta_path=aa_fasta,
        output_fasta_path=os.devnull,
        model_checkpoint_path=model_ckpt,
        batch_size=args.get("batch_size", 4),
        device=device,
        output_confidence_fasta=output_plddt_fasta,
    )


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

    # Iterative Transformer Classification Head with Learned Halting
    p.add_argument("--use-iterative-transformer-head", action="store_true",
                   help="Use an iterative TransformerEncoder head with learned halting (PonderNet-style). "
                        "The model learns when to stop refining predictions.")
    p.add_argument("--iterative-head-max-iterations", type=int, default=5,
                   help="Maximum number of refinement iterations (default: 5). "
                        "Model may stop earlier based on learned halting.")
    p.add_argument("--iterative-head-halt-threshold", type=float, default=0.95,
                   help="Cumulative halt probability threshold for early stopping during inference (default: 0.95).")
    p.add_argument("--iterative-head-lambda-p", type=float, default=0.01,
                   help="Weight for halting regularization loss (KL with geometric prior). "
                        "Higher values encourage earlier stopping (default: 0.01).")
    p.add_argument("--iterative-head-prior-p", type=float, default=0.5,
                   help="Geometric prior parameter. Controls expected number of iterations. "
                        "p=0.5 expects ~2 iterations, p=0.33 expects ~3, p=0.2 expects ~5 (default: 0.5).")
    p.add_argument("--use-iterative-backbone-head", action="store_true",
                   help="Use iterative refinement over the protein LLM's final k layers (requires ESM3di_model copy.py implementation).")
    p.add_argument("--iterative-backbone-k-layers", type=int, default=2,
                   help="Number of final backbone layers to re-apply each iteration when using --use-iterative-backbone-head.")
    p.add_argument("--no-positional-encoding", dest="use_positional_encoding", action="store_false",
                   help="Disable positional encoding mixing in iterative transformer head.")
    p.add_argument("--no-hidden-state-feedback", dest="use_hidden_state_feedback", action="store_false",
                   help="Use logits-based feedback instead of hidden-state feedback in iterative transformer head.")
    p.add_argument("--use-gru-gate", action="store_true",
                   help="Use GRU-style gating for controlled hidden state updates in iterative transformer head. "
                        "Helps with gradient flow by selectively blending previous and current states.")
    p.set_defaults(use_positional_encoding=True, use_hidden_state_feedback=True, use_gru_gate=False)

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
    p.add_argument("--use-plddt-prediction-head", action="store_true",
                   help="Add an optional auxiliary token head that predicts pLDDT bins (0-9) during training and inference.")
    p.add_argument("--plddt-prediction-mode", type=str, default="regression",
                   choices=["classification", "regression"],
                   help="Auxiliary pLDDT head mode: 'classification' predicts 10 bins, 'regression' predicts continuous pLDDT.")
    p.add_argument("--plddt-regression-loss", type=str, default="huber",
                   choices=["huber", "mse"],
                   help="Regression loss for auxiliary pLDDT head when --plddt-prediction-mode=regression.")
    p.add_argument("--plddt-loss-weight", type=float, default=1.0,
                   help="Weight for the auxiliary pLDDT prediction loss when --use-plddt-prediction-head is enabled.")
    
    # Auxiliary structural tracks (bend, torsion, …)
    p.add_argument("--aux-fastas", type=str, default=None,
                   help="JSON dict mapping track name to FASTA path for auxiliary categorical "
                        "tracks. E.g. '{\"bend\":\"bend_bin.fasta\",\"torsion\":\"torsion_bin.fasta\"}'. "
                        "Can also be set via config file as a dict.")
    p.add_argument("--val-aux-fastas", type=str, default=None,
                   help="JSON dict mapping track name to validation FASTA path (same keys as --aux-fastas).")
    p.add_argument("--aux-track-num-bins", type=str, default=None,
                   help="JSON dict mapping track name to number of bins. "
                        "E.g. '{\"bend\":8,\"torsion\":12}'. Must match the track data. "
                        "Can also be set via config file as a dict.")
    p.add_argument("--aux-loss-weight", type=float, default=0.5,
                   help="Scalar weight applied to the total auxiliary track classification loss "
                        "(default: 0.5). Applied per-track before summing.")
    
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
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="Maximum sequence length. Sequences longer than this will be "
                        "truncated. If not set, uses the model's default max length. "
                        "Reducing this can help with OOM errors on large models.")
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

    head_count = sum([
        args.use_cnn_head,
        args.use_transformer_head,
        getattr(args, 'use_iterative_transformer_head', False)
    ])
    if head_count > 1:
        p.error("Only one iterative/classifier head mode can be used at a time")

    if getattr(args, 'use_iterative_backbone_head', False):
        head_count += 1
    if head_count > 1:
        p.error("Only one of --use-cnn-head, --use-transformer-head, --use-iterative-transformer-head, or --use-iterative-backbone-head can be used")

    if args.use_plddt_prediction_head and not args.plddt_bins_fasta:
        p.error("--use-plddt-prediction-head requires --plddt-bins-fasta")

    if args.use_plddt_prediction_head and (
        getattr(args, 'use_iterative_transformer_head', False)
        or getattr(args, 'use_iterative_backbone_head', False)
    ):
        p.error("--use-plddt-prediction-head is currently supported only for non-iterative ESM/T5 heads")

    # Parse JSON-string aux-track args (from CLI); config-file dicts pass through as-is
    import json as _json
    for _attr in ('aux_fastas', 'val_aux_fastas', 'aux_track_num_bins'):
        _val = getattr(args, _attr, None)
        if isinstance(_val, str) and _val.strip():
            try:
                setattr(args, _attr, _json.loads(_val))
            except ValueError as e:
                p.error(f"--{_attr.replace('_', '-')} must be valid JSON: {e}")
        elif _val is None:
            setattr(args, _attr, {})

    # Validate aux track consistency
    _aux_fastas = getattr(args, 'aux_fastas', {}) or {}
    _aux_bins  = getattr(args, 'aux_track_num_bins', {}) or {}
    if _aux_fastas or _aux_bins:
        missing_bins  = set(_aux_fastas) - set(_aux_bins)
        missing_fastas = set(_aux_bins) - set(_aux_fastas)
        if missing_bins:
            p.error(f"--aux-track-num-bins missing entries for tracks: {missing_bins}")
        if missing_fastas:
            p.error(f"--aux-fastas missing entries for tracks: {missing_fastas}")
    
    return args


def main():
    """Main entry point for training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
