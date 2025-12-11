#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from ESM3di_model import ESM3DiModel, Seq3DiDataset, make_collate_fn
from esm.models.esmc import ESMC


class ESM3DiLightningModule(pl.LightningModule):
    """PyTorch Lightning module for ESM 3Di training."""
    
    def __init__(
        self,
        hf_model: str,
        label_vocab: list,
        tokenizer,
        char2idx: dict,
        mask_label_chars: str,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        use_cnn_head: bool = False,
        cnn_num_layers: int = 2,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler_type: str = 'cosine',
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer', 'label_vocab', 'char2idx'])
        
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.char2idx = char2idx
        self.mask_label_chars = mask_label_chars
        
        # Create ESM3Di model wrapper
        esm_model = ESM3DiModel(
            hf_model_name=hf_model,
            num_labels=len(label_vocab),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_cnn_head=use_cnn_head,
            cnn_num_layers=cnn_num_layers,
            cnn_kernel_size=cnn_kernel_size,
            cnn_dropout=cnn_dropout
        )
        self.model = esm_model.get_model()
        
        # Store scheduler params
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        
    def forward(self, **batch):
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        # Calculate metrics
        token_count = (batch["labels"] != -100).sum()
        per_residue_loss = loss / token_count.clamp(min=1)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/loss_per_residue', per_residue_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/tokens', token_count.float(), on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        # Calculate metrics
        token_count = (batch["labels"] != -100).sum()
        per_residue_loss = loss / token_count.clamp(min=1)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss_per_residue', per_residue_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        # Get trainable parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Estimate total steps if not provided
        if self.total_steps is None:
            # This is a rough estimate - actual steps depend on dataset size and batch size
            self.total_steps = self.trainer.estimated_stepping_batches
        
        # Determine warmup steps
        if self.warmup_steps is not None:
            warmup_steps = self.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.warmup_ratio)
        
        # Create scheduler based on type
        if self.scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.scheduler_type == 'constant':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=0.5,
            )
        else:
            # No scheduler
            return optimizer
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
    
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        if self.max_grad_norm > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.max_grad_norm,
                gradient_clip_algorithm="norm"
            )


class ESM3DiDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for ESM 3Di training."""
    
    def __init__(
        self,
        aa_fasta: str,
        three_di_fasta: str,
        tokenizer,
        mask_label_chars: str = "X",
        batch_size: int = 2,
        num_workers: int = 0,
        val_split: float = 0.0,
        val_fasta_aa: Optional[str] = None,
        val_fasta_3di: Optional[str] = None,
    ):
        super().__init__()
        self.aa_fasta = aa_fasta
        self.three_di_fasta = three_di_fasta
        self.tokenizer = tokenizer
        self.mask_label_chars = mask_label_chars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.val_fasta_aa = val_fasta_aa
        self.val_fasta_3di = val_fasta_3di
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        # Load full dataset
        full_dataset = Seq3DiDataset(
            self.aa_fasta,
            self.three_di_fasta,
            mask_label_chars=self.mask_label_chars,
        )
        
        self.label_vocab = full_dataset.label_vocab
        self.char2idx = full_dataset.char2idx
        
        # Split into train/val if needed
        if self.val_fasta_aa and self.val_fasta_3di:
            # Use separate validation files
            self.train_dataset = full_dataset
            self.val_dataset = Seq3DiDataset(
                self.val_fasta_aa,
                self.val_fasta_3di,
                mask_label_chars=self.mask_label_chars,
            )
        elif self.val_split > 0:
            # Split dataset
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        else:
            # No validation
            self.train_dataset = full_dataset
            self.val_dataset = None
    
    def train_dataloader(self):
        collate_fn = make_collate_fn(
            self.tokenizer,
            self.char2idx,
            mask_label_chars=self.mask_label_chars,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        
        collate_fn = make_collate_fn(
            self.tokenizer,
            self.char2idx,
            mask_label_chars=self.mask_label_chars,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


def load_config_file(config_path: str) -> dict:
    """Load training configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_args():
    p = argparse.ArgumentParser(
        description="ESM 3Di Training with PyTorch Lightning (SLURM-ready)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file. If provided, overrides other arguments.")
    
    # Data
    p.add_argument("--aa-fasta", type=str, required=False,
                   help="FASTA with amino-acid sequences (training).")
    p.add_argument("--three-di-fasta", type=str, required=False,
                   help="FASTA with matching 3Di sequences (training).")
    p.add_argument("--val-fasta-aa", type=str, default=None,
                   help="FASTA with amino-acid sequences (validation).")
    p.add_argument("--val-fasta-3di", type=str, default=None,
                   help="FASTA with matching 3Di sequences (validation).")
    p.add_argument("--val-split", type=float, default=0.0,
                   help="Fraction of training data to use for validation (if no val files provided).")

    # Model
    p.add_argument("--hf-model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="HuggingFace model name or ESMC model identifier.")
    p.add_argument("--mask-label-chars", type=str, default="X",
                   help="Characters in 3Di FASTA to treat as masked.")

    # LoRA
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (r)")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora-target-modules", type=str, default="",
                   help="Comma-separated list of module names to apply LoRA to.")

    # CNN Head
    p.add_argument("--use-cnn-head", action="store_true",
                   help="Use multi-layer CNN classification head.")
    p.add_argument("--cnn-num-layers", type=int, default=2,
                   help="Number of CNN layers.")
    p.add_argument("--cnn-kernel-size", type=int, default=3,
                   help="CNN kernel size.")
    p.add_argument("--cnn-dropout", type=float, default=0.1,
                   help="CNN dropout rate.")

    # Training
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size per GPU")
    p.add_argument("--accumulate-grad-batches", type=int, default=1,
                   help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Weight decay")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Max gradient norm for clipping. Set to 0 to disable.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of DataLoader workers per GPU")

    # Learning rate scheduler
    p.add_argument("--scheduler-type", type=str, default='cosine',
                   choices=['cosine', 'linear', 'constant', 'none'],
                   help="Learning rate scheduler type.")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Number of warmup steps.")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Warmup ratio (fraction of total steps).")

    # PyTorch Lightning Trainer args
    p.add_argument("--devices", type=int, default=1,
                   help="Number of GPUs per node. Use -1 for all available GPUs.")
    p.add_argument("--num-nodes", type=int, default=1,
                   help="Number of nodes (for multi-node training).")
    p.add_argument("--strategy", type=str, default="auto",
                   choices=["auto", "ddp", "ddp_find_unused_parameters_true", "fsdp"],
                   help="Distributed training strategy.")
    p.add_argument("--precision", type=str, default="32",
                   choices=["32", "16", "bf16", "16-mixed", "bf16-mixed"],
                   help="Training precision. Use 16-mixed or bf16-mixed for mixed precision.")
    p.add_argument("--check-val-every-n-epoch", type=int, default=1,
                   help="Run validation every N epochs.")
    p.add_argument("--log-every-n-steps", type=int, default=10,
                   help="Log metrics every N steps.")

    # Checkpointing
    p.add_argument("--out-dir", type=str, default="esm_lightning_ckpts",
                   help="Directory to save checkpoints")
    p.add_argument("--save-top-k", type=int, default=3,
                   help="Save top K checkpoints based on validation loss.")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from.")

    # Logging
    p.add_argument("--tensorboard-log-dir", type=str, default="lightning_logs",
                   help="Directory for TensorBoard logs")
    p.add_argument("--experiment-name", type=str, default="esm3di",
                   help="Experiment name for logging")

    # SLURM
    p.add_argument("--slurm", action="store_true",
                   help="Enable SLURM cluster mode (auto-detected if not set)")

    args = p.parse_args()
    
    # Load config file if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
        
        for key, value in config.items():
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                setattr(args, arg_key, value)
    
    # Validate required arguments
    if not args.aa_fasta or not args.three_di_fasta:
        p.error("--aa-fasta and --three-di-fasta are required")
    
    return args


def main():
    """
    Main entry point for PyTorch Lightning training.
    
    Usage:
        Single GPU:
            python esmretrain_lightning.py --config config.json --devices 1
        
        Multi-GPU on single node:
            python esmretrain_lightning.py --config config.json --devices 4
        
        Multi-GPU on SLURM cluster:
            sbatch slurm_train.sh
            
        Multi-node on SLURM:
            sbatch slurm_train_multinode.sh
    """
    args = parse_args()
    
    # Set up experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.experiment_name}_{timestamp}"
    
    print("\n" + "="*60)
    print("ESM 3Di Training with PyTorch Lightning")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Config: {args.config if args.config else 'command line args'}")
    print(f"Devices: {args.devices}")
    print(f"Nodes: {args.num_nodes}")
    print(f"Strategy: {args.strategy}")
    print(f"Precision: {args.precision}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"Effective batch size: {args.batch_size * args.devices * args.num_nodes * args.accumulate_grad_batches}")
    print("="*60 + "\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.hf_model}")
    if 'esmc' in args.hf_model:
        esm_model = ESMC.from_pretrained("esmc_300m")
        tokenizer = esm_model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    print("✓ Tokenizer loaded\n")
    
    # Create data module
    print("Setting up data module...")
    data_module = ESM3DiDataModule(
        aa_fasta=args.aa_fasta,
        three_di_fasta=args.three_di_fasta,
        tokenizer=tokenizer,
        mask_label_chars=args.mask_label_chars,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        val_fasta_aa=args.val_fasta_aa,
        val_fasta_3di=args.val_fasta_3di,
    )
    data_module.setup()
    print(f"✓ Loaded {len(data_module.train_dataset)} training sequences")
    if data_module.val_dataset:
        print(f"✓ Loaded {len(data_module.val_dataset)} validation sequences")
    print(f"✓ 3Di vocab ({len(data_module.label_vocab)}): {data_module.label_vocab}\n")
    
    # Parse target modules
    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    # Create Lightning module
    print("Creating model...")
    model = ESM3DiLightningModule(
        hf_model=args.hf_model,
        label_vocab=data_module.label_vocab,
        tokenizer=tokenizer,
        char2idx=data_module.char2idx,
        mask_label_chars=args.mask_label_chars,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        use_cnn_head=args.use_cnn_head,
        cnn_num_layers=args.cnn_num_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_dropout=args.cnn_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )
    print("✓ Model created\n")
    
    # Print trainable parameters
    try:
        model.model.print_trainable_parameters()
    except AttributeError:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} || Total: {total:,} || "
              f"Trainable %: {100 * trainable / total:.2f}")
    print()
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        filename=f"{experiment_name}-{{epoch:02d}}-{{val/loss:.4f}}" if data_module.val_dataset else f"{experiment_name}-{{epoch:02d}}",
        save_top_k=args.save_top_k,
        monitor="val/loss" if data_module.val_dataset else None,
        mode="min",
        save_last=True,
        every_n_epochs=1,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.tensorboard_log_dir,
        name=experiment_name,
    )
    
    # Setup strategy for DDP
    strategy_config = args.strategy
    if args.strategy == "ddp" or (args.strategy == "auto" and args.devices > 1):
        strategy_config = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # More efficient
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy_config,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        deterministic=False,
    )
    
    print("Starting training...\n")
    
    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoints saved to: {args.out_dir}")
    print(f"TensorBoard logs: {args.tensorboard_log_dir}/{experiment_name}")
    print(f"\nTo view results:")
    print(f"  tensorboard --logdir={args.tensorboard_log_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
