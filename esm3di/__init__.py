"""ESM3Di: ESM + PEFT LoRA for 3Di per-residue prediction."""

__version__ = "0.1.0"

from .ESM3di_model import (
    ESM3DiModel,
    ESMWithCNNHead,
    CNNClassificationHead,
    Seq3DiDataset,
    read_fasta,
    discover_lora_target_modules,
    freeze_all_but_lora_and_classifier,
    merge_lora_weights,
    make_collate_fn,
)

from .esmretrain import (
    predict_3di_for_fasta,
)

__all__ = [
    # Main model class
    "ESM3DiModel",
    # Model components
    "ESMWithCNNHead",
    "CNNClassificationHead",
    # Data utilities
    "Seq3DiDataset",
    "read_fasta",
    "make_collate_fn",
    # LoRA utilities
    "discover_lora_target_modules",
    "freeze_all_but_lora_and_classifier",
    "merge_lora_weights",
    # Inference
    "predict_3di_for_fasta",
]
