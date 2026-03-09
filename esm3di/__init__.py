"""ESM3Di: ESM + PEFT LoRA for 3Di per-residue prediction."""

__version__ = "0.1.0"

from .ESM3di_model import (
    ESM3DiModel,
    ESMWithCNNHead,
    ESMWithTransformerHead,
    CNNClassificationHead,
    TransformerClassificationHead,
    Lion,
    Seq3DiDataset,
    read_fasta,
    discover_lora_target_modules,
    freeze_all_but_lora_and_classifier,
    merge_lora_weights,
    make_collate_fn,
)

from .losses import (
    FocalLoss,
    CyclicalFocalLoss,
    PLDDTWeightedFocalLoss,
    PLDDTWeightedCyclicalFocalLoss,
    GammaSchedulerOnPlateau,
    DEFAULT_PLDDT_BIN_WEIGHTS,
)

from .esmretrain import (
    predict_3di_for_fasta,
)

from .tree_utils import (
    check_foldseek_installed,
    run_foldseek_createdb,
    run_foldseek_allvall,
    read_foldseek_db,
    tajima_distance,
    write_distance_matrix,
    run_mafft,
    run_raxml,
    run_quicktree,
    predict_3di_from_sequences,
)

__all__ = [
    # Main model class
    "ESM3DiModel",
    # Model components
    "ESMWithCNNHead",
    "ESMWithTransformerHead",
    "CNNClassificationHead",
    "TransformerClassificationHead",
    # Loss functions
    "FocalLoss",
    "CyclicalFocalLoss",
    "PLDDTWeightedFocalLoss",
    "PLDDTWeightedCyclicalFocalLoss",
    "GammaSchedulerOnPlateau",
    "DEFAULT_PLDDT_BIN_WEIGHTS",
    # Optimizers
    "Lion",
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
    # Tree building and FoldSeek utilities
    "check_foldseek_installed",
    "run_foldseek_createdb",
    "run_foldseek_allvall",
    "read_foldseek_db",
    "tajima_distance",
    "write_distance_matrix",
    "run_mafft",
    "run_raxml",
    "run_quicktree",
    "predict_3di_from_sequences",
]
