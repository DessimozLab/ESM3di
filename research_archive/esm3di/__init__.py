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

# Iterative head components (shared between ESM and T5 models)
from .iterative_head import (
    IterativeTransformerClassificationHead,
    IterativeTransformerOutput,
    ModelWithIterativeTransformerHead,
    ESMWithIterativeTransformerHead,  # Backward compatibility alias
)

from .T5Model import (
    T5ProteinModel,
    T5WithClassificationHead,
    is_t5_model,
    is_prostt5_model,
    discover_t5_lora_modules,
    prepare_t5_batch,
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
    # Main model classes
    "ESM3DiModel",
    "T5ProteinModel",
    # Model components
    "ESMWithCNNHead",
    "ESMWithTransformerHead",
    "ESMWithIterativeTransformerHead",
    "ModelWithIterativeTransformerHead",
    "T5WithClassificationHead",
    "CNNClassificationHead",
    "TransformerClassificationHead",
    "IterativeTransformerClassificationHead",
    "IterativeTransformerOutput",
    # Model detection
    "is_t5_model",
    "is_prostt5_model",
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
    "prepare_t5_batch",
    # LoRA utilities
    "discover_lora_target_modules",
    "discover_t5_lora_modules",
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
