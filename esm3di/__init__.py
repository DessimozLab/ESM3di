"""ESM3Di: ESM++ fine-tuning for per-residue 3Di structural-token prediction.

Supported pipeline (see README and ``esm3di --help``):

1. ``esm3di prepare``  structures -> paired AA / 3Di / pLDDT-bin FASTAs (FoldSeek)
2. ``esm3di mlm``      unsupervised masked-LM fine-tuning of ESM++ on sequences
3. ``esm3di train``    supervised 3Di tagging (LoRA + CNN head), starting from 2.
4. ``esm3di predict``  3Di FASTA (and optionally a FoldSeek DB) from sequences

Submodules are imported lazily so ``import esm3di`` stays cheap.
"""

import importlib

__version__ = "0.2.0"

# public name -> module that defines it
_LAZY = {
    # core model
    "ESM3DiModel": "ESM3di_model",
    "ESMWithCNNHead": "ESM3di_model",
    "ESMWithTransformerHead": "ESM3di_model",
    "CNNClassificationHead": "ESM3di_model",
    "TransformerClassificationHead": "ESM3di_model",
    "Lion": "ESM3di_model",
    "Seq3DiDataset": "ESM3di_model",
    "read_fasta": "ESM3di_model",
    "discover_lora_target_modules": "ESM3di_model",
    "freeze_all_but_lora_and_classifier": "ESM3di_model",
    "merge_lora_weights": "ESM3di_model",
    "make_collate_fn": "ESM3di_model",
    # losses
    "FocalLoss": "losses",
    "CyclicalFocalLoss": "losses",
    "PLDDTWeightedFocalLoss": "losses",
    "PLDDTWeightedCyclicalFocalLoss": "losses",
    "GammaSchedulerOnPlateau": "losses",
    "DEFAULT_PLDDT_BIN_WEIGHTS": "losses",
    # checkpoints / pipeline
    "load_3di_checkpoint": "checkpoints",
    "load_mlm_model": "checkpoints",
    "Predictor": "predict",
    "predict_records": "predict",
    "write_foldseek_db": "foldseek_db",
    "fasta2foldseek": "foldseek_db",
    "prepare": "prepare_data",
    # experimental (kept for backward compatibility)
    "IterativeTransformerClassificationHead": "experimental.iterative_head",
    "IterativeTransformerOutput": "experimental.iterative_head",
    "ModelWithIterativeTransformerHead": "experimental.iterative_head",
    "ESMWithIterativeTransformerHead": "experimental.iterative_head",
    "T5ProteinModel": "experimental.T5Model",
    "T5WithClassificationHead": "experimental.T5Model",
    "is_t5_model": "experimental.T5Model",
    "is_prostt5_model": "experimental.T5Model",
    "discover_t5_lora_modules": "experimental.T5Model",
    "prepare_t5_batch": "experimental.T5Model",
    "predict_3di_for_fasta": "experimental.esmretrain_full",
    "check_foldseek_installed": "experimental.tree_utils",
    "run_foldseek_createdb": "experimental.tree_utils",
    "run_foldseek_allvall": "experimental.tree_utils",
    "read_foldseek_db": "experimental.tree_utils",
    "tajima_distance": "experimental.tree_utils",
    "write_distance_matrix": "experimental.tree_utils",
    "run_mafft": "experimental.tree_utils",
    "run_raxml": "experimental.tree_utils",
    "run_quicktree": "experimental.tree_utils",
    "predict_3di_from_sequences": "experimental.tree_utils",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'esm3di' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
