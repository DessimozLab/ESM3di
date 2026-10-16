"""ESM3Di inference and shared ESM++ checkpoint runtime."""

import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.2.0"

_LAZY = {
    "ESM3DiModel": "_core.model",
    "ESMWithCNNHead": "_core.model",
    "CNNClassificationHead": "_core.model",
    "read_fasta": "_core.model",
    "discover_lora_target_modules": "_core.model",
    "freeze_all_but_lora_and_classifier": "_core.model",
    "load_3di_checkpoint": "_core.checkpoints",
    "load_mlm_model": "_core.checkpoints",
    "Predictor": "predict",
    "predict_records": "predict",
    "write_foldseek_db": "foldseek_db",
    "fasta2foldseek": "foldseek_db",
}
_TRAINING = {
    "FocalLoss": "losses", "CyclicalFocalLoss": "losses",
    "PLDDTWeightedFocalLoss": "losses", "PLDDTWeightedCyclicalFocalLoss": "losses",
    "GammaSchedulerOnPlateau": "losses", "DEFAULT_PLDDT_BIN_WEIGHTS": "losses",
    "prepare": "prepare_data",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    elif name in _TRAINING:
        value = getattr(importlib.import_module(f"esm3di_training.{_TRAINING[name]}"), name)
    else:
        raise AttributeError(f"module 'esm3di' has no attribute {name!r}")
    globals()[name] = value
    return value
