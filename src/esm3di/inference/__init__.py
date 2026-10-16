"""Inference-release interface, independent of the training CLI."""

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import predictor
    value = getattr(predictor, name)
    globals()[name] = value
    return value


__all__ = [
    "ESM3DiPredictor", "DEFAULT_HF_REPO", "DEFAULT_BATCH_SIZE",
    "DEFAULT_REVISION", "VOCAB_3DI",
]
