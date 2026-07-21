import torch

# Fixes legacy PyTorch config validation blocks for third-party libraries globally
try:
    import torch._dynamo.config as dynamo_config
    if not hasattr(dynamo_config, "recompile_limit"):
        # 1. Add it to the strict "allowed keys" validation set
        if hasattr(dynamo_config, "_allowed_keys"):
            dynamo_config._allowed_keys.add("recompile_limit")
        # 2. Bind the attribute so it exists
        dynamo_config.recompile_limit = getattr(dynamo_config, "cache_size_limit", 64)
except (ImportError, AttributeError):
    pass
