# Retired runtime implementations

This is non-installable historical source, not a supported alternate package.
It preserves the earlier ESM-2/T5, iterative/transformer heads, auxiliary-track
models, monolithic trainers, and experimental data tools. Original imports and
paths are historical and are not expected to run from this archive unchanged.

Neither the inference nor training wheel includes this code. Supported runtime
architectures are ESM++ with LoRA and the CNN head. Checkpoints requiring the
retired architectures now fail explicitly rather than being partially loaded.
Supported ESM++ MLM, supervised CNN checkpoints, Hub inference, and cut-site
classifier heads remain supported.
