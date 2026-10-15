# ESM3Di training distribution

This distribution adds the supported ESM++ masked-language-model and supervised
LoRA/CNN 3Di trainers to the lightweight `esm3di` inference distribution.

From the monorepo root, install both local projects:

```bash
pip install -e . -e ./training
```

For published packages, use `pip install 'esm3di[training]'`.
The two wheels own distinct files: `esm3di` provides inference/core; this
distribution provides `esm3di_training` and the old `esm3di.training` namespace.
No retired T5, ESM-2, iterative-head, or other experimental implementations are
included in either wheel. Their source is kept in the research archive.

Use `esm3di-learn --help` or `python -m esm3di_training --help`.
See the repository README and configs for the supported two-stage recipe.
