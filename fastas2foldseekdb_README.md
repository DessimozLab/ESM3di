# fastas2foldseekdb

Build a FoldSeek database from amino-acid FASTA files by predicting 3Di tokens with a trained ESM3Di checkpoint. This is a one-step wrapper around `esm3di predict` + `esm3di foldseekdb`. See the [README](README.md) for training.

FoldSeek must be on `PATH`, or pass `--foldseek-bin` (or set `FOLDSEEK_BIN`).

## Usage

```bash
# Predict 3Di and build the DB
esm3di-makefoldseekdb \
    --aa-fasta proteins.fasta \
    --model-ckpt runs/3di_small/best.pt \
    --output-db dbs/proteins \
    --device cuda:0 [--num-gpus 2] [--keep-fastas]

# Use a pre-computed 3Di FASTA (no model needed)
esm3di-makefoldseekdb \
    --aa-fasta proteins.fasta \
    --three-di-fasta proteins_3di.fasta \
    --output-db dbs/proteins \
    --skip-inference
```

`python -m esm3di.fastas2foldseekdb` is equivalent. With the unified CLI, the same can be done as:

```bash
esm3di predict --checkpoint runs/3di_small/best.pt -i proteins.fasta -o proteins_3di.fasta --foldseek-db dbs/proteins
esm3di foldseekdb --aa-fasta proteins.fasta --three-di-fasta proteins_3di.fasta -o dbs/proteins
```

Any ESM3Di 3Di checkpoint works, including old `esmretrain.py` checkpoints and the Hub models (`cactuskid13/ESMpp_small_3Di`, `cactuskid13/esm2small_3di`).

## Options

| Option | Description |
|--------|-------------|
| `--aa-fasta` | Input amino-acid FASTA (required) |
| `--output-db` | Output FoldSeek database prefix (required) |
| `--model-ckpt` | Trained 3Di checkpoint (`.pt`) |
| `--three-di-fasta` + `--skip-inference` | Use existing 3Di predictions instead of running the model |
| `--device` | `cuda`, `cuda:1`, `cpu` |
| `--num-gpus` | Shard inference over GPUs `0..N-1` |
| `--batch-size` | Sequences per batch |
| `--revision` | Pin the HF remote-code revision of the backbone (normally automatic) |
| `--keep-fastas` | Also write `<output-db>_aa.fasta` and `<output-db>_3di.fasta` |
| `--output-aa-fasta`, `--output-3di-fasta`, `--output-confidence-fasta` | Explicit paths for the intermediate FASTAs / per-residue confidence |
| `--foldseek-bin` | Path to the foldseek binary |

## Output

FoldSeek sequence-DB files `{output_db}`, `{output_db}_ss` (3Di), `{output_db}_h` (headers) plus their `.index`/`.dbtype` files, ready for `foldseek search` / `easy-search`.

The earlier, more feature-rich script (taxonomy, custom models) is kept at `esm3di/experimental/fastas2foldseekdb_full.py`.
