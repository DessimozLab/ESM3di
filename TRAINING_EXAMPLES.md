# ESM3di Training Examples

## Basic Training (Single GPU)

```bash
python esmretrain.py \
  --hf-model facebook/esm2_t12_35M_UR50D \
  --aa-fasta data/sequences.fasta \
  --three-di-fasta data/3di.fasta \
  --batch-size 8 \
  --epochs 10 \
  --lr 1e-4
```

## Mixed Precision Training (Faster, Less Memory)

Recommended for all CUDA training - provides ~2x speedup and ~40% memory reduction:

```bash
python esmretrain.py \
  --hf-model Synthyra/ESMplusplus_small \
  --aa-fasta data/sequences.fasta \
  --three-di-fasta data/3di.fasta \
  --batch-size 16 \
  --mixed-precision \
  --epochs 10
```

## Multi-GPU Training

Automatically uses all available GPUs:

```bash
python esmretrain.py \
  --hf-model Synthyra/ESMplusplus_large \
  --aa-fasta data/sequences.fasta \
  --three-di-fasta data/3di.fasta \
  --batch-size 8 \
  --multi-gpu \
  --epochs 10
```

## Multi-GPU + Mixed Precision (Recommended for Best Performance)

Combines both optimizations for maximum throughput:

```bash
python esmretrain.py \
  --hf-model Synthyra/ESMplusplus_large \
  --aa-fasta data/sequences.fasta \
  --three-di-fasta data/3di.fasta \
  --batch-size 16 \
  --multi-gpu \
  --mixed-precision \
  --gradient-accumulation-steps 2 \
  --epochs 10 \
  --lr 1e-4
```

## With CNN Head and Advanced Features

```bash
python esmretrain.py \
  --hf-model Synthyra/ESMplusplus_small \
  --aa-fasta data/sequences.fasta \
  --three-di-fasta data/3di.fasta \
  --batch-size 8 \
  --mixed-precision \
  --multi-gpu \
  --use-cnn-head \
  --cnn-num-layers 3 \
  --cnn-kernel-size 5 \
  --gradient-accumulation-steps 4 \
  --scheduler-type cosine \
  --warmup-ratio 0.1 \
  --epochs 20
```

## Using Configuration File

```bash
python esmretrain.py --config config.json
```

Example `config.json`:
```json
{
  "hf_model": "Synthyra/ESMplusplus_large",
  "aa_fasta": "data/sequences.fasta",
  "three_di_fasta": "data/3di.fasta",
  "batch_size": 16,
  "mixed_precision": true,
  "multi_gpu": true,
  "gradient_accumulation_steps": 2,
  "epochs": 15,
  "lr": 1e-4,
  "scheduler_type": "cosine",
  "warmup_ratio": 0.1,
  "use_cnn_head": true,
  "out_dir": "checkpoints_esmpp"
}
```

## Performance Recommendations

### Memory Optimization
- Use `--mixed-precision` to reduce memory by ~40%
- Increase `--gradient-accumulation-steps` to simulate larger batches
- Reduce `--batch-size` if OOM errors occur

### Speed Optimization
- Use `--mixed-precision` for ~2x speedup on modern GPUs
- Use `--multi-gpu` to scale across multiple GPUs
- Increase `--num-workers` for faster data loading (try 4-8)

### Model Size vs Batch Size Guide

| Model | GPUs | Mixed Precision | Recommended Batch Size |
|-------|------|-----------------|------------------------|
| ESM2-35M | 1x A100 (40GB) | Yes | 32-64 |
| ESM2-150M | 1x A100 (40GB) | Yes | 16-32 |
| ESM2-650M | 1x A100 (40GB) | Yes | 8-16 |
| ESM++-small (333M) | 1x A100 (40GB) | Yes | 12-24 |
| ESM++-large (575M) | 1x A100 (40GB) | Yes | 8-12 |
| ESM++-large | 2x A100 (40GB) | Yes | 16-24 |
| ESM++-large | 4x A100 (40GB) | Yes | 32-48 |

### Typical Training Command for Production

```bash
python esmretrain.py \
  --hf-model Synthyra/ESMplusplus_large \
  --aa-fasta train/sequences.fasta \
  --three-di-fasta train/3di.fasta \
  --batch-size 12 \
  --mixed-precision \
  --multi-gpu \
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --scheduler-type cosine \
  --warmup-ratio 0.1 \
  --num-workers 8 \
  --out-dir checkpoints_production \
  --tensorboard-log-dir logs_production
```

## Monitoring Training

Start TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir=tensorboard_logs
```

Then open http://localhost:6006 in your browser.

## Resume Training from Checkpoint

```bash
python esmretrain.py \
  --config config.json \
  --resume-from-checkpoint checkpoints/epoch_5.pt
```
