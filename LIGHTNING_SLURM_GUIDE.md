# PyTorch Lightning SLURM Training Guide

This guide explains how to use the PyTorch Lightning training script with SLURM clusters for distributed multi-GPU training.

## Overview

The PyTorch Lightning implementation (`esmretrain_lightning.py`) provides several advantages:

- **Automatic distributed training**: Lightning handles all DDP/FSDP setup
- **SLURM integration**: Native support for SLURM clusters
- **Cleaner code**: Separates model, data, and training logic
- **Better logging**: Automatic TensorBoard integration
- **Checkpointing**: Automatic checkpoint saving with monitoring
- **Scalability**: Easy scaling from single GPU to multi-node clusters

## Quick Start

### 1. Local Single GPU Training

```bash
python esmretrain_lightning.py \
    --config config_example.json \
    --devices 1
```

### 2. Local Multi-GPU Training

```bash
python esmretrain_lightning.py \
    --config config_example.json \
    --devices 4 \
    --strategy ddp \
    --precision bf16-mixed
```

### 3. SLURM Single Node Multi-GPU

```bash
# Edit slurm_train.sh to configure resources
# Then submit:
sbatch slurm_train.sh

# This uses torchrun with --standalone for single-node training
```

### 4. SLURM Multi-Node Training

```bash
# Edit slurm_train_multinode.sh to configure resources
# Then submit:
sbatch slurm_train_multinode.sh

# This uses torchrun with rendezvous backend for multi-node coordination
```

## SLURM Scripts

### Available Scripts

1. **slurm_train.sh** - Single node, multi-GPU training
   - Default: 1 node, 4 GPUs
   - Uses: `torchrun --standalone` for single-node DDP
   - Good for: Most training runs, model sizes up to ~1B parameters

2. **slurm_train_multinode.sh** - Multi-node, multi-GPU training
   - Default: 4 nodes, 4 GPUs per node (16 total GPUs)
   - Uses: `torchrun` with c10d rendezvous for multi-node coordination
   - Good for: Large datasets, faster training

3. **slurm_train_fsdp.sh** - FSDP (Fully Sharded Data Parallel)
   - Default: 2 nodes, 8 GPUs per node (16 total GPUs)
   - Uses: `torchrun` with FSDP strategy for model sharding
   - Good for: Very large models that don't fit in single GPU memory

**Note:** All scripts now use `torchrun` (PyTorch's recommended distributed launcher) instead of raw `srun` for better fault tolerance and distributed coordination.

### Customizing SLURM Scripts

Edit the SBATCH directives at the top of each script:

```bash
#SBATCH --nodes=4              # Number of nodes
#SBATCH --gres=gpu:4           # GPUs per node
#SBATCH --cpus-per-task=8      # CPU cores per GPU
#SBATCH --time=48:00:00        # Max runtime
#SBATCH --mem=128G             # Memory per node
#SBATCH --partition=gpu        # Partition name (adjust for your cluster)
```

Common adjustments:
- **More GPUs**: Increase `--gres=gpu:N`
- **More memory**: Increase `--mem=XXXG`
- **Longer jobs**: Increase `--time=HH:MM:SS`
- **Different partition**: Change `--partition=NAME`

### Cluster-Specific Setup

You'll need to adjust module loading in the scripts based on your cluster:

```bash
# Example for cluster with modules
module load cuda/11.8
module load python/3.10
module load nccl/2.15

# Then activate your environment
conda activate esm3di
# OR
source venv/bin/activate
```

## Configuration

### JSON Config File

Create a configuration file (e.g., `config_lightning.json`):

```json
{
    "aa_fasta": "data/train_aa.fasta",
    "three_di_fasta": "data/train_3di.fasta",
    "val_fasta_aa": "data/val_aa.fasta",
    "val_fasta_3di": "data/val_3di.fasta",
    "hf_model": "facebook/esm2_t30_150M_UR50D",
    "mask_label_chars": "X",
    
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
    
    "batch_size": 8,
    "accumulate_grad_batches": 2,
    "epochs": 10,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    "scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    "num_workers": 4,
    "precision": "bf16-mixed",
    
    "out_dir": "checkpoints_lightning",
    "tensorboard_log_dir": "lightning_logs",
    "experiment_name": "esm2_150m_lora"
}
```

### Command-Line Arguments

All config options can be overridden via command line:

```bash
python esmretrain_lightning.py \
    --config config.json \
    --batch-size 16 \
    --lr 1e-4 \
    --devices 4 \
    --precision bf16-mixed
```

## Training Strategies

### DDP (DistributedDataParallel)

Best for most use cases. Each GPU holds a full copy of the model.

```bash
python esmretrain_lightning.py \
    --config config.json \
    --devices 4 \
    --strategy ddp
```

SLURM: Use `slurm_train.sh` or `slurm_train_multinode.sh`

### FSDP (Fully Sharded Data Parallel)

Best for very large models. Shards model parameters across GPUs.

```bash
python esmretrain_lightning.py \
    --config config.json \
    --devices 8 \
    --strategy fsdp
```

SLURM: Use `slurm_train_fsdp.sh`

## Mixed Precision Training

Significantly speeds up training and reduces memory usage:

- **bf16-mixed**: Brain float 16 (recommended if GPU supports it)
- **16-mixed**: Float 16 (more compatible, but may have stability issues)
- **32**: Full precision (slowest, most stable)

```bash
python esmretrain_lightning.py \
    --config config.json \
    --precision bf16-mixed
```

## Monitoring

### TensorBoard

Lightning automatically logs to TensorBoard:

```bash
# On login node or compute node with port forwarding
tensorboard --logdir=lightning_logs
```

Access at `http://localhost:6006`

For remote access via SSH tunnel:
```bash
# On your local machine
ssh -L 6006:localhost:6006 username@cluster.address
# Then open http://localhost:6006 in your browser
```

### SLURM Job Status

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View output (while running)
tail -f logs/slurm-<job_id>.out

# Cancel job
scancel <job_id>
```

## Checkpointing

### Automatic Checkpointing

Lightning automatically saves checkpoints:
- `last.ckpt` - Most recent checkpoint
- `epoch=XX-val_loss=Y.YY.ckpt` - Best checkpoints based on validation loss

### Resume Training

```bash
python esmretrain_lightning.py \
    --config config.json \
    --resume-from-checkpoint checkpoints_lightning/last.ckpt
```

SLURM: Edit the script and add the resume flag before submitting.

## Performance Optimization

### Batch Size Guidelines

Total batch size = `batch_size * devices * num_nodes * accumulate_grad_batches`

Example configurations:
- **Small model (35M)**: batch_size=16, accumulate=1, 4 GPUs → total=64
- **Medium model (150M)**: batch_size=8, accumulate=2, 4 GPUs → total=64
- **Large model (650M)**: batch_size=4, accumulate=4, 8 GPUs → total=128

### DataLoader Workers

Set `--num-workers` to number of CPU cores per GPU (usually 4-8):

```bash
--num-workers 8  # 8 CPU cores per GPU
```

### Gradient Accumulation

If you run out of memory, use gradient accumulation:

```bash
--batch-size 4 \
--accumulate-grad-batches 4  # Effective batch size = 16 per GPU
```

## Common Issues

### Out of Memory

Solutions:
1. Reduce batch size: `--batch-size 2`
2. Use gradient accumulation: `--accumulate-grad-batches 4`
3. Use mixed precision: `--precision bf16-mixed`
4. Reduce sequence length in dataset
5. Use FSDP strategy: `--strategy fsdp`

### NCCL Timeout

On multi-node training, increase timeout:

```bash
export NCCL_TIMEOUT=7200  # 2 hours
```

Add to SLURM script before training command.

### Different GPU Types

If nodes have different GPU types, be explicit about which GPUs to use:

```bash
#SBATCH --gres=gpu:a100:4  # Request 4 A100 GPUs specifically
```

## Example Workflows

### Quick Test Run

```bash
# Test on small data with 1 GPU
python esmretrain_lightning.py \
    --aa-fasta test_data_aa.fasta \
    --three-di-fasta test_data_3di.fasta \
    --hf-model facebook/esm2_t12_35M_UR50D \
    --batch-size 4 \
    --epochs 2 \
    --devices 1 \
    --out-dir test_checkpoints
```

### Full Training on SLURM

1. Prepare config file:
```bash
cp config_example.json config_my_run.json
# Edit config_my_run.json with your settings
```

2. Edit SLURM script:
```bash
nano slurm_train.sh
# Update: #SBATCH directives, module loads, config file path
```

3. Submit job:
```bash
sbatch slurm_train.sh
```

4. Monitor:
```bash
# Watch job status
watch -n 10 squeue -u $USER

# Watch output
tail -f logs/slurm-*.out

# Check TensorBoard (if on login node)
tensorboard --logdir=lightning_logs --port=6007
```

### Multi-Node Large Scale Training

```bash
# Edit slurm_train_multinode.sh
#SBATCH --nodes=8
#SBATCH --gres=gpu:8

# Submit
sbatch slurm_train_multinode.sh

# This will use 64 total GPUs (8 nodes × 8 GPUs)
```

## Validation

To enable validation during training:

```json
{
    "val_fasta_aa": "data/val_aa.fasta",
    "val_fasta_3di": "data/val_3di.fasta"
}
```

Or split training data:
```json
{
    "val_split": 0.1  // Use 10% for validation
}
```

## Best Practices

1. **Start small**: Test on 1 GPU before scaling to cluster
2. **Use checkpointing**: Always enable `--save-top-k 3`
3. **Monitor early**: Check TensorBoard after first epoch
4. **Validate config**: Test config locally before submitting expensive SLURM jobs
5. **Log everything**: Keep detailed notes in experiment names
6. **Resource requests**: Don't over-request resources (wastes queue time)
7. **Test datasets**: Validate on test data first before full training

## Torchrun Benefits

The scripts now use `torchrun` instead of directly launching Python, providing:

1. **Better fault tolerance**: Automatic restart on node failures
2. **Elastic training**: Dynamic scaling of workers (if configured)
3. **Proper environment setup**: Automatic RANK/WORLD_SIZE configuration
4. **Rendezvous backend**: Robust multi-node coordination with c10d
5. **Standard interface**: Works consistently across different cluster setups

### Torchrun Options Used

**Single-node** (`slurm_train.sh`):
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS
```

**Multi-node** (`slurm_train_multinode.sh`, `slurm_train_fsdp.sh`):
```bash
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
```

The rendezvous backend (c10d) handles node discovery and synchronization automatically.

## Additional Resources

- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [Distributed Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html)
