# Torchrun Quick Reference

This guide provides quick reference for using torchrun with ESM3di training scripts.

## What is Torchrun?

`torchrun` is PyTorch's official distributed training launcher that replaces `torch.distributed.launch`. It provides:
- Automatic environment variable setup (RANK, WORLD_SIZE, etc.)
- Fault tolerance and automatic restarts
- Elastic training capabilities
- Better multi-node coordination

## Local Training

### Single GPU (No torchrun needed)
```bash
python esmretrain_lightning.py --config config.json --devices 1
```

### Multi-GPU on Single Machine
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    esmretrain_lightning.py \
    --config config.json \
    --devices 4
```

Or use the convenience script:
```bash
./launch_local_torchrun.sh -n 4 -c config.json
```

## SLURM Cluster Training

### Single Node Multi-GPU
```bash
sbatch slurm_train.sh
```

The script uses:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS
```

### Multi-Node Training
```bash
sbatch slurm_train_multinode.sh
```

The script uses:
```bash
srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
```

## Torchrun Arguments Explained

### Basic Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--standalone` | Single-node mode (no rendezvous needed) | Use for single machine |
| `--nnodes` | Number of nodes | `--nnodes=4` |
| `--nproc_per_node` | Number of processes (GPUs) per node | `--nproc_per_node=8` |

### Multi-Node Coordination

| Argument | Description | Example |
|----------|-------------|---------|
| `--rdzv_backend` | Rendezvous backend (usually `c10d`) | `--rdzv_backend=c10d` |
| `--rdzv_endpoint` | Master node address:port | `--rdzv_endpoint=node01:29500` |
| `--rdzv_id` | Unique job ID | `--rdzv_id=$SLURM_JOB_ID` |

### Advanced Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--max_restarts` | Max number of worker restarts | `--max_restarts=3` |
| `--monitor_interval` | Monitoring interval (seconds) | `--monitor_interval=5` |
| `--start_method` | Process start method | `--start_method=spawn` |
| `--log_dir` | Directory for logs | `--log_dir=./torchrun_logs` |

## Environment Variables Set by Torchrun

Torchrun automatically sets these variables for each process:

- `RANK` - Global rank of the process (0 to world_size-1)
- `LOCAL_RANK` - Local rank on the node (0 to nproc_per_node-1)
- `WORLD_SIZE` - Total number of processes
- `MASTER_ADDR` - Address of the master node
- `MASTER_PORT` - Port of the master node

PyTorch Lightning automatically reads these variables.

## Common Patterns

### Test Locally First
```bash
# Test with 1 GPU
python esmretrain_lightning.py --config config.json --devices 1 --epochs 1

# Test with 2 GPUs
./launch_local_torchrun.sh -n 2 -c config.json
```

### Scale to SLURM
```bash
# Edit slurm_train.sh to set resources
nano slurm_train.sh

# Submit
sbatch slurm_train.sh
```

### Multi-Node on SLURM
```bash
# Edit for more nodes
nano slurm_train_multinode.sh
# Change: #SBATCH --nodes=8

# Submit
sbatch slurm_train_multinode.sh
```

## Debugging Torchrun

### Enable Verbose Logging
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --log_dir=./torchrun_logs \
    esmretrain_lightning.py --config config.json
```

### Check Environment Variables
Add to your script to see what torchrun sets:
```python
import os
print(f"RANK: {os.environ.get('RANK')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
```

### Common Issues

**"Address already in use"**
- Change master port: `--master_port=29501`
- Or let it choose automatically (default)

**"Connection timeout"**
- Check firewall rules between nodes
- Verify `MASTER_ADDR` is reachable
- Increase timeout: `export NCCL_TIMEOUT=7200`

**"NCCL error"**
- Set NCCL debug: `export NCCL_DEBUG=INFO`
- Check GPU compatibility across nodes
- Verify NCCL installation

## Torchrun vs Alternatives

### Torchrun (Recommended)
```bash
torchrun --standalone --nproc_per_node=4 train.py
```
✓ Fault tolerance  
✓ Automatic env setup  
✓ Modern, actively maintained  

### Python -m torch.distributed.launch (Deprecated)
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py
```
✗ Deprecated in PyTorch 1.10+  
✗ Use torchrun instead  

### Direct Python with DDP (Manual)
```bash
python train.py
```
✗ Manual rank/world_size setup  
✗ No fault tolerance  
✗ More boilerplate code  

## Integration with Lightning

PyTorch Lightning automatically detects torchrun environment:

```python
# Lightning automatically uses RANK, LOCAL_RANK, WORLD_SIZE
trainer = pl.Trainer(
    devices=4,      # Number of GPUs
    num_nodes=2,    # Number of nodes
    strategy="ddp", # Distributed strategy
)
```

No manual distributed setup needed - Lightning handles everything!

## Quick Commands

```bash
# Local 2 GPU test
./launch_local_torchrun.sh -n 2 -c config.json

# Local 4 GPU training
./launch_local_torchrun.sh -n 4 -c config.json

# SLURM single node
sbatch slurm_train.sh

# SLURM multi-node
sbatch slurm_train_multinode.sh

# SLURM with FSDP
sbatch slurm_train_fsdp.sh

# Check SLURM job
squeue -u $USER
tail -f logs/slurm-*.out

# View tensorboard
tensorboard --logdir=lightning_logs
```

## Resources

- [Torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Lightning Multi-GPU Training](https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html)
