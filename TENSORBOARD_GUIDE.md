# TensorBoard Logging Guide

## Overview
The `esmretrain.py` script now includes comprehensive TensorBoard logging to monitor training progress in real-time.

## What Gets Logged

### Per-Step Metrics
- **Loss/train_step**: Raw loss value for each training step
- **Loss/train_per_residue_step**: Loss normalized per residue for each step

### Running Average Metrics (every N steps)
- **Loss/train_running_avg**: Running average loss per residue
- **Training/tokens_processed**: Number of tokens processed

### Epoch-Level Metrics
- **Loss/train_epoch**: Average loss per residue for the entire epoch
- **Training/epoch_tokens**: Total tokens processed in the epoch

## Usage

### 1. Start Training
Run your training script as usual:
```bash
python esmretrain.py --config config.json
```

Or with specific TensorBoard log directory:
```bash
python esmretrain.py --config config.json --tensorboard-log-dir my_logs
```

### 2. Launch TensorBoard
While training is running (or after), in a new terminal:
```bash
tensorboard --logdir=tensorboard_logs
```

Or with a custom port:
```bash
tensorboard --logdir=tensorboard_logs --port=6007
```

### 3. View Results
Open your browser and navigate to:
- Default: http://localhost:6006
- Custom port: http://localhost:PORT

## Remote Access (SSH Tunneling)

If training on a remote server, use SSH port forwarding:

```bash
# On your local machine
ssh -L 6006:localhost:6006 user@remote-server
```

Then access http://localhost:6006 in your local browser.

## TensorBoard Features

### Scalars Tab
View all logged metrics:
- Loss curves over time
- Token processing statistics
- Compare different training runs

### Time Series Smoothing
Use the smoothing slider in TensorBoard to reduce noise in the loss curves.

### Run Comparison
All training runs are saved with timestamps. You can:
- Compare different hyperparameter settings
- Track improvements across experiments
- Identify the best performing model

## Log Directory Structure

```
tensorboard_logs/
├── run_20241203_143022/  # Timestamp: YYYYMMDD_HHMMSS
│   └── events.out.tfevents.*
├── run_20241203_150135/
│   └── events.out.tfevents.*
└── ...
```

## Tips

1. **Keep TensorBoard Running**: You can start TensorBoard before training begins - it will automatically update as new data is logged.

2. **Multiple Experiments**: Each training run creates a new subdirectory, so you can compare different experiments side-by-side.

3. **Disk Space**: TensorBoard logs are lightweight, but clean up old runs if disk space is a concern.

4. **Real-time Monitoring**: Refresh the TensorBoard page (click the refresh button or press R) to see the latest updates.

## Troubleshooting

### Port Already in Use
If port 6006 is occupied:
```bash
tensorboard --logdir=tensorboard_logs --port=6007
```

### Can't Connect
- Verify TensorBoard is running
- Check firewall settings
- Ensure correct port forwarding for remote access

### No Data Showing
- Check that training has started
- Verify the log directory path
- Click the refresh button in TensorBoard
