# Multi-GPU Training for UniTS

This document describes how to use multi-GPU training with the UniTS codebase.

## Overview

The training system now supports three modes of multi-GPU training:

1. **Single GPU** (default): Uses a single GPU
2. **DataParallel (DP)**: Legacy multi-GPU training using `torch.nn.DataParallel`
3. **DistributedDataParallel (DDP)**: Modern, efficient multi-GPU training using `torch.nn.parallel.DistributedDataParallel`

## Configuration

### Single GPU Training
```bash
python train.py --config_file config.json
```

### DataParallel Training (Legacy)
Set `dp=True` in your configuration file and run:
```bash
python train.py --config_file config.json
```

### DistributedDataParallel Training (Recommended)
Use `torchrun` to launch distributed training:
```bash
torchrun --nproc_per_node=N train.py --config_file config.json
```

Where `N` is the number of GPUs to use.

## Key Features

### Automatic Detection
The code automatically detects distributed training environment variables:
- `RANK`: Process rank
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Local GPU index

### Efficient Data Loading
- Uses `DistributedSampler` for data partitioning
- Each GPU processes a unique subset of the data
- Automatic shuffling with epoch synchronization

### Model Wrapping
- Models are automatically wrapped with `DDP` for distributed training
- EMA (Exponential Moving Average) models are also wrapped when enabled
- Proper handling of model state dicts for saving/loading

### Logging and Checkpointing
- Only rank 0 process performs logging and checkpoint saving
- All processes synchronize at barriers
- Checkpoints contain the underlying model state (not DDP wrapper)

## Usage Examples

### Example 1: Training on 4 GPUs
```bash
torchrun --nproc_per_node=4 train.py --config_file config/train_hiegnn_unitslib_1GPU.json
```

### Example 2: Training on all available GPUs
```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py --config_file config.json
```

### Example 3: Training with specific master port
```bash
torchrun --nproc_per_node=2 --master_port=29500 train.py --config_file config.json
```

## Configuration Parameters

### Distributed Training Parameters
The system automatically handles distributed training. No additional configuration parameters are needed in the config file.

### Existing Parameters
- `dp` (boolean): Enable DataParallel mode (legacy, not recommended for new training)
- `batch_size` (int): Global batch size (will be divided across GPUs)

## Best Practices

1. **Use DDP over DataParallel**: DDP is more efficient and scales better
2. **Adjust batch size**: When using multiple GPUs, you may want to increase the global batch size
3. **Monitor GPU memory**: Each GPU will process `batch_size / num_gpus` samples
4. **Use pin_memory**: Enabled by default for better performance
5. **Check logs**: Only rank 0 produces detailed logs

## Testing

Run the test script to verify multi-GPU functionality:
```bash
python test_multi_gpu.py
```

## Troubleshooting

### Common Issues

1. **"Address already in use"**: Change the master port
   ```bash
   torchrun --nproc_per_node=2 --master_port=29501 train.py --config_file config.json
   ```

2. **GPU memory errors**: Reduce batch size or use gradient accumulation

3. **Slow training**: Ensure `pin_memory=True` and adjust `num_workers`

4. **Checkpoint loading**: When loading checkpoints for inference, use the underlying model state

### Debugging

To debug distributed training, you can run with a single process:
```bash
torchrun --nproc_per_node=1 train.py --config_file config.json
```

## Performance Considerations

- **DDP vs DataParallel**: DDP is generally 10-30% faster
- **Batch size scaling**: Linear scaling with number of GPUs
- **Communication overhead**: Minimal with NCCL backend
- **Memory usage**: Each GPU stores its own model copy

## Implementation Details

The main changes for multi-GPU support are in:

1. `train.py`: Distributed initialization, DDP wrapping, data loading
2. `units/train.py`: Model unwrapping for EMA and gradient clipping
3. Automatic rank-based logging and checkpointing

The system maintains backward compatibility with single-GPU and DataParallel training.