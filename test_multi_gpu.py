#!/usr/bin/env python3
"""
Test script for multi-GPU training functionality.
This script demonstrates how to run training with multiple GPUs.
"""

import os
import subprocess
import sys
import json
import tempfile

def create_test_config():
    """Create a minimal test configuration for multi-GPU training"""
    config = {
        "dataset_path": "dataset",
        "name_regrex": "dataset_0_*.npy",
        "dataset_type": 2,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "seed": 42,
        "batch_size": 4,
        "test_reduce_ratio": 1,
        "num_workers": 0,
        "n_epochs": 2,
        "learning_rate": 0.001,
        "optimizer_type": "adamw",
        "scheduler_type": "steplr",
        "steplr_step_size": 10,
        "steplr_gamma": 0.5,
        "dp": False,  # Use DDP instead of DataParallel
        "ema_decay": 0.0,
        "clip_grad": False,
        "n_report_steps": 10,
        "n_keep_ckpt": 2,
        "break_train_epoch": False,
        "test_epochs": 1,
        "save_path": "test_output",
        "tag": "test_multi_gpu",
        "dynamic_type": "egnn",
        "n_dims": 3,
        "in_node_nf": 10,
        "context_node_nf": 128,
        "rct_cent_node_nf": 32,
        "hidden_nf": 32,
        "n_layers": 2,
        "attention": False,
        "condition_time": True,
        "time_embed_dim": 1,
        "tanh": False,
        "norm_constant": 1.0,
        "inv_sublayers": 1,
        "sin_embedding": False,
        "normalization_factor": 100,
        "aggregation_method": "sum",
        "scale_range": 1.0,
        "add_angle_info": False,
        "add_fpfh": False,
        "diffusion_steps": 10,
        "diffusion_parametrization": "eps",
        "diffusion_noise_schedule": "learned",
        "diffusion_noise_precision": 1e-4,
        "diffusion_loss_type": "l2",
        "norm_values": [1.0, 1.0],
        "norm_biases": [None, 0.0],
        "enc_num_layers": 2,
        "time_sample_method": "uniform",
        "time_sample_power": 4.0,
        "min_sample_power": 2.0,
        "max_sample_power": 6.0,
        "degree_as_continuous": True,
        "dynamic_context": False,
        "dynamic_context_temperature": 1.0,
        "loss_calc": "all",
        "tot_x_mae": True,
        "mol_encoder_type": "gcn",
        "use_context": False,
        "enc_gnn_aggr": "add",
        "enc_bond_feat_red": "mean",
        "enc_JK": "last",
        "enc_drop_ratio": 0,
        "enc_node_readout": "sum",
        "use_rct_cent": False,
        "rct_cent_encoder_type": "gcn",
        "rct_cent_readout": "mean",
        "focus_reaction_center": False,
        "rc_loss_weight": 2.0,
        "rc_distance_weight": 3.0,
        "rc_angle_weight": 2.0,
        "use_init_coords": False,
        "fix_step_from_init": 500,
        "link_rc": False,
        "data_enhance": False,
        "specific_test": False,
        "test_name_regrex": "",
        "data_truncated": 10,  # Small dataset for testing
        "ode_regularization": 0.0,
        "warmup_step": 1000,
        "equif_sphere_channels": 64,
        "equif_attn_hidden_channels": 64,
        "equif_num_heads": 4,
        "equif_attn_alpha_channels": 32,
        "equif_attn_value_channels": 16,
        "equif_ffn_hidden_channels": 128,
        "lmax": 2,
        "mmax": 2,
        "equif_add_node_feat": False
    }
    return config

def test_single_gpu():
    """Test single GPU training"""
    print("Testing single GPU training...")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = create_test_config()
        json.dump(config, f)
        config_file = f.name
    
    try:
        # Run training with single GPU
        cmd = [sys.executable, "train.py", "--config_file", config_file]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Single GPU training test passed!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("✗ Single GPU training test failed!")
            print("Error:", result.stderr)
            return False
    finally:
        # Clean up
        os.unlink(config_file)
    
    return True

def test_multi_gpu_ddp():
    """Test multi-GPU training with DDP"""
    print("\nTesting multi-GPU training with DDP...")
    
    # Check if multiple GPUs are available
    import torch
    if torch.cuda.device_count() < 2:
        print("⚠ Skipping multi-GPU test: Need at least 2 GPUs")
        return True
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = create_test_config()
        config["batch_size"] = 2  # Smaller batch for multi-GPU test
        json.dump(config, f)
        config_file = f.name
    
    try:
        # Run training with DDP
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=12355",
            "train.py",
            "--config_file", config_file
        ]
        print(f"Running command: {' '.join(cmd)}")
        
        # Set environment variables for DDP
        env = os.environ.copy()
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print("✓ Multi-GPU DDP training test passed!")
            # Look for DDP-related messages in output
            if "Training using 2 GPUs with DistributedDataParallel" in result.stdout:
                print("✓ DDP initialization successful")
            else:
                print("⚠ DDP initialization message not found")
        else:
            print("✗ Multi-GPU DDP training test failed!")
            print("Error:", result.stderr)
            return False
    finally:
        # Clean up
        os.unlink(config_file)
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Multi-GPU Training Functionality")
    print("=" * 60)
    
    # Test 1: Single GPU
    if not test_single_gpu():
        print("\n❌ Single GPU test failed. Aborting.")
        return 1
    
    # Test 2: Multi-GPU DDP
    if not test_multi_gpu_ddp():
        print("\n❌ Multi-GPU DDP test failed.")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    # Print usage instructions
    print("\nUsage instructions for multi-GPU training:")
    print("1. For single GPU training:")
    print("   python train.py --config_file config.json")
    print("\n2. For multi-GPU training with DDP (recommended):")
    print("   torchrun --nproc_per_node=N train.py --config_file config.json")
    print("\n3. For multi-GPU training with DataParallel (legacy):")
    print("   Set dp=True in config and run: python train.py --config_file config.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())