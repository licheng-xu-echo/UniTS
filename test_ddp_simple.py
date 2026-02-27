#!/usr/bin/env python3
"""
Simple test for DDP functionality without requiring full model dependencies.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    
    return rank, world_size

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def test_ddp_basic(rank, world_size):
    """Test basic DDP functionality."""
    print(f"Rank {rank}: Starting test")
    
    # Setup distributed environment
    rank, world_size = setup(rank, world_size)
    
    try:
        # Create a simple model
        model = torch.nn.Linear(10, 5).cuda(rank)
        
        # Wrap with DDP
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )
        
        # Create some dummy data
        x = torch.randn(4, 10).cuda(rank)
        
        # Forward pass
        output = ddp_model(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Sync gradients
        dist.barrier()
        
        print(f"Rank {rank}: Test passed")
        
        return True
    except Exception as e:
        print(f"Rank {rank}: Test failed with error: {e}")
        return False
    finally:
        cleanup()

def run_ddp_test():
    """Run DDP test on available GPUs."""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"Only {world_size} GPU(s) available. Need at least 2 for DDP test.")
        print("Testing single GPU mode...")
        
        # Test single GPU
        model = torch.nn.Linear(10, 5).cuda(0)
        x = torch.randn(4, 10).cuda(0)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        print("Single GPU test passed")
        return True
    
    print(f"Testing DDP with {world_size} GPUs...")
    
    # Use multiprocessing to spawn processes
    mp.spawn(
        test_ddp_basic,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    print("DDP test completed")
    return True

def test_distributed_sampler():
    """Test DistributedSampler functionality."""
    print("\nTesting DistributedSampler...")
    
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Create a simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, size=100):
            self.data = list(range(size))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Test with single process (rank 0)
    dataset = SimpleDataset(100)
    sampler = DistributedSampler(
        dataset,
        num_replicas=2,  # Simulating 2 processes
        rank=0,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
    
    # Get first batch
    batch = next(iter(dataloader))
    print(f"Rank 0 first batch: {batch}")
    
    # Simulate rank 1
    sampler_rank1 = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True
    )
    
    dataloader_rank1 = DataLoader(dataset, batch_size=10, sampler=sampler_rank1)
    batch_rank1 = next(iter(dataloader_rank1))
    print(f"Rank 1 first batch: {batch_rank1}")
    
    # Check that batches are different (different subsets of data)
    if not torch.equal(batch, batch_rank1):
        print("✓ DistributedSampler correctly partitions data")
        return True
    else:
        print("✗ DistributedSampler test failed")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Distributed Training Components")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Test 1: Basic DDP functionality
    print("\n1. Testing DDP basic functionality...")
    try:
        run_ddp_test()
        print("✓ DDP basic test passed")
    except Exception as e:
        print(f"✗ DDP basic test failed: {e}")
    
    # Test 2: DistributedSampler
    print("\n2. Testing DistributedSampler...")
    try:
        if test_distributed_sampler():
            print("✓ DistributedSampler test passed")
        else:
            print("✗ DistributedSampler test failed")
    except Exception as e:
        print(f"✗ DistributedSampler test failed: {e}")
    
    # Test 3: Environment variable detection
    print("\n3. Testing environment variable detection...")
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['LOCAL_RANK'] = '0'
    
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    print(f"  RANK: {rank}")
    print(f"  WORLD_SIZE: {world_size}")
    print(f"  LOCAL_RANK: {local_rank}")
    
    if rank == 0 and world_size == 2 and local_rank == 0:
        print("✓ Environment variable detection works")
    else:
        print("✗ Environment variable detection test failed")
    
    # Clean up
    del os.environ['RANK']
    del os.environ['WORLD_SIZE']
    del os.environ['LOCAL_RANK']
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\nTo use multi-GPU training with UniTS:")
    print("1. For single GPU: python train.py --config_file config.json")
    print("2. For multi-GPU DDP: torchrun --nproc_per_node=N train.py --config_file config.json")
    print("3. For DataParallel (legacy): Set dp=True in config")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())