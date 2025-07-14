# ImageFolder Dataset Guide

This guide provides a comprehensive tutorial on using the `ImageFolderDataset` class from the HPC-Inference package.

## Overview

The `ImageFolderDataset` is designed for efficient processing of image datasets stored in folder structures. It provides optimized data loading capabilities for HPC environments with features like:

- Parallel data loading
- Memory-efficient processing
- Integration with PyTorch [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html)
- Support for various image formats (JPG, PNG, TIFF, etc.)

## Example Dataset

This guide demonstrates working with the [NEON Beetle dataset](https://huggingface.co/datasets/imageomics/2018-NEON-beetles), which contains high-resolution images of beetles collected by the National Ecological Observatory Network.

## Basic Usage

### Simple Setup

```python
from hpc_inference.datasets import ImageFolderDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = ImageFolderDataset(
    root_dir="/path/to/beetle/images",
    transform=transform,
    extensions=('.jpg', '.jpeg', '.png', '.tiff')
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    shuffle=False
)
```

### Processing Images

```python
import torch

# Process all images
for batch_idx, (images, paths) in enumerate(dataloader):
    # Move to GPU if available
    if torch.cuda.is_available():
        images = images.cuda()
    
    # Your inference code here
    with torch.no_grad():
        outputs = model(images)
    
    # Process outputs
    print(f"Batch {batch_idx}: Processed {len(images)} images")
    for i, path in enumerate(paths):
        print(f"  {path}: {outputs[i].shape}")
```

## Advanced Features

### Multi-Model Preprocessing

**TODO**

### Validation and Error Handling

**TODO**

## Distributed Processing

### Multi-GPU Setup

**TODO**

## Performance Optimization

### Memory Management

```python
# Optimize memory usage
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2         # Prefetch batches per worker
)
```

### Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.4f} seconds")

# Profile data loading
with timer("Data loading"):
    for batch_idx, (images, paths) in enumerate(dataloader):
        if batch_idx >= 10:  # Profile first 10 batches
            break
        
        with timer(f"Batch {batch_idx}"):
            # Your processing code
            pass
```

## Best Practices

### 1. Choose Appropriate Batch Size
- Start with batch size 16 and adjust based on GPU memory
- Larger batches generally improve GPU utilization
- Monitor memory usage to avoid OOM errors

### 2. Optimize Number of Workers
- Start with `num_workers = num_gpus`
- Monitor CPU usage to find optimal value
- Too many workers can cause overhead

### 3. Use Pin Memory
- Enable `pin_memory=True` for GPU processing
- Speeds up data transfer to GPU

### 4. Handle Corrupted Files
- Always validate your dataset before processing to avoid unexpected job crashing
- Implement error handling in your data loading pipeline
- Log corrupted files for investigation

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Reduce number of workers
   - Use gradient checkpointing

2. **Slow Data Loading**
   - Increase number of workers
   - Use faster storage (SSD vs HDD)
   - Optimize image formats

3. **CUDA Errors**
   - Ensure CUDA compatibility
   - Check GPU memory usage
   - Verify data types and shapes
