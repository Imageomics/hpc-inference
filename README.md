# HPC-Inference

A high-performance computing solution for efficient batch inference on large-scale image datasets.

## Overview

`HPC-Inference` enables users to perform batch inference on image datasets with maximum efficiency and GPU utilization. The system provides optimized data loading, distributed processing capabilities, and SLURM job management for HPC environments.

Many batch inference workflows suffer from poor GPU utilization due to I/O bottlenecks and sequential processing. This leads to wasted computational resources and longer processing times.

**Standard inference pipeline bottlenecks:**
1. **Disk → RAM**: Slow sequential file loading
2. **CPU preprocessing**: Single-threaded image processing  
3. **CPU → GPU transfer**: Data transfer delays
4. **GPU inference**: GPU idle time waiting for data
5. **GPU → CPU transfer**: Result transfer overhead
6. **CPU I/O**: Sequential output writing

**Solution:**
- **Parallel data loading** with optimized dataset loaders
- **Asynchronous preprocessing** to maintain data queues
- **Multi-GPU distributed processing** across nodes
- **Resource profiling** to optimize configurations
- **SLURM integration** for seamless HPC deployment

## Features

### High-Performance Data Loading
- **Custom dataset loaders** for Parquet and image folder formats
- **Parallel I/O operations** to eliminate data loading bottlenecks
- **Memory-efficient streaming** for large Parquet datasets that may cause OOM

### Pre-trained Model Support
- **OpenCLIP models** for vision-language embedding generation
- **Extensible architecture** for adding custom models

### Distributed Computing
- **Multi-node, multi-GPU** processing with automatic load balancing
- **SLURM job templates** for easy cluster deployment
- **Configurable resource allocation** (CPU cores, memory, GPU count)

### Performance Monitoring
- **Compute resource profiling** (GPU utilization, memory usage)
- **Throughput metrics** and processing statistics

## Installation

### Prerequisites
- Python >= 3.8
- CUDA-capable GPU (for inference)
- PyTorch >= 1.11.0

### Quick Installation Guide

| Use Case | Installation Command | What's Included |
|----------|---------------------|-----------------|
| **Dataset Processing** | `pip install -e .` | Customized Torch Datasets |
| **OpenCLIP Inference** | `pip install -e ".[openclip]"` | Core + OpenCLIP embedding support|
| **Everything** | `pip install -e ".[all]"` | All Features |
| **Development** | `pip install -e ".[dev]"` | Core + Dev Tools |

### Installation Options

`HPC-Inference` uses a modular dependency structure to minimize installation requirements based on your use case.

#### Core Package (Recommended Start)
```bash
# Clone the repository
git clone https://github.com/Imageomics/hpc-inference.git
cd hpc-inference

# Standard installation - includes datasets, utilities, and profiling
pip install -e .
```

**What's included:**
- ✅ High-performance dataset loaders (`ParquetImageDataset`, `ImageFolderDataset`)
- ✅ Utility functions (`load_config`, `decode_image`, `save_emb_to_parquet`)
- ✅ Performance profiling and monitoring
- ✅ Resource usage tracking and visualization

**Use cases:** Custom inference pipelines, dataset processing, performance optimization

#### OpenCLIP Inference Support
```bash
# Core package + OpenCLIP embedding generation
pip install -e ".[openclip]"
```

**Additional features:**
- ✅ OpenCLIP model support for vision-language embeddings
- ✅ Ready-to-use embedding generation scripts
- ✅ SLURM job templates for HPC deployment

**Use cases:** Large-scale embedding generation, vision-language research

#### Complete Installation
```bash
# All current and future model support
pip install -e ".[all]"
```

**Use cases:** Research groups wanting all capabilities

#### Development Installation
```bash
# For contributors and package developers
pip install -e ".[dev]"
```

**Additional tools:** pytest, black, ruff, mypy, pre-commit hooks

### Verify Installation

Check what features are available in your installation:

```python
import hpc_inference

# Print installation status
hpc_inference.print_installation_guide()

# Or check programmatically
features = hpc_inference.list_available_features()
print(f"Available features: {features}")
```

**Example output:**
```
HPC-Inference Installation Status:
================================
✅ Core (datasets, utils, profiling): Available
❌ OpenCLIP: Missing

To enable OpenCLIP: pip install 'hpc-inference[openclip]'
To install everything: pip install 'hpc-inference[all]'
```

### Upgrading

```bash
# Upgrade to latest version
cd hpc-inference
git pull
pip install -e ".[your-extras]"  # e.g., .[openclip] or .[all]
```

## Quick Start

### Use Case 1: Custom Dataset Processing

**Installation required:** Core package (`pip install -e .`)

For users who want to leverage the high-performance dataset loaders for their own inference or training pipelines.

```python
from hpc_inference.datasets import ParquetImageDataset
from hpc_inference.utils import decode_image, load_config
from torch.utils.data import DataLoader
from torchvision import transforms

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
parquet_files = ["/path/to/file1.parquet", "/path/to/file2.parquet"]
dataset = ParquetImageDataset(
    parquet_files=parquet_files,
    col_uuid="uuid",
    rank=0,                    # Current process rank
    world_size=1,              # Total number of processes
    decode_fn=decode_image,
    preprocess=preprocess,
    read_batch_size=128,       # Parquet reading batch size
    read_columns=["uuid", "image"],
    evenly_distribute=True     # Load balance files by size
)

# Create DataLoader with optimized settings
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    prefetch_factor=16,
    pin_memory=True,
    shuffle=False
)

# Process your data
for uuids, images in dataloader:
    # Your custom inference logic here
    print(f"Processing batch: {len(uuids)} images")
```

### Use Case 2: OpenCLIP Batch Embedding Generation

**Installation required:** OpenCLIP support (`pip install -e ".[openclip]"`)

For users who want to generate embeddings from images using OpenCLIP models at scale. 

#### Step 1: Prepare Configuration
```bash
# Copy template and customize
cp configs/config_embed_template.yaml configs/my_bioclip_job.yaml
```

Edit your configuration:
```yaml
# configs/my_bioclip_job.yaml
models:
  bioclip:
    name: hf-hub:imageomics/bioclip-2
    pretrained: null

batch_size: 32              # Adjust based on GPU memory  
num_workers: 16             # Set to available CPU cores
prefetch_factor: 32         # Higher = better I/O performance
read_batch_size: 128        # Parquet batch reading size

read_columns:
  - uuid
  - image

max_rows_per_file: 500000   # Embeddings per output file
out_prefix: bioclip_embeds  # Output file prefix
```

#### Step 2: Single Node Execution
```bash
# Direct Python execution
python -m hpc_inference.inference.embed.open_clip_embed \
    configs/my_bioclip_job.yaml \
    /path/to/parquet/files \
    /path/to/output/directory
```

#### Step 3: HPC/SLURM Execution
```bash
# Copy and customize SLURM template
cp scripts/embed/open_clip_embed_template.slurm my_embedding_job.slurm

# Submit the job
sbatch my_embedding_job.slurm

# Monitor progress
squeue -u $USER
tail -f logs/embed_*.out
```

TODO: add documentation for SLURM

### Check output

```
output_directory/
├── embeddings/
│   ├── rank_0/
│   │   ├── embeddings_rank_0_0.parquet
│   │   ├── embeddings_rank_0_1.parquet
│   │   └── processed_files_rank0_*.log
│   └── rank_1/
│       └── ...
└── profile_results/
    ├── rank_0/
    │   ├── batch_stats.csv              # Timing per batch
    │   ├── usage_log.csv                # Resource utilization
    │   ├── computing_specs.json         # Hardware specs
    │   └── *.png                        # Performance plots
    └── rank_1/
        └── ...
```

## Project Structure

```
hpc-inference/
├── src/hpc_inference/     # Main package
│   ├── datasets/          # High-performance dataset loaders
│   ├── inference/         # Model inference modules
│   │   └── embed/         # Embedding generation
│   └── utils/             # Utility functions
├── configs/               # Configuration templates
├── scripts/               # SLURM job templates
├── tests/                 # Package tests
└── docs/                  # Documentation
```

## Output Structure




## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Installation Issues

**ImportError: No module named 'open_clip'**
```bash
# Install OpenCLIP support
pip install -e ".[openclip]"
```

**ImportError: No module named 'hpc_inference'**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**ModuleNotFoundError when running SLURM jobs**
```bash
# Check your SLURM script has proper environment activation
module load cuda/12.4.1
source /path/to/your/venv/bin/activate

# Verify package is installed in the activated environment
pip list | grep hpc-inference
```

### Feature Availability Issues

**Check what's installed:**
```python
import hpc_inference
hpc_inference.print_installation_guide()
```

**Missing features:**
- **No profiling plots:** Core installation includes all profiling features
- **OpenCLIP not available:** Install with `pip install -e ".[openclip]"`

### Performance Issues

**Out of GPU Memory:**
- Reduce `batch_size` in config
- Reduce `prefetch_factor`

**Slow I/O Performance:**
- Increase `num_workers`
- Increase `read_batch_size`
- Use SSD storage for input data

**Uneven Load Distribution:**
- Set `evenly_distribute: true` in dataset initialization
- Check file size distribution in your data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

