
<table width="100%">
  <tr>
    <td align="left" width="120">
      <img src="docs/imgs/Imageomics_logo_butterfly.png" alt="OpenCut Logo" width="100" />
    </td>
    <td align="right">
      <h1>HPC-Inference</h1>
      <h4 style="margin-top: -10px;">Batch inference solution for large-scale image datasets on HPC.</h3>
    </td>
  </tr>
</table>

## About

**Problem:** Many batch inference workflows waste GPU resources due to I/O bottlenecks and sequential processing, leading to poor GPU utilization and longer processing times.

**Key Bottlenecks:**
- Slow sequential large file loading (Disk → RAM)
- Single-threaded image preprocessing
- Data transfer delays (CPU ↔ GPU)
- GPU idle time waiting for data
- Sequential output writing

**`HPC-Inference` solves this by:**

- **Parallel data loading:** Eliminates disk I/O bottlenecks with optimized dataset loaders
- **Asynchronous preprocessing:** Keeps GPUs fed with continuous data queues  
- **SLURM integration:** Deploy seamlessly on HPC clusters
- **Multi-GPU distribution:** Scales across HPC nodes for maximum throughput
- **Resource profiling:** Logs timing metrics and CPU/GPU usage rates to help optimize your configuration


## Getting Started

### Setup with uv

The `hpc_inference` package's core functionality is the customized PyTorch datasets:
- `ParquetImageDataset` for image data stored as compressed binary columns across multiple large Parquet files.
- `ImageFolderDataset` for image data stored in a folder using open file format such as `JPG`, `PNG`, `TIFF`, etc. 

```bash
# Clone Repo
git clone https://github.com/Imageomics/hpc-inference.git
cd hpc-inference

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install package
uv venv hpc-inference-env
source hpc-inference-env/bin/activate

# Install base package
uv pip install -e .
```

**Verify installation:**

```bash
# Test the installation
python -c "import hpc_inference; print('✓ HPC-Inference installed successfully')"
```

The package also comes with a suite of **ready-to-use** job scripts to perform efficient batch inference using pretrained models on HPCs. To use these scripts, you'll need to install additional dependencies based on use cases:

``` bash
# Check installation status and available features
python -c "from hpc_inference import print_installation_guide; print_installation_guide()"

uv pip install -e ".[openclip]"     # For CLIP embedding
uv pip install -e ".[detection]"    # For face/animal detection  
uv pip install -e ".[all]"          # Install dependency for all use cases
```

### Use Cases Guide

Use case 1: 
- Image Folder Dataset
- Parquet Dataset
- Self-specified task

Use case 2:
- Large scale CLIP embedding

Use case 3:
- Large scale face detection

Use case 4:
- Large scale animal detection using megadetector

Use case 5:
- Grid search profiling

## Project Structure
