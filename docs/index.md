# HPC-Inference

<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <img src="imgs/Imageomics_logo_butterfly.png" alt="Imageomics Logo" width="100" style="margin-right: 20px;" />
  <img src="imgs/full_ABC_logo_for_web_and_digital.png" alt="ABC Logo" width="100" style="margin-right: 20px;" />
  <div>
    <h2 style="margin: 0;">HPC-Inference</h2>
    <p style="margin: 0; font-size: 1.1em; color: #666;">Batch inference solution for large-scale image datasets on HPC</p>
  </div>
</div>

## About

**Problem:** Many batch inference workflows waste GPU resources due to I/O bottlenecks and sequential processing, leading to poor GPU utilization and longer processing times.

### Key Bottlenecks
- Slow sequential large file loading (Disk → RAM)
- Single-threaded image preprocessing
- Data transfer delays (CPU ↔ GPU)
- GPU idle time waiting for data
- Sequential output writing

### HPC-Inference Solutions
- **Parallel data loading:** Eliminates disk I/O bottlenecks with optimized dataset loaders
- **Asynchronous preprocessing:** Keeps GPUs fed with continuous data queues  
- **SLURM integration:** Deploy seamlessly on HPC clusters
- **Multi-GPU distribution:** Scales across HPC nodes for maximum throughput
- **Resource profiling:** Logs timing metrics and CPU/GPU usage rates to help optimize your configuration

## Core Features

### Iterative PyTorch Dataset

The `hpc_inference` package's core functionality includes customized PyTorch datasets:

- **`ParquetImageDataset`** for image data stored as compressed binary columns across multiple large Parquet files
- **`ImageFolderDataset`** for image data stored in folders using open file formats such as `JPG`, `PNG`, `TIFF`, etc.
- **`HDF5ImageDataset`** for streaming image data from HDF5 files with built-in distributed processing support

```python
from pathlib import Path
from hpc_inference.datasets import ParquetImageDataset, ImageFolderDataset, HDF5ImageDataset

# Parquet dataset
parquet_files = list(Path("/path/to/data").glob("*.parquet"))
parquet_dataset = ParquetImageDataset(parquet_files, preprocess=preprocess)

# Image folder dataset
image_folder_dataset = ImageFolderDataset("/path/to/images", preprocess=preprocess)

# HDF5 dataset
hdf5_files = list(Path("/path/to/data").glob("*.h5"))
hdf5_dataset = HDF5ImageDataset(hdf5_files, preprocess=preprocess)
```

Use with PyTorch DataLoader for batch processing:

```python
from torch.utils.data import DataLoader

loader = DataLoader(hdf5_dataset, batch_size=32, num_workers=4)
for batch_ids, batch_tensors in loader:
    # Process batch
    embeddings = model(batch_tensors.to(device))
```

### Batch Inference Job Scripts

Ready-to-use inference scripts for common tasks:

| Task | Command Module | Description |
|------|----------------|-------------|
| CLIP Embedding | `hpc_inference.inference.embed.open_clip_embed` | Generate embeddings with OpenCLIP models |
| Face Detection | `hpc_inference.inference.detection.face_detect` | Detect faces using YOLO-based models |
| Animal Detection | `hpc_inference.inference.detection.animal_detect` | Wildlife detection with MegaDetector |

#### Basic Usage

```bash
python -m hpc_inference.inference.embed.open_clip_embed \
    /path/to/input \
    /path/to/output \
    --input_type parquet \
    --model_name ViT-B-32 \
    --pretrained openai \
    --batch_size 32 \
    --num_workers 8
```

#### Using Config Files (Recommended)

Config files are recommended for production workflows as they:

- Keep parameters organized and reproducible
- Allow easy sharing and version control
- Reduce command-line complexity

```bash
python -m hpc_inference.inference.embed.open_clip_embed \
    /path/to/input /path/to/output \
    --input_type parquet --config config.yaml
```

### Templates

- **Config templates** → [`configs/`](https://github.com/Imageomics/hpc-inference/tree/main/configs) (e.g., `config_embed_parquet_template.yaml`, `config_embed_hdf5_template.yaml`)
- **SLURM job templates** → [`scripts/`](https://github.com/Imageomics/hpc-inference/tree/main/scripts) (e.g., `open_clip_embed_parquet_template.slurm`)

#### SLURM Integration

The datasets automatically detect `SLURM_PROCID` and `SLURM_NTASKS` to distribute files across ranks. Use `srun` in your SLURM scripts to scale across multiple nodes.


## Acknowledgement

This project is a joint effort between the [Imageomics Institute](https://imageomics.osu.edu/) and the [ABC Global Center](https://www.biodiversityai.org/).
