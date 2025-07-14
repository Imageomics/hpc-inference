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

The `hpc_inference` package's core functionality includes customized PyTorch datasets:

- **`ParquetImageDataset`** for image data stored as compressed binary columns across multiple large Parquet files
- **`ImageFolderDataset`** for image data stored in folders using open file formats such as `JPG`, `PNG`, `TIFF`, etc.

The package also comes with a suite of **ready-to-use** job scripts to perform efficient batch inference using pretrained models on HPCs.

## Use Cases

1. **Image Folder Dataset** - Process images from directory structures
2. **Parquet Dataset** - Handle compressed image data in Parquet format
3. **Large scale CLIP embedding** - Generate embeddings for massive datasets
4. **Large scale face detection** - Detect faces across large image collections
5. **Large scale animal detection** - Use MegaDetector for wildlife analysis
6. **Grid search profiling** - Optimize processing parameters

## Quick Links


## Acknowledgement

This project is a joint effort between the [Imageomics Institute](https://imageomics.osu.edu/) and the [ABC Global Center](https://www.biodiversityai.org/).
