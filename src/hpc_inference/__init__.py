"""HPC-Inference: High-performance batch inference for image datasets."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@domain.edu"

# Import main classes for easy access
from .datasets import ParquetImageDataset, ImageFolderDataset
from .utils.common import decode_image, save_emb_to_parquet, format_time, load_config
from .utils import profiling

__all__ = [
    "__version__",
    "ParquetImageDataset",
    "ImageFolderDataset", 
    "decode_image",
    "save_emb_to_parquet",
    "format_time",
    "load_config",
    "profiling",
]
