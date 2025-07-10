"""High-performance dataset loaders for image data."""

from .parquet_dataset import ParquetImageDataset, multi_model_collate
from .image_folder_dataset import ImageFolderDataset

# Check if ParquetEmbeddingDataset exists in parquet_dataset.py
try:
    from .parquet_dataset import ParquetEmbeddingDataset
    __all__ = [
        "ParquetImageDataset",
        "ParquetEmbeddingDataset",
        "ImageFolderDataset",
        "multi_model_collate",
    ]
except ImportError:
    __all__ = [
        "ParquetImageDataset",
        "ImageFolderDataset", 
        "multi_model_collate",
    ]
