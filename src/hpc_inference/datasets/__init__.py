"""High-performance dataset loaders for image data."""

from .parquet_dataset import ParquetImageDataset
from .image_folder_dataset import ImageFolderDataset
from .hdf5_dataset import HDF5ImageDataset
from ..utils.distributed import multi_model_collate

# Check if ParquetEmbeddingDataset exists in parquet_dataset.py
try:
    from .parquet_dataset import ParquetEmbeddingDataset
    __all__ = [
        "ParquetImageDataset",
        "ParquetEmbeddingDataset",
        "ImageFolderDataset",
        "HDF5ImageDataset",
        "multi_model_collate",
    ]
except ImportError:
    __all__ = [
        "ParquetImageDataset",
        "ImageFolderDataset",
        "HDF5ImageDataset",
        "multi_model_collate",
    ]
