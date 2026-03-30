"""HPC-Inference: High-performance batch inference for image datasets."""

__version__ = "0.1.0"
__author__ = "Net Zhang"
__email__ = "zhang.11091@osu.edu"

# Import main classes for easy access
from .datasets import ParquetImageDataset, ImageFolderDataset, HDF5ImageDataset
from .utils.common import decode_image, save_emb_to_parquet, format_time, load_config
from .utils.distributed import assign_files_to_rank, assign_indices_to_rank, get_distributed_info, multi_model_collate, pil_image_collate
from .utils import profiling

__all__ = [
    "__version__",
    "ParquetImageDataset",
    "ImageFolderDataset",
    "HDF5ImageDataset",
    "decode_image",
    "save_emb_to_parquet",
    "format_time",
    "load_config",
    "assign_files_to_rank",
    "assign_indices_to_rank",
    "get_distributed_info",
    "multi_model_collate",
    "pil_image_collate",
    "profiling",
]

def list_available_features():
    """List available optional features based on installed dependencies."""
    features = {
        "core": True,       # Always available
        "profiling": True,  # Always available now
        "openclip": False,
        "conversion": False,
    }

    # Check OpenCLIP
    try:
        import open_clip
        features["openclip"] = True
    except ImportError:
        pass

    # Check WebDataset (conversion)
    try:
        import webdataset
        features["conversion"] = True
    except ImportError:
        pass

    return features

def print_installation_guide():
    """Print installation guide for missing features."""
    features = list_available_features()
    
    print("HPC-Inference Installation Status:")
    print("================================")
    
    print("[x] Core (datasets, utils, profiling): Available")
    print(f"{'[x]' if features['openclip'] else '[ ]'} OpenCLIP: {'Available' if features['openclip'] else 'Missing'}")
    print(f"{'[x]' if features['conversion'] else '[ ]'} Conversion (WebDataset to HDF5): {'Available' if features['conversion'] else 'Missing'}")

    missing = []
    if not features['openclip']:
        missing.append("  pip install 'hpc-inference[openclip]'")
    if not features['conversion']:
        missing.append("  pip install 'hpc-inference[conversion]'")
    if missing:
        print("\nTo enable missing features:")
        print("\n".join(missing))

    print("To install everything: pip install 'hpc-inference[all]'")
