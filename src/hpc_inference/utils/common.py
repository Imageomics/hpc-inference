import os
from typing import Dict, Any, Optional, Union, List, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Parsed configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        
    Examples:
        >>> config = load_config("config.yaml")
        >>> print(config["batch_size"])
        32
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string (e.g., "1h 30m 45.67s", "2m 15.34s", "12.45s").
        
    Examples:
        >>> format_time(3661.5)
        '1h 1m 1.50s'
        >>> format_time(125.67)
        '2m 5.67s'
        >>> format_time(45.23)
        '45.23s'
    """
    if seconds < 0:
        return "0.00s"
        
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h >= 1:
        return f"{int(h)}h {int(m)}m {s:.2f}s"
    elif m >= 1:
        return f"{int(m)}m {s:.2f}s"
    else:
        return f"{s:.2f}s"


def decode_image(
    row_data: Union[Dict[str, Any], pd.Series], 
    return_type: Literal["numpy", "pil"] = "pil"
) -> Optional[Union[np.ndarray, Image.Image]]:
    """
    Decode an image stored as raw bytes in a row dictionary into a NumPy array or PIL Image.

    The function expects the row dictionary to contain an "image" key with raw image bytes,
    and either an "original_size" or "resized_size" key specifying the (height, width) of the image.
    
    **It assumes the image has RGB channels, and reshapes the byte buffer accordingly.**

    Args:
        row_data: Dictionary containing image data and metadata.
            - "image": Raw image bytes (bytes or bytearray).
            - "original_size" or "resized_size": Tuple (height, width) of the image.
        return_type: Type of the returned image. Can be "numpy" for NumPy array or "pil" for PIL Image.

    Returns:
        Decoded image as a NumPy array (RGB format) or PIL Image, or None if decoding fails.

    Raises:
        None: Returns None on any decoding error, with warning logged.

    Notes:
        - Returns None if the image size does not match the expected dimensions.
        - Logs a warning if decoding fails.
        - Converts BGR to RGB channel order automatically.
        
    Examples:
        >>> row = {"image": image_bytes, "original_size": (224, 224)}
        >>> img = decode_image(row, return_type="pil")
        >>> isinstance(img, Image.Image)
        True
        
        >>> img_array = decode_image(row, return_type="numpy")
        >>> img_array.shape
        (224, 224, 3)
    """
    N_CHANNELS = 3
    
    # Validate input data
    if not isinstance(row_data, (dict, pd.Series)):
        logging.error("row_data must be a dictionary or pandas Series")
        return None
        
    if "image" not in row_data or not isinstance(row_data["image"], (bytes, bytearray)):
        logging.error("Missing or invalid 'image' key in row_data.")
        return None

    if return_type not in ["numpy", "pil"]:
        logging.error(f"Unsupported return type: {return_type}. Use 'numpy' or 'pil'.")
        return None

    try:
        image_bytes = row_data["image"]
        np_image = np.frombuffer(image_bytes, dtype=np.uint8)

        # Try to find size information
        for key in ["original_size", "resized_size"]:
            if key in row_data and isinstance(row_data[key], (tuple, list, np.ndarray)) and len(row_data[key]) == 2:
                height, width = [int(x) for x in row_data[key]]
                expected_size = height * width * N_CHANNELS
                
                if np_image.size == expected_size:
                    img_array = np_image.reshape((height, width, N_CHANNELS))
                    # Flip channels from BGR to RGB if needed
                    img_array = img_array[..., ::-1]
                    
                    if return_type == "numpy":
                        return img_array
                    elif return_type == "pil":
                        return Image.fromarray(img_array, mode='RGB')

        logging.warning("Image size does not match expected dimensions.")
        return None
        
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None


def save_emb_to_parquet(
    uuids: List[str],
    embeddings: Dict[str, Union[torch.Tensor, np.ndarray]],
    path: Union[str, Path],
    compression: str = "zstd"
) -> None:
    """
    Save embeddings (as a dict of column names to tensors/arrays) and uuids to a Parquet file.

    Args:
        uuids: List of unique identifiers corresponding to each embedding.
        embeddings: Dictionary mapping column names to torch.Tensor or np.ndarray.
            Each tensor/array should have the same first dimension as len(uuids).
        path: Output Parquet file path.
        compression: Compression type for Parquet file. Defaults to "zstd".
            Other options: "snappy", "gzip", "brotli", "lz4", "uncompressed".
            
    Raises:
        ValueError: If embeddings have mismatched dimensions or empty inputs.
        IOError: If file cannot be written.
        
    Examples:
        >>> uuids = ["img_001", "img_002", "img_003"]
        >>> embeddings = {
        ...     "clip_emb": torch.randn(3, 512),
        ...     "resnet_emb": torch.randn(3, 2048)
        ... }
        >>> save_emb_to_parquet(uuids, embeddings, "embeddings.parquet")
    """
    if not uuids:
        raise ValueError("Empty uuids list provided")
        
    if not embeddings:
        raise ValueError("Empty embeddings dictionary provided")
    
    # Validate all embeddings have the same first dimension
    expected_length = len(uuids)
    for colname, emb in embeddings.items():
        if len(emb) != expected_length:
            raise ValueError(f"Embedding '{colname}' length {len(emb)} doesn't match uuids length {expected_length}")
    
    try:
        # Create UUID array
        array_uuid = pa.array(uuids, type=pa.string())
        arrays = [array_uuid]
        names = ["uuid"]

        # Process each embedding
        for colname, emb in embeddings.items():
            # Convert to numpy if it's a torch tensor
            if isinstance(emb, torch.Tensor):
                emb_np = emb.cpu().numpy()
            else:
                emb_np = emb
                
            # Convert to list of lists for PyArrow
            emb_list = emb_np.tolist()
            arrays.append(pa.array(emb_list, type=pa.list_(pa.float32())))
            names.append(colname)

        # Create and write table
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, str(path), compression=compression)
        
        logging.info(f"Saved {len(uuids)} embeddings to {path}")
        
    except Exception as e:
        logging.error(f"Failed to save embeddings to {path}: {e}")
        raise


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that a configuration dictionary contains all required keys.
    
    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required key names.
        
    Returns:
        True if all required keys are present, False otherwise.
        
    Examples:
        >>> config = {"batch_size": 32, "learning_rate": 0.001}
        >>> validate_config(config, ["batch_size", "learning_rate"])
        True
        >>> validate_config(config, ["batch_size", "missing_key"])
        False
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
        
    logging.info("Configuration validation passed")
    return True


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create.
        
    Returns:
        Path object representing the created/existing directory.
        
    Examples:
        >>> ensure_dir("/path/to/output")
        PosixPath('/path/to/output')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path  