import numpy as np
import logging
from PIL import Image
import torch
import yaml

import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_config(config_path):
    """Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Parsed configuration
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.2f}s"
    elif m:
        return f"{int(m)}m {s:.2f}s"
    else:
        return f"{s:.2f}s"
    
def decode_image(row_data, return_type="pil"):
    """
    Decodes an image stored as raw bytes in a row dictionary into a NumPy array.

    The function expects the row dictionary to contain an "image" key with raw image bytes,
    and either an "original_size" or "resized_size" key specifying the (height, width) of the image.
    
    **It assumes the image has RGB channels, and reshapes the byte buffer accordingly.**

    Args:
        row_data (dict): Dictionary containing image data and metadata.
            - "image": Raw image bytes.
            - "original_size" or "resized_size": Tuple (height, width) of the image.
        return_type (str): Type of the returned image. Can be "numpy" for NumPy array or "pil" for PIL Image.

    Returns:
        np.ndarray: Decoded image as a NumPy array in RGB format, or None if decoding fails.

    Notes:
        - Returns None if the image size does not match the expected dimensions.
        - Logs a warning if decoding fails.
    """
    N_CHANNELS = 3
    if "image" not in row_data or not isinstance(row_data["image"], (bytes, bytearray)):
        logging.error("Missing or invalid 'image' key in row_data.")
        return None

    image_bytes = row_data["image"]
    np_image = np.frombuffer(image_bytes, dtype=np.uint8)

    for key in ["original_size", "resized_size"]:
        if key in row_data and isinstance(row_data[key], (tuple, list, np.ndarray)) and len(row_data[key]) == 2:
            #height, width = row_data[key]
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
                else:
                    logging.error(f"Unsupported return type: {return_type}. Use 'numpy' or 'pil'.")
                    return None

    logging.warning("Image size does not match expected dimensions.")
    return None

def save_emb_to_parquet(
    uuids,
    embeddings: dict,  # dict of {colname: tensor/ndarray}
    path: str,
    compression="zstd"
):
    """
    Save embeddings (as a dict of column names to tensors/arrays) and uuids to a Parquet file.

    Args:
        uuids (list): List of UUIDs.
        embeddings (dict): Dict mapping column names to torch.Tensor or np.ndarray.
        path (str): Output Parquet file path.
        compression (str): Compression type for Parquet.
    """
    array_uuid = pa.array(uuids, type=pa.string())
    arrays = [array_uuid]
    names = ["uuid"]

    for colname, emb in embeddings.items():
        # Convert to numpy if it's a torch tensor
        if isinstance(emb, torch.Tensor):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = emb
        arrays.append(pa.array(emb_np.tolist(), type=pa.list_(pa.float32())))
        names.append(colname)

    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(
        table, path,
        compression=compression
    )
