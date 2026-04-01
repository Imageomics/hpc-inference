import os
import io
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import random
import time
from typing import Optional, Union, Dict, Callable, List, Literal, Iterator, Tuple, Any
from pathlib import Path
from datetime import datetime
import h5py

import logging
from ..utils.distributed import assign_files_to_rank

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class HDF5ImageDataset(IterableDataset):
    """
    Loads images from HDF5 files in a streaming fashion.
    - HDF5 structure: file contains "images" group with datasets named by UUID
    - Each dataset contains encoded image bytes (JPEG/PNG/WebP format)
    - Supports distributed processing with rank-based file partitioning
    - Supports multi-worker data loading within each rank
    - Handles both single-model and multi-model preprocessing
    - Provides staggered worker starts and load balancing
    
    Returns (uuid, processed_data) tuples where:
    - uuid: dataset key (e.g., "image_uuid_det123")
    - processed_data: tensor (single model) or dict of tensors (multi-model)
    
    Example usage:
        Basic usage with PIL images (no preprocessing):
        >>> from pathlib import Path
        >>> from hpc_inference.datasets import HDF5ImageDataset
        >>> hdf5_files = list(Path("/path/to/crops").glob("*.h5"))
        >>> dataset = HDF5ImageDataset(hdf5_files, preprocess=None)
        >>> for uuid, image in dataset:
        >>>     print(f"UUID: {uuid}, Image size: {image.size}")
        
        With single model preprocessing:
        >>> from torchvision import transforms
        >>> from hpc_inference.datasets import HDF5ImageDataset
        >>> preprocess = transforms.Compose([
        >>>     transforms.Resize(224),
        >>>     transforms.ToTensor(),
        >>>     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        >>> ])
        >>> dataset = HDF5ImageDataset(hdf5_files, preprocess=preprocess)
        >>> for uuid, tensor in dataset:
        >>>     print(f"UUID: {uuid}, Tensor shape: {tensor.shape}")
        
        With DataLoader for batch processing:
        >>> from torch.utils.data import DataLoader
        >>> from hpc_inference.datasets import HDF5ImageDataset
        >>> dataset = HDF5ImageDataset(
        >>>     hdf5_files,
        >>>     preprocess=preprocess,
        >>>     rank=0,
        >>>     world_size=1,
        >>>     stagger=True
        >>> )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> for batch_uuids, batch_tensors in loader:
        >>>     # Process batch
        >>>     print(f"Batch size: {len(batch_uuids)}")
        
        Multi-model preprocessing:
        >>> from open_clip import create_model_and_transforms
        >>> from hpc_inference.datasets import HDF5ImageDataset
        >>> _, preprocess1 = create_model_and_transforms("ViT-B-32", pretrained="openai")
        >>> _, preprocess2 = create_model_and_transforms("ViT-L-14", pretrained="openai")
        >>> dataset = HDF5ImageDataset(
        >>>     hdf5_files,
        >>>     preprocess={"model1": preprocess1, "model2": preprocess2}
        >>> )
        >>> for uuid, tensors_dict in dataset:
        >>>     print(f"UUID: {uuid}")
        >>>     print(f"Model1 tensor: {tensors_dict['model1'].shape}")
        >>>     print(f"Model2 tensor: {tensors_dict['model2'].shape}")
        
        Distributed processing with SLURM:
        >>> # In a SLURM job, rank and world_size are automatically detected
        >>> from hpc_inference.datasets import HDF5ImageDataset
        >>> dataset = HDF5ImageDataset(
        >>>     hdf5_files,
        >>>     preprocess=preprocess,
        >>>     evenly_distribute=True,  # Balance load across ranks
        >>>     processed_files_log="processed_files.log"
        >>> )
        >>> # Each rank processes its assigned subset of files
    """
    def __init__(
        self,
        hdf5_files: List[Union[str, Path]],
        group_name: str = "images",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        evenly_distribute: bool = True,
        preprocess: Optional[Union[Callable, Dict[str, Callable]]] = None,
        color_mode: Literal['RGB', 'L', 'RGBA'] = 'RGB',
        validate: bool = False,
        stagger: bool = False,
        processed_files_log: Optional[Union[str, Path]] = None
    ):
        """
        Args:
            hdf5_files: List of paths to HDF5 files containing image data.
            group_name: Name of the HDF5 group containing images. Defaults to "images".
            rank: Current process rank for distributed processing.
            world_size: Total number of processes for distributed processing.
            evenly_distribute: Whether to distribute files evenly based on size. Defaults to True.
                If False, files are distributed in a round-robin manner.
            preprocess: Transform(s) to apply to images.
                - If callable: single model preprocessing
                - If dict: {model_name: preprocess_fn} for multi-model
                - If None: return decoded image as-is
            color_mode: Color mode for PIL.Image.convert (default: "RGB").
            validate: If True, validates images when loading. Defaults to False.
            stagger: Whether to stagger the start of each worker. Defaults to False.
            processed_files_log: Path to log file for tracking processed files. Optional.
        """
        super().__init__()
        self.hdf5_files: List[str] = [str(f) for f in hdf5_files]
        self.group_name: str = group_name
        self.rank: int = rank if rank is not None else int(os.environ.get("SLURM_PROCID", 0))
        self.world_size: int = world_size if world_size is not None else int(os.environ.get("SLURM_NTASKS", 1))
        self.preprocess: Optional[Union[Callable, Dict[str, Callable]]] = preprocess
        self.color_mode: str = color_mode
        self.validate: bool = validate
        self.stagger: bool = stagger
        self.processed_files_log: Optional[str] = str(processed_files_log) if processed_files_log else None

        # Apply rank-based file partitioning
        if self.world_size > 1:
            self.assigned_files = assign_files_to_rank(
                self.rank, self.world_size, self.hdf5_files, evenly_distribute
            )
        else:
            self.assigned_files = self.hdf5_files

        logging.info(f"Rank {self.rank} assigned {len(self.assigned_files)} out of {len(self.hdf5_files)} HDF5 files")

    def log_processed_file(self, file_path: str) -> None:
        """Log a processed file to the log file if configured."""
        if self.processed_files_log:
            try:
                with open(self.processed_files_log, 'a') as f:
                    f.write(f"{datetime.now().isoformat()}: {file_path}\n")
            except Exception as e:
                logging.error(f"Failed to log processed file {file_path}: {e}")

    def decode_image(self, image_bytes: bytes) -> Optional[Image.Image]:
        """
        Decode image bytes to PIL Image.
        
        Args:
            image_bytes: Encoded image bytes (JPEG/PNG/WebP format).
            
        Returns:
            PIL Image or None if decoding fails.
        """
        try:
            # Decode from bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Validate before processing to catch corrupted data early
            if self.validate:
                img.verify()
                # Re-open after verify (verify closes the file)
                img = Image.open(io.BytesIO(image_bytes))

            # Convert to desired color mode
            if img.mode != self.color_mode:
                img = img.convert(self.color_mode)

            return img
        except Exception as e:
            logging.error(f"Error decoding image: {e}")
            return None

    def process_hdf5_file(self, file_path: str) -> Iterator[Tuple[str, Any]]:
        """
        Process a single HDF5 file and yield processed data.
        
        Args:
            file_path: Path to the HDF5 file to process.
            
        Yields:
            Tuples of (uuid, processed_data) for each valid image in the file.
        """
        try:
            with h5py.File(file_path, 'r') as h5f:
                # Check if the images group exists
                if self.group_name not in h5f:
                    logging.error(f"Group '{self.group_name}' not found in {file_path}")
                    return
                
                images_group = h5f[self.group_name]
                
                # Iterate through all datasets in the group
                for uuid in images_group.keys():
                    try:
                        # Read encoded image bytes
                        image_bytes = images_group[uuid][()].tobytes()
                        
                        # Decode image
                        img = self.decode_image(image_bytes)
                        
                        if img is None:
                            logging.warning(f"Failed to decode image '{uuid}' in {file_path}")
                            continue
                        
                        # Apply preprocessing
                        if self.preprocess is None:
                            processed_data = img
                        elif isinstance(self.preprocess, dict):
                            if len(self.preprocess) == 1:
                                # Single model case - return tensor directly
                                key = next(iter(self.preprocess))
                                processed_data = self.preprocess[key](img)
                            else:
                                # Multi-model case - return dict of tensors
                                processed_data = {k: fn(img) for k, fn in self.preprocess.items()}
                        else:
                            # Single callable preprocessing
                            processed_data = self.preprocess(img)
                        
                        yield uuid, processed_data
                        
                    except Exception as e:
                        logging.error(f"[Rank {self.rank}] Error processing image '{uuid}' in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Error processing HDF5 file {file_path}: {e}")
            return
        finally:
            # Log processed file
            self.log_processed_file(file_path)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Iterate over HDF5 files and yield processed data.
        Supports multi-worker data loading within each rank.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Staggered processing: each worker starts at a different offset
        if self.stagger and worker_id > 0:
            delay = min(random.uniform(0.1, 0.5) * worker_id, 5.0)
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Staggering start by {delay:.2f} seconds")
            time.sleep(delay)

        # Assign files to workers within this rank
        worker_files = self.assigned_files[worker_id::num_workers]
        
        logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing {len(worker_files)} HDF5 files")

        # Process each assigned file
        for file_path in worker_files:
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {file_path}")
            yield from self.process_hdf5_file(file_path)
