import os
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms
import concurrent.futures
import random
import time
from typing import Optional, Union, Dict, Callable, List, Literal
from pathlib import Path

import logging
from ..utils.distributed import assign_files_to_rank

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class ImageFolderDataset(IterableDataset):
    """
    Loads images from a folder in a streaming fashion, with support for distributed processing.
    - Handles common image extensions
    - Loads images as RGB by default (can be changed via color_mode)
    - Validates images using PIL if validate=True
    - Supports distributed processing with rank-based file partitioning
    - Supports multi-worker data loading within each rank
    
    Returns (uuid, processed_data) tuples where:
    - uuid: filename (without path) as identifier
    - processed_data: tensor (single model) or dict of tensors (multi-model)
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    @staticmethod
    def validate_PIL(file_path: Union[str, Path]) -> bool:
        """
        Validates if the file can be opened by PIL.
        Returns True if valid, False otherwise.
        """
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that it is an image
            return True
        except (IOError, SyntaxError):
            return False
    
    @classmethod
    def validate_image_files(cls, image_files: List[str], max_workers: int = 16) -> List[str]:
        """Validates a list of image files using PIL."""
        if not image_files:
            raise ValueError(f"No valid image files found")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(cls.validate_PIL, image_files))
        
        valid_files = [f for f, is_valid in zip(image_files, results) if is_valid]
        invalid_files = [f for f, is_valid in zip(image_files, results) if not is_valid]
        if invalid_files:
            logging.warning(f"Invalid image count: {len(invalid_files)}")
            for invalid_file in invalid_files[:5]:  # Log first 5 invalid files
                logging.warning(f"Invalid image file: {invalid_file}")
            if len(invalid_files) > 5:
                logging.warning(f"... and {len(invalid_files) - 5} more invalid files")
        return valid_files

    def _init_uuid_generator(self, uuid_mode: Literal["filename", "relative", "fullpath", "hash"]) -> None:
        """Initialize the UUID generation function based on the specified mode."""
        if uuid_mode == "filename":
            self.generate_uuid = lambda img_path: os.path.basename(img_path)
        elif uuid_mode == "relative":
            # Calculate relative path from parent of image_dir to preserve folder structure
            parent_dir = os.path.dirname(self.image_dir)
            self.generate_uuid = lambda img_path: os.path.relpath(img_path, parent_dir)
        elif uuid_mode == "fullpath":
            self.generate_uuid = lambda img_path: img_path
        elif uuid_mode == "hash":
            import hashlib
            self.generate_uuid = lambda img_path: hashlib.md5(img_path.encode()).hexdigest()[:16]
        else:
            raise ValueError(f"Invalid uuid_mode: {uuid_mode}. "
                        f"Valid options: 'filename', 'relative', 'fullpath', 'hash'")

    def __init__(
        self, 
        image_dir: Union[str, Path], 
        preprocess: Optional[Union[Callable, Dict[str, Callable]]] = None, 
        color_mode: str = "RGB", 
        validate: bool = False,
        rank: Optional[int] = None, 
        world_size: Optional[int] = None, 
        evenly_distribute: bool = True, 
        stagger: bool = False, 
        uuid_mode: Literal["filename", "relative", "fullpath", "hash"] = "filename"
    ) -> None:
        """
        Args:
            image_dir: Path to image folder.
            preprocess: Transform(s) to apply to images.
                - If callable: single model preprocessing
                - If dict: {model_name: preprocess_fn} for multi-model
                - If None: return PIL image as-is
            color_mode: Color mode for PIL.Image.convert (default: "RGB").
            validate: If True, validates images in the directory using PIL.
            rank: Current process rank for distributed processing.
            world_size: Total number of processes for distributed processing.
            evenly_distribute: Whether to distribute files evenly based on size. Defaults to True.
                If False, files are distributed in a round-robin manner.
            stagger: Whether to stagger the start of each worker. Defaults to False.
            uuid_mode: How to generate UUIDs from image paths.
                - "filename": Use just the filename (image001.jpg)
                - "relative": Use relative path from image_dir (subfolder/image001.jpg)  
                - "fullpath": Use full absolute path
                - "hash": Use hash of the full path
        """
        self.image_dir: str = str(image_dir)
        self.preprocess: Optional[Union[Callable, Dict[str, Callable]]] = preprocess
        self.color_mode: str = color_mode
        self.rank: int = rank or 0
        self.world_size: int = world_size or 1
        self.stagger: bool = stagger
        
        # Initialize UUID generation function based on mode
        self._init_uuid_generator(uuid_mode)
        
        # Get all image files first
        all_image_files = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.IMG_EXTS)
        ])
        
        if not all_image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")
        
        # Apply rank-based partitioning
        if self.world_size > 1:
            rank_image_files = assign_files_to_rank(
                self.rank, self.world_size, all_image_files, evenly_distribute
            )
        else:
            rank_image_files = all_image_files
        
        # Validate files if requested
        if validate:
            rank_image_files = self.validate_image_files(rank_image_files)
            if not rank_image_files:
                raise ValueError(f"No valid images found for rank {self.rank} in directory: {image_dir}")
        
        self.image_files: List[str] = rank_image_files
        logging.info(f"Rank {self.rank} assigned {len(self.image_files)} out of {len(all_image_files)} images")

    def parse_image(self, img_path: str) -> Optional[tuple]:
        """Load and process a single image."""
        try:
            # Load image
            img = Image.open(img_path).convert(self.color_mode)
            
            # Apply preprocessing
            if self.preprocess is None:
                processed_img = img
            elif isinstance(self.preprocess, dict):
                if len(self.preprocess) == 1:
                    # Single model case - return tensor directly
                    key = next(iter(self.preprocess))
                    processed_img = self.preprocess[key](img)
                else:
                    # Multi-model case - return dict of tensors
                    processed_img = {k: fn(img) for k, fn in self.preprocess.items()}
            else:
                # Single callable preprocessing
                processed_img = self.preprocess(img)
            
            # Generate UUID 
            uuid = self.generate_uuid(img_path)
            
            return uuid, processed_img
            
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Error loading/processing image {img_path}: {e}")
            return None

    def __iter__(self):
        """
        Iterate over images, yielding (uuid, processed_image) tuples.
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
        worker_files = self.image_files[worker_id::num_workers]
        
        logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing {len(worker_files)} images")

        for img_path in worker_files:
            result = self.parse_image(img_path)
            if result is not None:
                yield result  # (uuid, processed_img)

    def __len__(self) -> int:
        """Return approximate length (actual length depends on worker assignment)."""
        return len(self.image_files)
