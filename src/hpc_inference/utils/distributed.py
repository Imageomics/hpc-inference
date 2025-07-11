import os
import torch
from collections import defaultdict
from typing import List, Union, Tuple, Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def assign_files_to_rank(
    rank: int, 
    world_size: int, 
    files: List[Union[str, Path]], 
    evenly_distribute: bool = True
) -> List[str]:
    """
    Assign files to the current rank based on the world size.
    
    This method ensures that each rank gets a unique set of files to process.
    The files can be distributed evenly based on their size (LPT algorithm) or simply by their order.
    This is useful for large datasets where some files may be significantly larger than others.

    Args:
        rank: Current process rank (0-indexed).
        world_size: Total number of processes across all ranks.
        files: List of file paths to distribute across ranks.
        evenly_distribute: Whether to distribute files evenly based on file size. Defaults to True.
            - If True: Uses Longest Processing Time (LPT) algorithm for load balancing
            - If False: Uses simple round-robin distribution

    Returns:
        List of file paths assigned to the given rank.
        
    Raises:
        ValueError: If rank is negative or >= world_size.
        FileNotFoundError: If any file in the list doesn't exist when evenly_distribute=True.
        
    Examples:
        >>> files = ["/path/file1.parquet", "/path/file2.parquet", "/path/file3.parquet"]
        >>> assign_files_to_rank(0, 2, files, evenly_distribute=False)
        ["/path/file1.parquet", "/path/file3.parquet"]
        
        >>> assign_files_to_rank(1, 2, files, evenly_distribute=False) 
        ["/path/file2.parquet"]
    """
    # Input validation
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Rank {rank} must be between 0 and {world_size-1}")
    
    if world_size <= 0:
        raise ValueError(f"World size {world_size} must be positive")
        
    if not files:
        logging.warning("Empty file list provided to assign_files_to_rank")
        return []
    
    # Convert to strings for consistency
    file_paths = [str(f) for f in files]
    
    if world_size == 1:
        logging.info(f"Single rank mode: assigning all {len(file_paths)} files to rank 0")
        return file_paths

    if not evenly_distribute:
        # Simple round-robin distribution
        assigned_files = file_paths[rank::world_size]
        logging.info(f"Rank {rank} assigned {len(assigned_files)} files via round-robin")
        return assigned_files

    # Get file sizes for load balancing
    try:
        file_sizes = [(f, os.path.getsize(f)) for f in file_paths]
    except FileNotFoundError as e:
        logging.error(f"File not found during size calculation: {e}")
        raise
    
    # Sort files by size (largest first for better load balancing)
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    assignments = defaultdict(list)
    load_per_rank = [0] * world_size

    # Use Longest Processing Time (LPT) algorithm for load balancing
    for fpath, size in file_sizes:
        min_rank = load_per_rank.index(min(load_per_rank))
        assignments[min_rank].append(fpath)
        load_per_rank[min_rank] += size
    
    assigned_files = assignments[rank]
    total_size_gb = load_per_rank[rank] / (1024**3)
    
    logging.info(f"Rank {rank} assigned {len(assigned_files)} files "
                f"(total size: {total_size_gb:.2f} GB)")
    
    return assigned_files


def assign_indices_to_rank(
    rank: int, 
    world_size: int, 
    total_items: int, 
    evenly_distribute: bool = True
) -> Tuple[int, int]:
    """
    Assign item indices to the current rank based on the world size.
    
    This function calculates which range of indices each rank should process
    when working with datasets that can be split by index ranges.
    
    Args:
        rank: Current process rank (0-indexed).
        world_size: Total number of processes across all ranks.
        total_items: Total number of items to distribute across ranks.
        evenly_distribute: Whether to distribute items evenly. Defaults to True.
            - If True: Each rank gets approximately equal number of items
            - If False: Simple round-robin style distribution
    
    Returns:
        Tuple of (start_idx, end_idx) representing the range of indices for this rank.
        The range is [start_idx, end_idx) (end_idx is exclusive).
        
    Raises:
        ValueError: If rank is negative or >= world_size, or if total_items is negative.
        
    Examples:
        >>> assign_indices_to_rank(0, 3, 10, evenly_distribute=True)
        (0, 4)  # Rank 0 gets indices 0,1,2,3
        
        >>> assign_indices_to_rank(1, 3, 10, evenly_distribute=True) 
        (4, 7)  # Rank 1 gets indices 4,5,6
        
        >>> assign_indices_to_rank(2, 3, 10, evenly_distribute=True)
        (7, 10) # Rank 2 gets indices 7,8,9
    """
    # Input validation
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Rank {rank} must be between 0 and {world_size-1}")
        
    if world_size <= 0:
        raise ValueError(f"World size {world_size} must be positive")
        
    if total_items < 0:
        raise ValueError(f"Total items {total_items} must be non-negative")
    
    if total_items == 0:
        logging.warning("Zero items to distribute")
        return 0, 0
    
    if world_size == 1:
        logging.info(f"Single rank mode: assigning all {total_items} items to rank 0")
        return 0, total_items
    
    if not evenly_distribute:
        # Simple round-robin assignment
        items_per_rank = total_items // world_size
        remainder = total_items % world_size
        
        start_idx = rank * items_per_rank + min(rank, remainder)
        end_idx = start_idx + items_per_rank + (1 if rank < remainder else 0)
        
        logging.info(f"Rank {rank} assigned indices [{start_idx}, {end_idx}) "
                    f"({end_idx - start_idx} items) via round-robin")
        return start_idx, end_idx
    else:
        # Even distribution
        items_per_rank = total_items // world_size
        remainder = total_items % world_size
        
        # Distribute remainder among first 'remainder' ranks
        if rank < remainder:
            start_idx = rank * (items_per_rank + 1)
            end_idx = start_idx + items_per_rank + 1
        else:
            start_idx = rank * items_per_rank + remainder
            end_idx = start_idx + items_per_rank
            
        logging.info(f"Rank {rank} assigned indices [{start_idx}, {end_idx}) "
                    f"({end_idx - start_idx} items) via even distribution")
        return start_idx, end_idx


def get_distributed_info() -> Tuple[int, int]:
    """
    Get distributed training information from environment variables.
    
    This function extracts rank and world size from common distributed training
    environment variables (SLURM, torchrun, etc.).
    
    Returns:
        Tuple of (rank, world_size) extracted from environment variables.
        If no distributed environment is detected, returns (0, 1).
        
    Examples:
        >>> # In SLURM environment
        >>> get_distributed_info()
        (2, 8)  # rank=2, world_size=8
        
        >>> # In single process environment  
        >>> get_distributed_info()
        (0, 1)  # rank=0, world_size=1
    """
    # Try SLURM environment variables first
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    # If SLURM not found, try torchrun/pytorch distributed
    if world_size == 1:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # If still not found, try other common variables
    if world_size == 1:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logging.info(f"Detected distributed environment: rank={rank}, world_size={world_size}")
    return rank, world_size


def validate_distributed_setup(rank: int, world_size: int) -> bool:
    """
    Validate distributed training setup parameters.
    
    Args:
        rank: Process rank to validate.
        world_size: World size to validate.
        
    Returns:
        True if setup is valid, False otherwise.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if rank < 0:
        raise ValueError(f"Rank {rank} must be non-negative")
        
    if world_size <= 0:
        raise ValueError(f"World size {world_size} must be positive")
        
    if rank >= world_size:
        raise ValueError(f"Rank {rank} must be less than world size {world_size}")
        
    logging.info(f"Distributed setup validated: rank={rank}, world_size={world_size}")
    return True


def pil_image_collate(batch: List[Tuple[str, Any]]) -> Tuple[List[str], List[Any]]:
    """
    Custom collate function for batches containing PIL Images.
    
    This function is required when working with datasets that return PIL Images
    because PyTorch's default collate function only handles tensors, numpy arrays,
    numbers, dicts, and lists - not PIL Image objects.
    
    Args:
        batch: List of (uuid, image) tuples where image is a PIL Image.
               Each tuple contains a UUID string and a PIL Image object.
    
    Returns:
        Tuple containing:
            - uuids: List of UUID strings from the batch
            - images: List of PIL Image objects from the batch
            
    Examples:
        >>> from PIL import Image
        >>> batch = [("img1.jpg", Image.new("RGB", (100, 100))), 
        ...           ("img2.jpg", Image.new("RGB", (200, 200)))]
        >>> uuids, images = pil_image_collate(batch)
        >>> print(uuids)
        ['img1.jpg', 'img2.jpg']
        >>> print([img.size for img in images])
        [(100, 100), (200, 200)]
    """
    uuids, images = zip(*batch)
    return list(uuids), list(images)


def multi_model_collate(batch: List[Tuple[str, Dict[str, Any]]]) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """
    Collate function for batches where each sample is (uuid, {model_name: tensor, ...}).
    
    Args:
        batch: List of (uuid, processed_data_dict) tuples from dataset.
    
    Returns:
        Tuple containing:
            - uuids: List of UUID strings
            - batch_dict: Dict mapping model names to batched tensors
    """
    uuids, processed_list = zip(*batch)  # unzip the batch
    batch_dict = {}
    # Use keys from the first processed dict (assumes all have the same keys)
    for key in processed_list[0]:
        # Stack tensors for each model key
        batch_dict[key] = torch.stack([d[key] for d in processed_list])
    return list(uuids), batch_dict
