"""
Convert WebDataset (WDS) shards to HDF5 format with WebP compression.

Supports distributed processing across multiple ranks and workers for efficient
parallel conversion of large-scale datasets. Automatically splits large shards
into multiple HDF5 files when max_items is specified.

Examples:
    Single worker processing:
        $ python wds_to_hdf5.py \\
            --wds-dir /path/to/shards \\
            --output-dir /path/to/output \\
            --max-items 10000

    SLURM distributed processing (4 nodes, 8 tasks):
        $ srun -N 4 -n 8 python wds_to_hdf5.py \\
            --wds-dir /fs/ess/data/shards \\
            --output-dir /fs/scratch/output \\
            --max-items 50000 \\
            --webp-quality 95 \\
            --lossy \\
            --verbose

    Custom shard pattern:
        $ python wds_to_hdf5.py \\
            --wds-dir /path/to/shards \\
            --output-dir /path/to/output \\
            --shard-pattern "shard-*.tar" \\
            --max-items 20000

Output Structure:
    output_dir/
        data/
            {shard_name}.h5 (or {shard_name}_part0.h5, _part1.h5, ...)
        metadata/
            {shard_name}.parquet (or {shard_name}_part0.parquet, ...)

HDF5 Format:
    Each HDF5 file contains:
        - "images" (group): UUID-keyed datasets containing WebP compressed images
            - Each dataset is named by its UUID and contains a vlen uint8 array

Metadata Format:
    Each parquet file contains:
        - "uuid": Sample UUID/key from WDS
        - "h5_file": Absolute path to the h5 file containing this UUID
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_shard_files(wds_dir: str, pattern: str = "*.tar") -> List[str]:
    """
    Get list of shard files from directory.
    
    Args:
        wds_dir: Directory containing WDS shards
        pattern: Glob pattern for shard files
        
    Returns:
        Sorted list of shard file paths
    """
    wds_path = Path(wds_dir)
    if not wds_path.exists():
        raise ValueError(f"WDS directory does not exist: {wds_dir}")
    
    shards = sorted([str(p) for p in wds_path.glob(pattern)])
    if not shards:
        raise ValueError(f"No shards found in {wds_dir} with pattern {pattern}")
    
    logger.info(f"Found {len(shards)} shards in {wds_dir}")
    return shards


def assign_shards_to_worker(
    shards: List[str],
    global_worker_id: int,
    total_workers: int
) -> List[str]:
    """
    Assign shards to a specific worker for distributed processing.
    
    Args:
        shards: List of all shard paths
        global_worker_id: Global worker ID across all ranks
        total_workers: Total number of workers
        
    Returns:
        List of shard paths assigned to this worker
    """
    assigned = [s for i, s in enumerate(shards) if i % total_workers == global_worker_id]
    logger.info(f"Worker {global_worker_id}/{total_workers} assigned {len(assigned)} shards")
    return assigned


def process_shard_to_hdf5(
    shard_path: str,
    output_dir: str,
    max_items: Optional[int] = None,
    webp_quality: int = 100,
    lossless: bool = True,
    disable_progress: bool = False
) -> int:
    """
    Convert a single WDS shard to HDF5 with WebP compression.
    Creates multiple HDF5 files if max_items is exceeded.
    Generates metadata parquet files mapping UUIDs to h5 files.
    
    Args:
        shard_path: Path to input WDS shard
        output_dir: Base output directory (will create data/ and metadata/ subdirs)
        max_items: Maximum items per HDF5 file (None for no limit)
        webp_quality: WebP quality (1-100)
        lossless: Use lossless WebP compression
        disable_progress: Disable tqdm progress bar
        
    Returns:
        Total number of items successfully processed
    """
    logger.info(f"Processing {shard_path}")
    
    # Create data and metadata directories
    data_dir = Path(output_dir) / "data"
    metadata_dir = Path(output_dir) / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Get shard name for output files
    shard_name = Path(shard_path).stem
    
    total_processed = 0
    file_part = 0
    current_h5f = None
    current_file_count = 0
    current_metadata = []
    current_h5_path = None
    current_images_group = None
    
    def save_metadata():
        """Save current metadata to parquet file."""
        if current_metadata and current_h5_path:
            df = pd.DataFrame(current_metadata)
            metadata_file = metadata_dir / f"{Path(current_h5_path).stem}.parquet"
            df.to_parquet(metadata_file, index=False)
            logger.info(f"Saved metadata to {metadata_file}")
    
    try:
        dataset = wds.WebDataset(shard_path, handler=wds.ignore_and_continue, shardshuffle=False)
        
        iterator = dataset if disable_progress else tqdm(dataset, desc=f"Processing {Path(shard_path).name}")
        
        for sample in iterator:
            key = sample.get("__key__")
            img_bytes = sample.get("jpg") or sample.get("png") or sample.get("jpeg")
            
            if not key or img_bytes is None:
                continue
            
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buffer = io.BytesIO()
                pil_img.save(buffer, format="WEBP", lossless=lossless, quality=webp_quality)
                webp_raw = buffer.getvalue()
                webp_arr = np.frombuffer(webp_raw, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Failed to process image {key}: {e}")
                continue
            
            # Open new file if needed
            if current_h5f is None or (max_items and current_file_count >= max_items):
                if current_h5f is not None:
                    current_h5f.close()
                    save_metadata()
                    logger.info(f"Completed part {file_part} with {current_file_count} items")
                
                # Create new h5 file
                if max_items:
                    h5_filename = f"{shard_name}_part{file_part}.h5"
                else:
                    h5_filename = f"{shard_name}.h5"
                
                current_h5_path = str(data_dir / h5_filename)
                current_h5f = h5py.File(current_h5_path, "w")
                
                # Create images as a group for UUID-based access
                current_images_group = current_h5f.create_group("images")
                
                current_file_count = 0
                current_metadata = []
                file_part += 1
            
            # Store image with UUID as key (raw uint8 array)
            current_images_group.create_dataset(
                key, 
                data=webp_arr,
                compression=None  # Already compressed as WebP
            )
            
            # Add to metadata
            current_metadata.append({
                "uuid": key,
                "h5_file": current_h5_path
            })
            
            current_file_count += 1
            total_processed += 1
        
        if current_h5f is not None:
            current_h5f.close()
            save_metadata()
            logger.info(f"Completed part {file_part - 1} with {current_file_count} items")
        
        logger.info(f"Successfully processed {total_processed} items from {shard_path}")
        return total_processed
        
    except Exception as e:
        logger.error(f"Failed to process shard {shard_path}: {e}")
        if current_h5f is not None:
            current_h5f.close()
        raise


def main(
    wds_dir: str,
    output_dir: str,
    shard_pattern: str = "*.tar",
    global_worker_id: int = 0,
    total_workers: int = 1,
    max_items: Optional[int] = None,
    webp_quality: int = 100,
    lossless: bool = True,
    verbose: bool = False,
    disable_progress: bool = False
):
    """
    Main entry point for WDS to HDF5 conversion.
    
    Args:
        wds_dir: Directory containing WDS shard files
        output_dir: Output directory for HDF5 files
        shard_pattern: Glob pattern for shard files
        global_worker_id: Global worker ID for distributed processing
        total_workers: Total number of workers across all ranks
        max_items: Maximum items per HDF5 file (splits if exceeded)
        webp_quality: WebP quality (1-100)
        lossless: Use lossless WebP compression
        verbose: Enable verbose logging
        disable_progress: Disable tqdm progress bar
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting conversion - Global Worker {global_worker_id}/{total_workers}")
    
    # Get all shards and assign to this worker
    try:
        all_shards = get_shard_files(wds_dir, shard_pattern)
        assigned_shards = assign_shards_to_worker(all_shards, global_worker_id, total_workers)
        
        if not assigned_shards:
            logger.warning("No shards assigned to this worker")
            return
        
        # Process each assigned shard
        total_processed = 0
        for shard_path in assigned_shards:
            processed = process_shard_to_hdf5(
                shard_path=shard_path,
                output_dir=output_dir,
                max_items=max_items,
                webp_quality=webp_quality,
                lossless=lossless,
                disable_progress=disable_progress
            )
            total_processed += processed
        
        logger.info(f"Worker {global_worker_id} completed: {total_processed} total items processed")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert WebDataset shards to HDF5 with WebP compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--wds-dir", type=str, required=True,
                        help="Directory containing WDS shard files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for HDF5 files")
    parser.add_argument("--shard-pattern", type=str, default="*.tar",
                        help="Glob pattern for shard files")
    
    # Processing arguments
    parser.add_argument("--max-items", type=int, default=None,
                        help="Maximum items per HDF5 file (splits shard if exceeded)")
    parser.add_argument("--webp-quality", type=int, default=100,
                        help="WebP quality (1-100)")
    parser.add_argument("--lossy", action="store_true",
                        help="Use lossy WebP compression (default: lossless)")
    parser.add_argument("--disable-progress", action="store_true",
                        help="Disable tqdm progress bar (recommended for SLURM jobs)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Get SLURM environment variables for distributed processing
    global_worker_id = int(os.environ.get("SLURM_PROCID", 0))
    total_workers = int(os.environ.get("SLURM_NTASKS", 1))
    
    main(
        wds_dir=args.wds_dir,
        output_dir=args.output_dir,
        shard_pattern=args.shard_pattern,
        global_worker_id=global_worker_id,
        total_workers=total_workers,
        max_items=args.max_items,
        webp_quality=args.webp_quality,
        lossless=not args.lossy,
        verbose=args.verbose,
        disable_progress=args.disable_progress
    )