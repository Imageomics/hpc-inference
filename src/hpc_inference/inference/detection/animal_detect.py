import time
import threading
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Any, List
import pyarrow as pa
import pyarrow.parquet as pq
import json

from ...datasets.parquet_dataset import ParquetImageDataset
from ...datasets.image_folder_dataset import ImageFolderDataset
from ...utils.common import format_time, decode_image, save_emb_to_parquet, load_config
from ...utils.transforms import MegaDetector_v5_Transform
from ...utils import profiling
from .animal_detector import AnimalDetector

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def save_detection_results(uuids: List[str], detection_results: List[Dict[str, Any]], output_file: str) -> None:
    """Save animal detection results to parquet file using PyArrow for better performance."""
    
    # Extract data for parquet storage
    max_scores = []
    num_detections = []
    detections_json = []
    
    for result in detection_results:
        max_scores.append(result["max_detection_score"])
        num_detections.append(result["num_detections"])
        # Store detections as JSON string for complex data
        detections_json.append(json.dumps(result["detections"]))
    
    # Create PyArrow table
    table = pa.table({
        'uuid': pa.array(uuids, type=pa.string()),
        'max_detection_score': pa.array(max_scores, type=pa.float32()),
        'num_detections': pa.array(num_detections, type=pa.int32()),
        'detections': pa.array(detections_json, type=pa.string())
    })
    
    # Write to parquet with optimized settings
    pq.write_table(
        table, 
        output_file,
        compression='snappy',
        use_dictionary=True,
        write_statistics=True
    )
    
    total_detections = sum(num_detections)
    logging.info(f"Saved {len(uuids)} images with {total_detections} total detections to {output_file}")

@torch.no_grad()
def main(
    config: Dict[str, Any], 
    target_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    input_type: str,
    file_list: Optional[Union[str, Path]] = None
) -> None:
    """
    Main function for YOLO animal detection using MegaDetector.
    
    Args:
        config: Configuration dictionary containing model and processing parameters.
        target_dir: Directory containing input data (Parquet files or images).
        output_dir: Directory to save output detection results and profiles.
        input_type: Type of input data ("images" or "parquet").
        file_list: Optional file containing list of Parquet files to process.
    """
    # Validate input type
    if input_type not in ["images", "parquet"]:
        raise ValueError(f"Invalid input_type: {input_type}. Must be 'images' or 'parquet'")

    # =============== #
    # ---- Setup ----
    # =============== #
    start_time = time.time()
    
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = 0
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    logging.info(f"Global rank: {global_rank}, Local rank: {local_rank}, World size: {world_size}")
    
    base_output_dir = os.path.abspath(str(output_dir))
    detections_output_dir = os.path.abspath(os.path.join(base_output_dir, "detections", f"rank_{global_rank}"))
    os.makedirs(detections_output_dir, exist_ok=True)
    profile_dir = os.path.abspath(os.path.join(base_output_dir, "profile_results", f"rank_{global_rank}"))
    os.makedirs(profile_dir, exist_ok=True)
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Initialize and load animal detector
    animal_detector = AnimalDetector(config, device)
    animal_detector.load_model()
    
    # Create MegaDetector-specific preprocessing transforms
    # Uses transforms from utils.transforms module (adapted from Microsoft CameraTraps)
    image_size = config.get("image_size", 1280)  # MegaDetector standard input size
    preprocess = MegaDetector_v5_Transform(target_size=image_size)
    
    logging.info(f"Using MegaDetector v5 transforms with image size: {image_size}x{image_size}")

    # Create dataset based on input type
    if input_type == "images":
        logging.info(f"Processing image directory: {target_dir}")
        
        dataset = ImageFolderDataset(
            target_dir, 
            preprocess=preprocess,
            validate=config.get("validate_images", False),
            rank=global_rank,
            world_size=world_size,
            evenly_distribute=config.get("evenly_distribute", True),
            stagger=config.get("stagger", False),
            uuid_mode=config.get("uuid_mode", "filename")
        )
        
    elif input_type == "parquet":
        logging.info(f"Processing Parquet files from: {target_dir}")
        
        # Find all Parquet files in the target directory
        target_path = Path(target_dir)
        parquet_files = []
        
        if file_list is None:
            parquet_files = [str(p) for p in target_path.rglob('*.parquet')]
            logging.info(f"Found {len(parquet_files)} Parquet files in {target_dir}")
        else:
            file_list_path = Path(file_list)
            if file_list_path.exists():
                with open(file_list_path, "r") as f:
                    parquet_files = [line.strip() for line in f if line.strip().endswith('.parquet')]
                logging.info(f"Loaded {len(parquet_files)} Parquet files from {file_list}")
            else:
                raise FileNotFoundError(f"File list not found: {file_list}")
        
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {target_dir}")
        
        processed_files_log = os.path.join(
            detections_output_dir, f"processed_files_rank{global_rank}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
        )
        dataset = ParquetImageDataset(
            parquet_files,
            col_uuid="uuid",
            rank=global_rank, 
            world_size=world_size,  
            evenly_distribute=config.get("evenly_distribute", True),
            decode_fn=decode_image,
            preprocess=preprocess,  
            read_batch_size=config.get("read_batch_size", 128),
            read_columns=config.get("read_columns", ["uuid", "original_size", "resized_size", "image"]),
            stagger=config.get("stagger", False),
            processed_files_log=processed_files_log
        )

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 28),
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 16)
    )

    all_detection_results = []
    all_uuids = []
    all_batch_stats = []
    usage_log = []
    file_idx = 0
    n_imgs_processed = 0

    usage_stop = threading.Event()
    usage_thread = threading.Thread(
        target=profiling.start_usage_logging, 
        args=(usage_log, usage_stop, 0.5, 0)
    )
    usage_thread.start()
    
    max_rows_per_file = config.get("max_rows_per_file", 10000)
    out_prefix = config.get("out_prefix", "animal_detection_results")
    conf_threshold = config.get("confidence_threshold", 0.2)

    # Main batch loop
    for batch_idx, (uuids, images) in enumerate(loader):
        batch_stats = {"batch": batch_idx, "batch_size": len(uuids)}

        t0 = time.perf_counter()
        images = images.to(device)
        t1 = time.perf_counter()
        
        # Run animal detection using AnimalDetector
        detection_results = animal_detector.detect(images, conf_threshold)
        t2 = time.perf_counter()
        
        all_detection_results.extend(detection_results)
        all_uuids.extend(uuids)
        n_imgs_processed += len(uuids)

        # Count total detections in this batch for logging
        batch_detections = sum(result["num_detections"] for result in detection_results)
        # logging.info(f"Batch {batch_idx}: {len(uuids)} images, {batch_detections} detections")
        
        # Save results when reaching max_rows_per_file
        if len(all_uuids) >= max_rows_per_file:
            out_file = os.path.join(
                detections_output_dir, 
                f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
            )
            save_detection_results(all_uuids, all_detection_results, out_file)
            file_idx += 1
            all_detection_results = []
            all_uuids = []
            
        batch_stats.update({
            "preprocessing_s": t1 - t0,
            "inference_s": t2 - t1,
            "total_batch_s": t2 - t0,
            "detections_found": batch_detections
        })
        all_batch_stats.append(batch_stats)
    
    # Save remaining results
    if len(all_uuids) > 0:
        out_file = os.path.join(
            detections_output_dir, 
            f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
        )
        save_detection_results(all_uuids, all_detection_results, out_file)

    # Stop profiling and save results
    usage_stop.set()
    usage_thread.join()

    elapsed = time.time() - start_time
    total_detections = sum(stats.get("detections_found", 0) for stats in all_batch_stats)
    
    logging.info(f"Total images processed: {n_imgs_processed}")
    logging.info(f"Total detections found: {total_detections}")
    logging.info(f"Total time taken: {format_time(elapsed)}")
    if n_imgs_processed > 0:
        logging.info(f"Avg time/image: {elapsed/n_imgs_processed:.4f} sec")
        logging.info(f"Throughput: {n_imgs_processed/elapsed:.2f} images/sec")
        logging.info(f"Avg detections/image: {total_detections/n_imgs_processed:.2f}")
    
    profiling.log_computing_specs(
        profile_dir, 
        config.get("batch_size", 16), config.get("num_workers", 28),
        extra_info={
            "prefetch_factor": config.get("prefetch_factor", 16),
            "read_batch_size": config.get("read_batch_size", 128),
            "max_rows_per_file": config.get("max_rows_per_file", 10000),
            "task": "Animal detection",
            "model": config["model"]["weights"],
            "confidence_threshold": conf_threshold,
            "throughput": f"{n_imgs_processed/elapsed:.2f} images/sec" if n_imgs_processed > 0 else "0 images/sec",
            "total_images": n_imgs_processed,
            "total_detections": total_detections,
            "avg_detections_per_image": f"{total_detections/n_imgs_processed:.2f}" if n_imgs_processed > 0 else "0",
            "total_time_s": elapsed,
            "input_type": input_type
        }
    )

    stats_df = profiling.save_batch_stats(all_batch_stats, profile_dir)
    usage_df = profiling.save_usage_log(usage_log, profile_dir)
    profiling.save_usage_plots(usage_df, profile_dir)
    profiling.save_batch_timings_plot(stats_df, profile_dir)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Animal Detection (MegaDetector) with Config File or Command Line Arguments")
    parser.add_argument("target_dir", type=str, help="Directory containing input data")
    parser.add_argument("output_dir", type=str, help="Directory to save output detection results")
    parser.add_argument("--input_type", type=str, required=True, choices=["images", "parquet"],
                        help="Type of input data: 'images' for image directory, 'parquet' for Parquet files")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")
    parser.add_argument("--file_list", type=str, default=None,
                        help="File containing list of Parquet files to process (only for --input_type parquet)")
    
    # Model configuration arguments (used when no config file provided)
    parser.add_argument("--model_weights", type=str, default="md_v5a.0.0.pt", 
                        help="YOLO model weights file (default: md_v5a.0.0.pt - MegaDetector v5a)")
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="Confidence threshold for animal detection (default: 0.2)")
    parser.add_argument("--image_size", type=int, default=1280,
                        help="Input image size for MegaDetector (default: 1280)")
    
    # Compute arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=20, help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=16, help="Dataloader prefetch factor")
    parser.add_argument("--read_batch_size", type=int, default=128, help="Parquet read batch size")
    parser.add_argument("--max_rows_per_file", type=int, default=10000, help="Max rows per output file")
    parser.add_argument("--out_prefix", type=str, default="animal_detection_results", help="Output file prefix")
    parser.add_argument("--read_columns", type=str, nargs="+", 
                        default=["uuid", "original_size", "resized_size", "image"],
                        help="Columns to read from Parquet files (only for --input_type parquet)")
    parser.add_argument("--evenly_distribute", action="store_true", default=True,
                        help="Distribute files evenly based on size (recommended for better load balancing)")
    parser.add_argument("--stagger", action="store_true", 
                        help="Stagger worker start times")
    
    # Image folder specific arguments
    parser.add_argument("--validate_images", action="store_true", 
                        help="Validate images using PIL (slower but safer, only for --input_type images)")
    parser.add_argument("--uuid_mode", type=str, default="filename", 
                        choices=["filename", "relative", "fullpath", "hash"],
                        help="How to generate UUIDs from image paths (only for --input_type images)")
    
    args = parser.parse_args()

    # Validate argument combinations
    if args.input_type == "parquet" and args.file_list and not os.path.exists(args.file_list):
        parser.error(f"File list does not exist: {args.file_list}")
    
    if args.input_type == "images" and args.file_list:
        parser.error("--file_list is only applicable when --input_type is 'parquet'")

    # Load config or create from arguments
    if args.config:
        config = load_config(args.config)
        print(f"Using config file: {args.config}")
    else:
        # Create config from command line arguments
        config = {
            "model": {
                "weights": args.model_weights
            },
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "read_batch_size": args.read_batch_size,
            "max_rows_per_file": args.max_rows_per_file,
            "out_prefix": args.out_prefix,
            "read_columns": args.read_columns,
            "validate_images": args.validate_images,
            "uuid_mode": args.uuid_mode,
            "evenly_distribute": args.evenly_distribute,
            "stagger": args.stagger,
            "confidence_threshold": args.confidence_threshold,
            "image_size": args.image_size
        }
        print("Using command line arguments (no config file provided)")

    main(
        config,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        input_type=args.input_type,
        file_list=args.file_list
    )
