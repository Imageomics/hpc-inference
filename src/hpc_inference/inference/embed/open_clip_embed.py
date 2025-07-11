import time
import threading
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Any

from ...datasets.parquet_dataset import ParquetImageDataset
from ...datasets.image_folder_dataset import ImageFolderDataset
from ...utils.common import format_time, decode_image, save_emb_to_parquet, load_config
from ...utils.distributed import multi_model_collate
from ...utils import profiling

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    open_clip = None

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def check_openclip_dependencies() -> None:
    """Check if OpenCLIP dependencies are available."""
    if not OPENCLIP_AVAILABLE:
        raise ImportError(
            "OpenCLIP is not installed. Install with: "
            "pip install 'hpc-inference[openclip]'"
        )

@torch.no_grad()
def main(
    config: Dict[str, Any], 
    target_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    input_type: str,
    file_list: Optional[Union[str, Path]] = None
) -> None:
    """
    Main function for CLIP embedding generation.
    
    Args:
        config: Configuration dictionary containing model and processing parameters.
        target_dir: Directory containing input data (Parquet files or images).
        output_dir: Directory to save output embeddings and profiles.
        input_type: Type of input data ("images" or "parquet").
        file_list: Optional file containing list of Parquet files to process.
    """
    # Check required dependencies
    check_openclip_dependencies()

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
    embeddings_output_dir = os.path.abspath(os.path.join(base_output_dir, "embeddings", f"rank_{global_rank}"))
    os.makedirs(embeddings_output_dir, exist_ok=True)
    profile_dir = os.path.abspath(os.path.join(base_output_dir, "profile_results", f"rank_{global_rank}"))
    os.makedirs(profile_dir, exist_ok=True)
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Load models from config
    models = {}
    preprocessors = {}
    for key, spec in config["models"].items():
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["name"],
            pretrained=spec.get("pretrained"),
            device=device
        )
        model = torch.compile(model.to(device))
        model.eval()
        models[key] = model
        preprocessors[key] = preprocess

    # Create dataset based on input type
    if input_type == "images":
        logging.info(f"Processing image directory: {target_dir}")
        
        dataset = ImageFolderDataset(
            target_dir, 
            preprocess=preprocessors,
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
            embeddings_output_dir, f"processed_files_rank{global_rank}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
        )
        dataset = ParquetImageDataset(
            parquet_files,
            col_uuid="uuid",
            rank=global_rank, 
            world_size=world_size,  
            evenly_distribute=config.get("evenly_distribute", True),
            decode_fn=decode_image,
            preprocess=preprocessors,
            read_batch_size=config.get("read_batch_size", 128),
            read_columns=config.get("read_columns", ["uuid", "original_size", "resized_size", "image"]),
            stagger=config.get("stagger", False),
            processed_files_log=processed_files_log
        )

        
    collate_fn = multi_model_collate if len(models) > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 28),
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 16),
        collate_fn=collate_fn
    )

    all_embeds = {key: [] for key in models}
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
    out_prefix = config.get("out_prefix", "embed_results")

    # Main batch loop: handle single and multi-model cases
    model_keys = list(models.keys())
    single_model = len(models) == 1
    model_key = model_keys[0] if single_model else None

    for batch_idx, (uuids, data) in enumerate(loader):
        batch_stats = {"batch": batch_idx, "batch_size": len(uuids)}

        t0 = time.perf_counter()
        if single_model:
            images = data.to(device, non_blocking=True)
            t1 = time.perf_counter()
            embeds = models[model_key].encode_image(images)
            t2 = time.perf_counter()
            all_embeds[model_key].append(embeds.cpu().numpy())
            t3 = time.perf_counter()
            del images, embeds  # Free memory
        else:
            batch_dict = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            t1 = time.perf_counter()
            embeds = {}
            for key, model in models.items():
                embeds[key] = model.encode_image(batch_dict[key])
            t2 = time.perf_counter()
            for key in models:
                all_embeds[key].append(embeds[key].cpu().numpy())
            t3 = time.perf_counter()
            del batch_dict, embeds  # Free memory

        torch.cuda.empty_cache()  # Clear GPU memory
        all_uuids.extend(uuids)
        n_imgs_processed += len(uuids)
        
        if len(all_uuids) >= max_rows_per_file:
            out_file = os.path.join(
                embeddings_output_dir, 
                f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
            )
            save_emb_to_parquet(
                all_uuids,
                {f"emb_{key}": np.vstack(all_embeds[key]) for key in models},
                out_file
            )
            #logging.info(f"Saved {len(all_uuids)} embeddings to {out_file}")
            file_idx += 1
            all_embeds = {key: [] for key in models}
            all_uuids = []
            
        batch_stats.update({
            "cpu_to_gpu_s": t1 - t0,
            "gpu_inference_s": t2 - t1,
            "gpu_to_cpu_s": t3 - t2,
            "total_batch_s": t3 - t0,
        })
        all_batch_stats.append(batch_stats)
    
    # Save remaining embeddings
    if len(all_uuids) > 0:
        out_file = os.path.join(
            embeddings_output_dir, 
            f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
        )
        save_emb_to_parquet(
            all_uuids,
            {f"emb_{key}": np.vstack(all_embeds[key]) for key in models},
            out_file
        )
        #logging.info(f"Saved {len(all_uuids)} embeddings to {out_file}")

    # Stop profiling and save results
    usage_stop.set()
    usage_thread.join()

    elapsed = time.time() - start_time
    logging.info(f"Total images embedded: {n_imgs_processed}")
    logging.info(f"Total time taken: {format_time(elapsed)}")
    if n_imgs_processed > 0:
        logging.info(f"Avg time/image: {elapsed/n_imgs_processed:.4f} sec")
        logging.info(f"Throughput: {n_imgs_processed/elapsed:.2f} images/sec")
    
    profiling.log_computing_specs(
        profile_dir, 
        config.get("batch_size", 16), config.get("num_workers", 28),
        extra_info={
            "prefetch_factor": config.get("prefetch_factor", 16),
            "read_batch_size": config.get("read_batch_size", 128),
            "max_rows_per_file": config.get("max_rows_per_file", 10000),
            "task": "Image embedding",
            "model": [spec["name"] for spec in config["models"].values()],
            "throughput": f"{n_imgs_processed/elapsed:.2f} images/sec" if n_imgs_processed > 0 else "0 images/sec",
            "total_images": n_imgs_processed,
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
    parser = argparse.ArgumentParser(description="CLIP Batch Embedding with Config File or Command Line Arguments")
    parser.add_argument("target_dir", type=str, help="Directory containing input data")
    parser.add_argument("output_dir", type=str, help="Directory to save output embeddings")
    parser.add_argument("--input_type", type=str, required=True, choices=["images", "parquet"],
                        help="Type of input data: 'images' for image directory, 'parquet' for Parquet files")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")
    parser.add_argument("--file_list", type=str, default=None,
                        help="File containing list of Parquet files to process (only for --input_type parquet)")
    
    # Model configuration arguments (used when no config file provided)
    parser.add_argument("--model_name", type=str, default="ViT-B-32", 
                        help="OpenCLIP model name (default: ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights (default: openai)")
    
    # Compute arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=28, help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=16, help="Dataloader prefetch factor")
    parser.add_argument("--read_batch_size", type=int, default=128, help="Parquet read batch size")
    parser.add_argument("--max_rows_per_file", type=int, default=10000, help="Max rows per output file")
    parser.add_argument("--out_prefix", type=str, default="embed_results", help="Output file prefix")
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
            "models": {
                "default": {
                    "name": args.model_name,
                    "pretrained": args.pretrained
                }
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
            "stagger": args.stagger
        }
        print("Using command line arguments (no config file provided)")

    main(
        config,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        input_type=args.input_type,
        file_list=args.file_list
    )