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
import pandas as pd
import json

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from ...datasets.parquet_dataset import ParquetImageDataset
from ...datasets.image_folder_dataset import ImageFolderDataset
from ...utils.common import format_time, decode_image
from ...utils import profiling
from ...utils.distributed import pil_image_collate

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

PRETRAINED_MODEL_NAME = "google/owlv2-base-patch16-ensemble"


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x

def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_cpu(x) for x in obj]
    else:
        return obj

def save_detection_results(uuids: List[str], detection_results: List[Dict[str, Any]], output_file: str) -> None:
    
    """
    Save detection results to a Parquet file using PyArrow.
    
    Args:
        uuids: List of image UUIDs
        detection_results: List of detection result dicts corresponding to each image
        output_file: Path to the output Parquet file
    """
    assert len(uuids) == len(detection_results), "UUIDs and detection results must have the same length"

    table = parse_detections_to_arrow_table(detection_results, uuids, score_threshold=None)

    pq.write_table(table, output_file)
    logging.info(f"Saved detection results to {output_file}")

def parse_detections_to_arrow_table(
    results: List[Dict[str, Any]],
    uuids: List[str],
    score_threshold: Optional[float] = None,
) -> pa.Table:
    """
    Parse detection results into a PyArrow Table.
    
    Args:
        results: list of dicts (one per image) with keys: 'scores','labels','boxes','text_labels'
        uuids: list[str], same length/order as results
        score_threshold: optional float to drop low-score boxes (None = keep all)

    Returns: 
        PyArrow Table with schema:
          - image_uuid: string
          - detection_id: int32 (nullable)
          - score: float32
          - text_label: string
          - box: list<float32>  # [x1,y1,x2,y2]
    """
    assert len(results) == len(uuids), "results and uuids must align"

    # Collect columns as lists
    col_image_uuid = []
    col_detection_id = []
    col_score = []
    col_text_label = []
    col_box = []

    for img_idx, (uuid, det) in enumerate(zip(uuids, results)):
        scores = _to_list(det.get("scores", [])) or []
        labels = _to_list(det.get("labels", [])) or []
        boxes  = _to_list(det.get("boxes",  [])) or []
        names  = det.get("text_labels", None)     # may be per-box or class list

        n = len(scores)

        # No detections: add a null row
        if n == 0:
            col_image_uuid.append(uuid)
            col_detection_id.append(None)
            col_score.append(None)
            col_text_label.append(None)
            col_box.append(None)
            continue

        has_valid_detection = False
        for k in range(n):
            s = float(scores[k])
            if score_threshold is not None and s < score_threshold:
                continue

            has_valid_detection = True

            # label text: handle per-box names vs class-list mapping
            label_txt = None
            if names is not None:
                if len(names) == n:
                    label_txt = names[k]                   # per-box list
                elif k < len(labels):
                    label_idx = int(labels[k])
                    if 0 <= label_idx < len(names):
                        label_txt = names[label_idx]       # class list
            # fallback if no names:
            if label_txt is None and k < len(labels):
                label_txt = str(int(labels[k]))

            box = boxes[k]  # [x1,y1,x2,y2] (xyxy)
            # ensure plain python list of floats
            box_list = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

            col_image_uuid.append(uuid)
            col_detection_id.append(k)
            col_score.append(s)
            col_text_label.append(label_txt)
            col_box.append(box_list)

        # If all detections were filtered out by threshold, still record a null row
        if not has_valid_detection:
            col_image_uuid.append(uuid)
            col_detection_id.append(None)
            col_score.append(None)
            col_text_label.append(None)
            col_box.append(None)

    # Create PyArrow arrays with proper types
    arrays = [
        pa.array(col_image_uuid, type=pa.string()),
        pa.array(col_detection_id, type=pa.int32()),
        pa.array(col_score, type=pa.float32()),
        pa.array(col_text_label, type=pa.string()),
        pa.array(col_box, type=pa.list_(pa.float32())),
    ]

    # Create schema
    schema = pa.schema([
        ("image_uuid", pa.string()),
        ("detection_id", pa.int32()),
        ("score", pa.float32()),
        ("text_label", pa.string()),
        ("box", pa.list_(pa.float32())),
    ])

    return pa.Table.from_arrays(arrays, schema=schema)


@torch.inference_mode()
def main(
    config: Dict[str, Any], 
    target_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    input_type: str,
    file_list: Optional[Union[str, Path]] = None
) -> None:
    
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


    processor = Owlv2Processor.from_pretrained(PRETRAINED_MODEL_NAME)
    model = Owlv2ForObjectDetection.from_pretrained(PRETRAINED_MODEL_NAME)
    model.to(device)
    torch.compile(model)
    model.eval()
    
    # Create dataset based on input type
    if input_type == "images":
        logging.info(f"Processing image directory: {target_dir}")
        
        dataset = ImageFolderDataset(
            target_dir, 
            preprocess=None,
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
            preprocess=None,  
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
        prefetch_factor=config.get("prefetch_factor", 16),
        collate_fn=pil_image_collate
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
    
    max_uuids_per_file = config.get("max_uuids_per_file", 10000)
    out_prefix = config.get("out_prefix", "OWLv2_detection_results")
    conf_threshold = config.get("confidence_threshold", 0.1)
    text_label_list = config.get("text_labels", ["fish"])
    
    for batch_idx, (uuids, images) in enumerate(loader):
        batch_stats = {"batch": batch_idx, "batch_size": len(uuids)}
        
        text_labels = [text_label_list for _ in range(len(uuids))]
        target_sizes = [img.size[::-1] for img in images]
        
        t0 = time.perf_counter()
        inputs = processor(text=text_labels, images=images, return_tensors="pt").to(device)
        t1 = time.perf_counter()
        
        # Zero-shot detection
        outputs = model(**inputs)
        detection_results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold, text_labels=text_labels
        )
        detection_results_cpu = _to_cpu(detection_results)
        t2 = time.perf_counter()

        all_detection_results.extend(detection_results_cpu)
        all_uuids.extend(uuids)
        n_imgs_processed += len(uuids)

        del inputs, outputs, detection_results
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
         # Save results when reaching max_uuids_per_file
        if len(all_uuids) >= max_uuids_per_file:
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
            "total_batch_s": t2 - t0
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
    
    logging.info(f"Total images processed: {n_imgs_processed}")
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
            "max_uuids_per_file": config.get("max_uuids_per_file", 10000),
            "task": "OWLv2 Zero-Shot Object Detection",
            "model": PRETRAINED_MODEL_NAME,
            "text_labels": text_label_list,
            "confidence_threshold": conf_threshold,
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
    from ...utils.common import load_config
    
    parser = argparse.ArgumentParser(description="OWLv2 Zero-Shot Object Detection with Config File or Command Line Arguments")
    parser.add_argument("target_dir", type=str, help="Directory containing input data")
    parser.add_argument("output_dir", type=str, help="Directory to save output detection results")
    parser.add_argument("--input_type", type=str, required=True, choices=["images", "parquet"],
                        help="Type of input data: 'images' for image directory, 'parquet' for Parquet files")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")
    parser.add_argument("--file_list", type=str, default=None,
                        help="File containing list of Parquet files to process (only for --input_type parquet)")
    
    # Model configuration arguments (used when no config file provided)
    parser.add_argument("--text_labels", type=str, nargs="+", default=["fish"],
                        help="Text labels for zero-shot detection (default: fish)")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                        help="Confidence threshold for detection (default: 0.1)")
    
    # Compute arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=20, help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=16, help="Dataloader prefetch factor")
    parser.add_argument("--read_batch_size", type=int, default=128, help="Parquet read batch size")
    parser.add_argument("--max_uuids_per_file", type=int, default=10000, help="Max UUIDs per output file")
    parser.add_argument("--out_prefix", type=str, default="OWLv2_detection_results", help="Output file prefix")
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
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "read_batch_size": args.read_batch_size,
            "max_uuids_per_file": args.max_uuids_per_file,
            "out_prefix": args.out_prefix,
            "read_columns": args.read_columns,
            "validate_images": args.validate_images,
            "uuid_mode": args.uuid_mode,
            "evenly_distribute": args.evenly_distribute,
            "stagger": args.stagger,
            "confidence_threshold": args.confidence_threshold,
            "text_labels": args.text_labels
        }
        print("Using command line arguments (no config file provided)")

    main(
        config,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        input_type=args.input_type,
        file_list=args.file_list
    )