import time
import threading
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.inference.classify.similarity import get_predictions
from src.datasets.ParquetIterableDataset import ParquetEmbeddingDataset
from src.utils import format_time
import src.profiling as profiling

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def save_preds_to_parquet(
    all_preds, path: str, compression="zstd"
):
    
    uuids = []
    pred_labels = []
    pred_scores = []
    for preds in all_preds:
        uuids.extend(preds["uuids"])
        pred_labels.extend(preds["pred_label"])
        pred_scores.extend(preds["pred_score"])
    
    table = pa.table({
        "uuid": uuids,
        "pred_label": pred_labels,
        "pred_score": pred_scores
    })

    pq.write_table(table, path, compression=compression)

@torch.no_grad()
def main(
    target_dir, output_dir, class_embeddings_dict_path,
    file_list=None,  # File list to process, if None, process all Parquet files in target_dir
    batch_size=8, num_workers=24, prefetch_factor=4, read_batch_size=640,
    max_rows_per_file=10000,  # Max rows per results
    out_prefix="clip_embed_results",
    profile_dir="profile_results"
):
    # =============== #
    # ---- Setup ----
    # =============== #
    ## Timer
    start_time = time.time()
    
    ## Rank and world size
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = 0 # int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    logging.info(f"Global rank: {global_rank}, Local rank: {local_rank}, World size: {world_size}")     

    ## Output directory
    output_dir = os.path.abspath(os.path.join(output_dir, f"rank_{global_rank}"))
    os.makedirs(output_dir, exist_ok=True)
    profile_dir = os.path.abspath(os.path.join(profile_dir, f"rank_{global_rank}"))
    os.makedirs(profile_dir, exist_ok=True)

    ## Find all Parquet files in the target directory
    parquet_files = []
    if file_list is None:
        parquet_files = [str(p) for p in Path(target_dir).rglob('*.parquet')]
    else:
        if os.path.exists(file_list):
            with open(file_list, "r") as f:
                parquet_files = [line.strip() for line in f if line.strip().endswith('.parquet')]
    
    dataset = ParquetEmbeddingDataset(
        parquet_files,
        col_uuid="uuid",
        col_embedding="embedding",
        rank=global_rank, world_size=world_size, 
        evenly_distribute=True,
        read_batch_size=read_batch_size,
        processed_files_log = None
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
    )

    all_preds = []
    all_uuids = []
    all_batch_stats = []
    usage_log = []
    file_idx = 0
    n_processed = 0

    ## Start CPU/GPU usage logging in a background thread
    usage_stop = threading.Event()
    usage_thread = threading.Thread(
        target=profiling.start_usage_logging, 
        args=(usage_log, usage_stop, 0.5, 0)
    )
    usage_thread.start()

    # ==================== #
    # ---- Similarity ----
    # ==================== #

    class_embeddings_dict = torch.load(class_embeddings_dict_path)
    class_labels = list(class_embeddings_dict.keys())
    class_embeddings = torch.stack(
        list(class_embeddings_dict.values())
    ).squeeze(1).to(f"cuda:{local_rank}", non_blocking=True, dtype=torch.float32)




    for batch_idx, (uuids, embeddings) in enumerate(loader):
        batch_stats = {"batch": batch_idx, "batch_size": len(uuids)}

        # 1. CPU->GPU transfer
        t0 = time.perf_counter()
        embeddings = embeddings.to(f"cuda:{local_rank}", non_blocking=True)
        t1 = time.perf_counter()

        # 2. Similarity computation + GPU->CPU transfer
        predictions = get_predictions(
            input_embeddings=embeddings,
            class_embeddings=class_embeddings,
            class_labels=class_labels,
            uuid_list=uuids,
            device=f"cuda:{local_rank}"
        )
        t2 = time.perf_counter()

        all_preds.append(predictions)
        all_uuids.extend(predictions["uuids"])
        n_processed += len(uuids)

        # Save predictions to file if we reach the max rows per file
        if len(all_uuids) >= max_rows_per_file:
            out_file = os.path.join(
                output_dir, 
                f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
            )


            save_preds_to_parquet(
                all_preds, 
                out_file,
                compression="zstd"
            )
            
            logging.info(f"Saved {len(all_uuids)} predictions to {out_file}")
            
            all_preds = []
            all_uuids = []
            file_idx += 1
        
        # Log batch stats
        batch_stats.update({
            "cpu_to_gpu_s": t1 - t0,
            "total_batch_s": t2 - t1
        })
        all_batch_stats.append(batch_stats)
    
    # If there are remaining predictions after the loop, save them
    if len(all_uuids) > 0:
        out_file = os.path.join(
            output_dir, 
            f"{out_prefix}_rank_{global_rank}_{file_idx}.parquet"
        )
        save_preds_to_parquet(
            all_preds, 
            out_file,
            compression="zstd"
        )
        logging.info(f"Saved {len(all_uuids)} predictions to {out_file}")
        all_preds = []
        all_uuids = []
        file_idx += 1
    
    # Stop the usage logging thread
    usage_stop.set()
    usage_thread.join()

    # Log final stats
    elapsed = time.time() - start_time
    logging.info(f"Processed {n_processed} rows in {format_time(elapsed)}")
    logging.info(f"Total time taken: {format_time(elapsed)}")
    logging.info(f"Avg time/image: {elapsed/n_processed:.4f} sec")
    logging.info(f"Throughput: {n_processed/elapsed:.2f} images/sec")


    profiling.log_computing_specs(
        profile_dir, 
        batch_size, num_workers,
        extra_info={
            "prefetch_factor": prefetch_factor,
            "task": "Classification",
            "throughput": f"{n_processed/elapsed:.2f} images/sec",
            "total_processed": n_processed,
            "total_time_s": elapsed
        }
    )

    stats_df = profiling.save_batch_stats(all_batch_stats, profile_dir)
    usage_df = profiling.save_usage_log(usage_log, profile_dir)
    profiling.save_usage_plots(usage_df, profile_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run classification inference on Parquet files.")
    parser.add_argument("target_dir", type=str, help="Directory containing Parquet files.")
    parser.add_argument("output_dir", type=str, help="Directory to save output results.")
    parser.add_argument("class_embeddings_dict_path", type=str, help="Path to the class embeddings dictionary.")
    parser.add_argument("--file_list", type=str, default=None, help="File list to process, if None, process all Parquet files in target_dir.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of workers for DataLoader.")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor for DataLoader.")
    parser.add_argument("--read_batch_size", type=int, default=640, help="Batch size for reading Parquet files.")
    parser.add_argument("--max_rows_per_file", type=int, default=10000, help="Max rows per output file.")
    parser.add_argument("--out_prefix", type=str, default="clip_embed_results", help="Prefix for output files.")
    parser.add_argument("--profile_dir", type=str, default="profile_results", help="Directory to save profiling results.")

    args = parser.parse_args()

    main(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        class_embeddings_dict_path=args.class_embeddings_dict_path,
        file_list=args.file_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        read_batch_size=args.read_batch_size,
        max_rows_per_file=args.max_rows_per_file,
        out_prefix=args.out_prefix,
        profile_dir=args.profile_dir
    )




        