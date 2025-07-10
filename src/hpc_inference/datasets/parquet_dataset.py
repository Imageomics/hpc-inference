import os
from collections import defaultdict
import random
import time
from datetime import datetime
import logging

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def assign_files_to_rank(rank, world_size, parquet_files, evenly_distribute=True):
    """
    Assign files to the current rank based on the world size.
    This method ensures that each rank gets a unique set of files to process.
    The files can be distributed evenly based on their size (LPT algorithm) or simply by their order.
    This is useful for large datasets where some files may be significantly larger than others.

    Args:
        rank (int): Current process rank.
        world_size (int): Total number of processes.
        parquet_files (List[str]): List of file paths.
        evenly_distribute (bool): Whether to distribute files evenly based on size.

    Returns:
        List[str]: File paths assigned to the given rank.
    """

    if not evenly_distribute:
        return parquet_files[rank::world_size]

    # Get file sizes
    file_sizes = [(f, os.path.getsize(f)) for f in parquet_files]
    # Sort files by size
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    assignments = defaultdict(list)
    load_per_rank = [0] * world_size

    for fpath, size in file_sizes:
        min_rank = load_per_rank.index(min(load_per_rank))
        assignments[min_rank].append(fpath)
        load_per_rank[min_rank] += size
    
    return assignments[rank]

def multi_model_collate(batch):
    """
    Collate function for batches where each sample is (uuid, {model_name: tensor, ...}).
    Returns:
        uuids: list of uuids
        batch_dict: dict of {model_name: batch_tensor}
    """
    uuids, processed_list = zip(*batch)  # unzip the batch
    batch_dict = {}
    # Use keys from the first processed dict (assumes all have the same keys)
    for key in processed_list[0]:
        # Stack tensors for each model key
        batch_dict[key] = torch.stack([d[key] for d in processed_list])
    return list(uuids), batch_dict

class ParquetImageDataset(IterableDataset):
    """
    An IterableDataset that reads images from Parquet files in a distributed manner.
    Each worker reads a subset of the files based on its rank and world size.
    The dataset yields tuples of (uuid, image) where uuid is the unique identifier
    for the image and image is a transformed tensor.
    The dataset can be used with PyTorch's DataLoader for distributed inference & training.
    
    In PyTorch, there are two main types of datasets you can use with `torch.utils.data.DataLoader`:
    1. `torch.utils.data.Dataset`: This is the standard dataset class that allows random access to data samples.
       It is suitable for datasets that can fit into memory or can be indexed efficiently.
       When using this type of dataset, the DataLoader can shuffle the data and load it in parallel using multiple workers.
    2. `torch.utils.data.IterableDataset`: This is a more flexible dataset class that allows you to define a dataset that can be iterated over.
       It is suitable for datasets that are too large to fit into memory or when the data is generated on-the-fly.
       When using this type of dataset, the DataLoader will not shuffle the data, and each worker will get a different subset of the data to process.
    
    Use `Dataset` for random-access data.
    Use `IterableDataset` for streaming or sequential data where random access is not possible or practical.
    
    Args:
        parquet_files (list): List of paths to Parquet files.
        col_uuid (str): Column name in the Parquet file that contains the UUIDs.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        decode_fn (callable): Function to decode images from the Parquet file.
        preprocess (callable): Preprocessing function to apply to the images.
        read_batch_size (int, optional): Number of rows to read from each Parquet file at once. Defaults to 100.
        processed_files_log (str, optional): Path to the log file for tracking processed files. Defaults to None.
        evenly_distribute (bool, optional): Whether to distribute files evenly based on size. Defaults to True.
        If False, files are distributed in a round-robin manner.
    """
    def __init__(
        self, 
        parquet_files, col_uuid, 
        rank, world_size, 
        decode_fn, preprocess, 
        read_batch_size=100,
        stagger=False,
        read_columns=None, 
        processed_files_log=None, evenly_distribute=True
    ):
        
        self.col_uuid = col_uuid
        def safe_decode_fn(row):
            try:
                return decode_fn(row)
            except Exception as e:
                logging.error(f"Error decoding row: {e}", exc_info=True)
                return None
        self.decode_fn = safe_decode_fn  # decode_image_to_pil function
        self.preprocess = preprocess 
        self.read_batch_size = read_batch_size
        self.read_columns = read_columns
        self.stagger = stagger  
        self.rank = rank # Store rank for logging
        self.world_size = world_size

        self.files = assign_files_to_rank(
            rank, world_size,
            parquet_files, 
            evenly_distribute=evenly_distribute
        )

        self.processed_files_log = processed_files_log or f"processed_files_rank{rank}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
        self.processed_files = self.load_processed_files()

        logging.info(f"[Rank {self.rank}] Assigned {len(self.files)} parquet files")

    def load_processed_files(self):
        """Load processed files from the log."""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, "r") as f:
                return set(f.read().splitlines())
        return set()

    def save_processed_file(self, file_path):
        """Save a processed file to the log file, creating the directory if needed."""
        log_dir = os.path.dirname(self.processed_files_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        with open(self.processed_files_log, "a") as f:
            f.write(f"{file_path}\n")

    def parse_row(self, row):
        image = self.decode_fn(row)
        if image is None:
            raise ValueError("Decoding failed, returned None")
        if isinstance(self.preprocess, dict):
            if len(self.preprocess) == 1:
                # Only one model, return tensor directly
                key = next(iter(self.preprocess))
                processed = self.preprocess[key](image)
            else:
                processed = {k: fn(image) for k, fn in self.preprocess.items()}
        else:
            processed = self.preprocess(image)
        uuid = row.get(self.col_uuid, 'UUID_MISSING')
        return uuid, processed

    def __iter__(self):
        """
        Iterate over the dataset, yielding (uuid, image) tuples.
        Each worker processes its assigned files and yields the results.
        The dataset is designed to be used with PyTorch's DataLoader for distributed processing.
        The function handles exceptions at various levels to ensure robust processing.
        It skips already processed files and logs errors for individual rows and batches.
        This allows for efficient and fault-tolerant processing of large datasets.
        
        Yields:
            tuple: A tuple containing the UUID and the preprocessed image tensor.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Staggered processing: each worker starts at a different offset
        if self.stagger:
            delay = min(random.uniform(1.0, 2.0) * worker_id, 20.0)
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Staggering start by {delay:.2f} seconds")
            time.sleep(delay)

        # Assign files to workers
        worker_files = self.files[worker_id::num_workers]

        for path in worker_files:
            if path in self.processed_files:
                logging.info(f"[Rank {self.rank}/Worker {worker_id}] Skipping already processed file: {path}")
                continue

            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {path}")
            try:
                pf = pq.ParquetFile(path)
                for batch_idx, batch in enumerate(pf.iter_batches(batch_size=self.read_batch_size, columns=self.read_columns)):
                    try:
                        df = batch.to_pandas()
                        for _, row in df.iterrows():
                            try:
                                yield self.parse_row(row)
                            except Exception as e:
                                uuid = row.get(self.col_uuid, 'UUID_UNKNOWN')
                                logging.error(f"[Rank {self.rank}/Worker {worker_id}] Error parsing row UUID={uuid} in {path}: {e}", exc_info=True)
                                continue
                    except Exception as e:
                        logging.error(f"[Rank {self.rank}/Worker {worker_id}] Error in batch {batch_idx} in file {path}: {e}", exc_info=True)
                        continue
                self.save_processed_file(path)  # Mark file as processed
            except Exception as e:
                logging.error(f"[Rank {self.rank}/Worker {worker_id}] Failed to open file {path}: {e}", exc_info=True)
                continue


class ParquetEmbeddingDataset(IterableDataset):
    """
    An IterableDataset that reads embeddings from Parquet files in a distributed manner.
    Each worker reads a subset of the files based on its rank and world size.
    The dataset yields tuples of (uuid, embedding) where uuid is the unique identifier
    for the embedding and embedding is a tensor.
    
    Args:
        parquet_files (list): List of paths to Parquet files.
        col_uuid (str): Column name in the Parquet file that contains the UUIDs.
        col_embedding (str): Column name in the Parquet file that contains the embeddings.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        read_batch_size (int, optional): Number of rows to read from each Parquet file at once. Defaults to 100.
        stagger (bool, optional): Whether to stagger the start of each worker. Defaults to False.
        processed_files_log (str, optional): Path to the log file for tracking processed files. Defaults to None.
        evenly_distribute (bool, optional): Whether to distribute files evenly based on size. Defaults to True.
    """
    def __init__(
        self, 
        parquet_files, col_uuid, col_embedding,
        rank, world_size,
        read_batch_size=100,
        stagger=False,
        processed_files_log=None, evenly_distribute=True
    ):
        
        self.col_uuid = col_uuid
        self.col_embedding = col_embedding
        self.read_batch_size = read_batch_size
        self.stagger = stagger  
        self.rank = rank # Store rank for logging
        self.world_size = world_size

        self.files = assign_files_to_rank(
            rank, world_size,
            parquet_files, 
            evenly_distribute=evenly_distribute
        )

        self.processed_files_log = processed_files_log or f"processed_files_rank{rank}.log"
        self.processed_files = self.load_processed_files()

        logging.info(f"[Rank {self.rank}] Assigned {len(self.files)} parquet files")

    def load_processed_files(self):
        """Load processed files from the log."""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, "r") as f:
                return set(f.read().splitlines())
        return set()

    def save_processed_file(self, file_path):
        """Save a processed file to the log file, creating the directory if needed."""
        log_dir = os.path.dirname(self.processed_files_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        with open(self.processed_files_log, "a") as f:
            f.write(f"{file_path}\n")
    
    
    def __iter__(self):
        """
        Iterate over the dataset, yielding (uuid, embedding) tuples.
        Each worker processes its assigned files and yields the results.
        The dataset is designed to be used with PyTorch's DataLoader for distributed processing.
        The function handles exceptions at various levels to ensure robust processing.
        It skips already processed files and logs errors for individual rows and batches.
        This allows for efficient and fault-tolerant processing of large datasets.
        
        Yields:
            tuple: A tuple containing the UUID and the embedding tensor.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Staggered processing: each worker starts at a different offset
        if self.stagger:
            delay = min(random.uniform(1.0, 2.0) * worker_id, 20.0)
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Staggering start by {delay:.2f} seconds")
            time.sleep(delay)
        
        # Assign files to workers
        worker_files = self.files[worker_id::num_workers]

        for path in worker_files:
            if path in self.processed_files:
                logging.info(f"[Rank {self.rank}/Worker {worker_id}] Skipping already processed file: {path}")
                continue

            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {path}")

            try:
                pf = pq.ParquetFile(path)
                for batch_idx, batch in enumerate(pf.iter_batches(batch_size=self.read_batch_size, columns=[self.col_uuid, self.col_embedding])):
                    try:
                        uuids = batch[self.col_uuid].to_pylist()
                        embeddings = torch.tensor(
                            batch[self.col_embedding].to_pylist(),
                            dtype=torch.float32
                        )
                        for uuid, embedding in zip(uuids, embeddings):
                            yield uuid, embedding
                    except Exception as e:
                        logging.error(f"[Rank {self.rank}/Worker {worker_id}] Error in batch {batch_idx} in file {path}: {e}", exc_info=True)
                        continue
                self.save_processed_file(path)  # Mark file as processed
            except Exception as e:
                logging.error(f"[Rank {self.rank}/Worker {worker_id}] Failed to open file {path}: {e}", exc_info=True)
                continue
