import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Union, Dict, Callable, List, Any, Tuple, Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
import pyarrow.parquet as pq
import pandas as pd

import logging
from ..utils.distributed import assign_files_to_rank

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class ParquetImageDataset(IterableDataset):
    """
    Loads images from Parquet files in a streaming fashion, with support for distributed processing.
    - Reads image data from Parquet files containing encoded image bytes
    - Supports distributed processing with rank-based file partitioning
    - Supports multi-worker data loading within each rank
    - Handles both single-model and multi-model preprocessing
    - Provides staggered worker starts and load balancing
    
    Returns (uuid, processed_data) tuples where:
    - uuid: Unique identifier from the UUID column in Parquet
    - processed_data: Tensor (single model) or dict of tensors (multi-model)
    """

    def __init__(
        self,
        parquet_files: List[Union[str, Path]],
        col_uuid: str = "uuid",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        evenly_distribute: bool = True,
        decode_fn: Optional[Callable] = None,
        preprocess: Optional[Union[Callable, Dict[str, Callable]]] = None,
        read_batch_size: int = 128,
        read_columns: Optional[List[str]] = None,
        stagger: bool = False,
        processed_files_log: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Args:
            parquet_files: List of paths to Parquet files containing image data.
            col_uuid: Name of the UUID column in Parquet files. Defaults to "uuid".
            rank: Current process rank for distributed processing.
            world_size: Total number of processes for distributed processing.
            evenly_distribute: Whether to distribute files evenly based on size. Defaults to True.
                If False, files are distributed in a round-robin manner.
            decode_fn: Function to decode image bytes to PIL Image. Required for image processing.
            preprocess: Transform(s) to apply to images.
                - If callable: single model preprocessing
                - If dict: {model_name: preprocess_fn} for multi-model
                - If None: return decoded image as-is
            read_batch_size: Number of rows to read from Parquet at a time. Defaults to 128.
            read_columns: List of column names to read from Parquet. If None, reads all columns.
                Typically includes ["uuid", "image", "original_size", "resized_size"].
            stagger: Whether to stagger the start of each worker. Defaults to False.
            processed_files_log: Path to log file for tracking processed files. Optional.
        """
        def safe_decode_fn(row):
            try:
                return decode_fn(row)
            except Exception as e:
                logging.error(f"Error decoding row: {e}", exc_info=True)
                return None

        self.parquet_files: List[str] = [str(f) for f in parquet_files]
        self.col_uuid: str = col_uuid
        self.rank: int = rank or 0
        self.world_size: int = world_size or 1
        self.decode_fn: Optional[Callable] = safe_decode_fn
        self.preprocess: Optional[Union[Callable, Dict[str, Callable]]] = preprocess
        self.read_batch_size: int = read_batch_size
        self.read_columns: Optional[List[str]] = read_columns
        self.stagger: bool = stagger
        self.processed_files_log: Optional[str] = str(processed_files_log) if processed_files_log else None

        # Apply rank-based file partitioning
        if self.world_size > 1:
            self.assigned_files = assign_files_to_rank(
                self.rank, self.world_size, self.parquet_files, evenly_distribute
            )
        else:
            self.assigned_files = self.parquet_files

        logging.info(f"Rank {self.rank} assigned {len(self.assigned_files)} out of {len(self.parquet_files)} Parquet files")

        # Validate required parameters
        if self.decode_fn is None:
            logging.warning("No decode_fn provided. Images will not be decoded from bytes.")

    def log_processed_file(self, file_path: str) -> None:
        """Log a processed file to the log file if configured."""
        if self.processed_files_log:
            try:
                with open(self.processed_files_log, 'a') as f:
                    f.write(f"{datetime.now().isoformat()}: {file_path}\n")
            except Exception as e:
                logging.error(f"Failed to log processed file {file_path}: {e}")

    def parse_batch_data(self, batch_df: pd.DataFrame) -> Iterator[Tuple[str, Any]]:
        """
        Parse a batch of data from Parquet DataFrame.
        
        Args:
            batch_df: DataFrame containing batch data from Parquet file.
            
        Yields:
            Tuples of (uuid, processed_data) for each row in the batch.
        """
        for _, row in batch_df.iterrows():
            try:
                uuid = str(row[self.col_uuid])
                
                # Decode image if decode function is provided
                if self.decode_fn and 'image' in row:
                    img = self.decode_fn(row)
                else:
                    img = row.get('image', None)
                
                # Apply preprocessing
                if img is None:
                    logging.warning(f"No image data found for UUID {uuid}")
                    continue
                    
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
                logging.error(f"[Rank {self.rank}] Error processing row with UUID {row.get(self.col_uuid, 'unknown')}: {e}")
                continue

    def process_parquet_file(self, file_path: str) -> Iterator[Tuple[str, Any]]:
        """
        Process a single Parquet file and yield processed data.
        
        Args:
            file_path: Path to the Parquet file to process.
            
        Yields:
            Tuples of (uuid, processed_data) for each valid row in the file.
        """
        try:
            # Read Parquet file in batches
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.read_batch_size):
                # Convert to pandas DataFrame
                batch_df = batch.to_pandas()
                
                # Filter columns if specified
                if self.read_columns:
                    available_columns = [col for col in self.read_columns if col in batch_df.columns]
                    if len(available_columns) != len(self.read_columns):
                        missing = set(self.read_columns) - set(available_columns)
                        logging.warning(f"Missing columns in {file_path}: {missing}")
                    batch_df = batch_df[available_columns]
                
                # Process batch data
                yield from self.parse_batch_data(batch_df)
                
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Error processing Parquet file {file_path}: {e}")
            return
        finally:
            # Log processed file
            self.log_processed_file(file_path)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Iterate over Parquet files and yield processed data.
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
        
        logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing {len(worker_files)} Parquet files")

        # Process each assigned file
        for file_path in worker_files:
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {file_path}")
            yield from self.process_parquet_file(file_path)



class ParquetEmbeddingDataset(IterableDataset):
    """
    Loads pre-computed embeddings from Parquet files in a streaming fashion.
    - Reads embedding vectors from Parquet files
    - Supports distributed processing with rank-based file partitioning  
    - Supports multi-worker data loading within each rank
    - Optimized for loading numerical data (embeddings) rather than images
    
    Returns (uuid, embedding) tuples where:
    - uuid: Unique identifier from the UUID column in Parquet
    - embedding: Numpy array or tensor containing the embedding vector
    """

    def __init__(
        self,
        parquet_files: List[Union[str, Path]],
        col_uuid: str = "uuid",
        col_embedding: str = "embedding",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        evenly_distribute: bool = True,
        read_batch_size: int = 1000,
        read_columns: Optional[List[str]] = None,
        stagger: bool = False,
        return_tensor: bool = True
    ) -> None:
        """
        Args:
            parquet_files: List of paths to Parquet files containing embedding data.
            col_uuid: Name of the UUID column in Parquet files. Defaults to "uuid".
            col_embedding: Name of the embedding column in Parquet files. Defaults to "embedding".
            rank: Current process rank for distributed processing.
            world_size: Total number of processes for distributed processing.
            evenly_distribute: Whether to distribute files evenly based on size. Defaults to True.
                If False, files are distributed in a round-robin manner.
            read_batch_size: Number of rows to read from Parquet at a time. Defaults to 1000.
            read_columns: List of column names to read from Parquet. If None, reads all columns.
                Typically includes ["uuid", "embedding"].
            stagger: Whether to stagger the start of each worker. Defaults to False.
            return_tensor: If True, convert embeddings to PyTorch tensors. If False, keep as numpy arrays.
        """
        self.parquet_files: List[str] = [str(f) for f in parquet_files]
        self.col_uuid: str = col_uuid
        self.col_embedding: str = col_embedding
        self.rank: int = rank or 0
        self.world_size: int = world_size or 1
        self.read_batch_size: int = read_batch_size
        self.read_columns: Optional[List[str]] = read_columns
        self.stagger: bool = stagger
        self.return_tensor: bool = return_tensor

        # Apply rank-based file partitioning
        if self.world_size > 1:
            self.assigned_files = assign_files_to_rank(
                self.rank, self.world_size, self.parquet_files, evenly_distribute
            )
        else:
            self.assigned_files = self.parquet_files

        logging.info(f"Rank {self.rank} assigned {len(self.assigned_files)} out of {len(self.parquet_files)} Parquet files")

    def process_parquet_file(self, file_path: str) -> Iterator[Tuple[str, Any]]:
        """
        Process a single Parquet file and yield embedding data.
        
        Args:
            file_path: Path to the Parquet file to process.
            
        Yields:
            Tuples of (uuid, embedding) for each row in the file.
        """
        try:
            # Read Parquet file in batches
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.read_batch_size):
                # Convert to pandas DataFrame
                batch_df = batch.to_pandas()
                
                # Filter columns if specified
                if self.read_columns:
                    available_columns = [col for col in self.read_columns if col in batch_df.columns]
                    if len(available_columns) != len(self.read_columns):
                        missing = set(self.read_columns) - set(available_columns)
                        logging.warning(f"Missing columns in {file_path}: {missing}")
                    batch_df = batch_df[available_columns]
                
                # Process each row
                for _, row in batch_df.iterrows():
                    try:
                        uuid = str(row[self.col_uuid])
                        embedding = row[self.col_embedding]
                        
                        # Convert to tensor if requested
                        if self.return_tensor and hasattr(torch, 'from_numpy'):
                            embedding = torch.from_numpy(embedding)
                        
                        yield uuid, embedding
                        
                    except Exception as e:
                        logging.error(f"[Rank {self.rank}] Error processing row with UUID {row.get(self.col_uuid, 'unknown')}: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Error processing Parquet file {file_path}: {e}")
            return

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Iterate over Parquet files and yield embedding data.
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
        
        logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing {len(worker_files)} Parquet files")

        # Process each assigned file
        for file_path in worker_files:
            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {file_path}")
            yield from self.process_parquet_file(file_path)
