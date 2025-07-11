import os
import time
import json
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import psutil
import pynvml
import pandas as pd
import matplotlib.pyplot as plt
import torch

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def log_computing_specs(
    profile_dir: Union[str, Path], 
    batch_size: int, 
    num_workers: int, 
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log computing specifications (CPU, GPU, memory) to a JSON file.
    
    This function captures detailed hardware specifications including CPU count,
    memory, GPU information, and training parameters for performance analysis.
    
    Args:
        profile_dir: Directory where the computing specs JSON file will be saved.
        batch_size: Batch size used for training/inference.
        num_workers: Number of data loader workers.
        extra_info: Additional information to include in the specs (e.g., model name, dataset size).
            
    Examples:
        >>> log_computing_specs("/path/to/profile", 32, 8, {"model": "ResNet50"})
        # Creates /path/to/profile/computing_specs.json with hardware info
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    specs: Dict[str, Any] = {}

    # CPU info
    specs["cpu_count"] = psutil.cpu_count(logical=True)  # total logical CPUs on the machine.
    specs['cpu_count_available'] = os.cpu_count()        # CPUs available to the current process.
    specs["cpu_freq"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    specs["memory_GB"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    # GPU info
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            gpu_count = torch.cuda.device_count()
            specs["gpu_count"] = gpu_count
            specs["gpus"] = []
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                specs["gpus"].append({
                    "index": i,
                    "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                    "total_mem_GB": round(mem.total / (1024 ** 3), 2)
                })
            pynvml.nvmlShutdown()
        except Exception as e:
            logging.warning(f"Failed to get GPU info: {e}")
            specs["gpu_count"] = 0
            specs["gpus"] = []
    else:
        specs["gpu_count"] = 0
        specs["gpus"] = []

    # Batch and worker info
    specs["batch_size"] = batch_size
    specs["num_workers"] = num_workers

    # Any extra info (e.g., model name, dataset size)
    if extra_info:
        specs.update(extra_info)

    # Save as JSON
    specs_file = profile_dir / "computing_specs.json"
    with open(specs_file, "w") as f:
        json.dump(specs, f, indent=2)
    
    logging.info(f"Computing specs saved to {specs_file}")


def start_usage_logging(
    usage_log: List[Tuple[float, float, float, float]], 
    stop_event: threading.Event, 
    interval: float = 0.5, 
    gpu_index: int = 0
) -> None:
    """
    Start logging system resource usage (CPU, GPU) in a separate thread.
    
    This function continuously monitors and logs CPU utilization, GPU utilization,
    and GPU memory usage until the stop event is set.
    
    Args:
        usage_log: List to store usage data tuples (timestamp, cpu_percent, gpu_util, gpu_mem_MB).
        stop_event: Threading event to signal when to stop logging.
        interval: Time interval between measurements in seconds. Defaults to 0.5.
        gpu_index: GPU index to monitor. Defaults to 0.
        
    Note:
        This function should be run in a separate thread. It will block until stop_event is set.
        
    Examples:
        >>> import threading
        >>> usage_log = []
        >>> stop_event = threading.Event()
        >>> thread = threading.Thread(target=start_usage_logging, args=(usage_log, stop_event))
        >>> thread.start()
        >>> # ... do some work ...
        >>> stop_event.set()
        >>> thread.join()
    """
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
        logging.info(f"Started usage logging for GPU {gpu_index} with {interval}s interval")
        
        while not stop_event.is_set():
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6  # MB
            except Exception as e:
                logging.warning(f"Failed to get GPU stats: {e}")
                gpu_util = 0.0
                gpu_mem_mb = 0.0
                
            usage_log.append((timestamp, cpu_percent, gpu_util, gpu_mem_mb))
            time.sleep(interval)
            
    except Exception as e:
        logging.error(f"Error in usage logging: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore shutdown errors
        logging.info("Usage logging stopped")


def save_batch_stats(
    all_batch_stats: List[Dict[str, Any]], 
    profile_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Save batch statistics to CSV and return as DataFrame.
    
    Args:
        all_batch_stats: List of dictionaries containing batch timing and performance stats.
        profile_dir: Directory where the profile log CSV will be saved.
        
    Returns:
        DataFrame containing the batch statistics.
        
    Examples:
        >>> batch_stats = [
        ...     {"batch": 0, "total_batch_s": 0.1, "gpu_inference_s": 0.05},
        ...     {"batch": 1, "total_batch_s": 0.09, "gpu_inference_s": 0.04}
        ... ]
        >>> df = save_batch_stats(batch_stats, "/path/to/profile")
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    stats_df = pd.DataFrame(all_batch_stats)
    
    csv_file = profile_dir / "profile_log.csv"
    stats_df.to_csv(csv_file, index=False)
    
    logging.info(f"Batch stats saved to {csv_file} ({len(stats_df)} batches)")
    return stats_df


def save_usage_log(
    usage_log: List[Tuple[float, float, float, float]], 
    profile_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Save system usage log to CSV and return as DataFrame.
    
    Args:
        usage_log: List of tuples containing (timestamp, cpu_percent, gpu_util, gpu_mem_MB).
        profile_dir: Directory where the usage log CSV will be saved.
        
    Returns:
        DataFrame containing the usage data with normalized timestamps.
        
    Examples:
        >>> usage_log = [(1234567890.0, 50.0, 80.0, 1024.0), ...]
        >>> df = save_usage_log(usage_log, "/path/to/profile")
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    usage_df = pd.DataFrame(usage_log, columns=["timestamp", "cpu_percent", "gpu_util", "gpu_mem_MB"])
    
    # Normalize timestamps to start from 0
    if not usage_df.empty:
        usage_df["timestamp"] -= usage_df["timestamp"].iloc[0]
    
    csv_file = profile_dir / "usage_log.csv"
    usage_df.to_csv(csv_file, index=False)
    
    logging.info(f"Usage log saved to {csv_file} ({len(usage_df)} measurements)")
    return usage_df


def save_usage_plots(usage_df: pd.DataFrame, profile_dir: Union[str, Path]) -> None:
    """
    Save system usage plots (CPU/GPU utilization and GPU memory) to PNG files.
    
    Args:
        usage_df: DataFrame containing usage data with columns:
            ["timestamp", "cpu_percent", "gpu_util", "gpu_mem_MB"]
        profile_dir: Directory where the plot PNG files will be saved.
        
    Examples:
        >>> usage_df = pd.DataFrame({
        ...     "timestamp": [0, 1, 2],
        ...     "cpu_percent": [50, 60, 55],
        ...     "gpu_util": [80, 85, 75],
        ...     "gpu_mem_MB": [1024, 1200, 1100]
        ... })
        >>> save_usage_plots(usage_df, "/path/to/profile")
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    if usage_df.empty:
        logging.warning("Empty usage DataFrame, skipping plots")
        return
    
    # Plot CPU & GPU Utilization vs. Time
    plt.figure(figsize=(12, 5))
    plt.plot(usage_df["timestamp"], usage_df["cpu_percent"], label="CPU Util (%)", linewidth=1.5)
    plt.plot(usage_df["timestamp"], usage_df["gpu_util"], label="GPU Util (%)", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 100) 
    plt.legend()
    plt.title("CPU and GPU Utilization Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    cpu_gpu_plot = profile_dir / "cpu_gpu_usage_plot.png"
    plt.savefig(cpu_gpu_plot, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    # Plot GPU memory usage
    plt.figure(figsize=(12, 4))
    plt.plot(usage_df["timestamp"], usage_df["gpu_mem_MB"], label="GPU Memory (MB)", 
             color='red', linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Memory (MB)")
    plt.legend()
    plt.title("GPU Memory Usage Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    gpu_mem_plot = profile_dir / "gpu_mem_usage_plot.png"
    plt.savefig(gpu_mem_plot, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    logging.info(f"Usage plots saved to {cpu_gpu_plot} and {gpu_mem_plot}")


def save_batch_timings_plot(stats_df: pd.DataFrame, profile_dir: Union[str, Path]) -> None:
    """
    Save batch timing breakdown plots to PNG file.
    
    Creates both linear and logarithmic scale plots showing the breakdown of batch processing times
    including total batch time, GPU inference time, and data transfer times.
    
    Args:
        stats_df: DataFrame containing batch statistics with timing columns:
            ["total_batch_s", "gpu_inference_s", "cpu_to_gpu_s", "gpu_to_cpu_s"]
        profile_dir: Directory where the timing plot PNG file will be saved.
        
    Examples:
        >>> stats_df = pd.DataFrame({
        ...     "total_batch_s": [0.1, 0.09, 0.11],
        ...     "gpu_inference_s": [0.05, 0.04, 0.06],
        ...     "cpu_to_gpu_s": [0.02, 0.02, 0.02],
        ...     "gpu_to_cpu_s": [0.01, 0.01, 0.01]
        ... })
        >>> save_batch_timings_plot(stats_df, "/path/to/profile")
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    if stats_df.empty:
        logging.warning("Empty stats DataFrame, skipping timing plots")
        return
    
    # Check if required columns exist
    required_cols = ['total_batch_s', 'gpu_inference_s', 'cpu_to_gpu_s', 'gpu_to_cpu_s']
    missing_cols = [col for col in required_cols if col not in stats_df.columns]
    if missing_cols:
        logging.warning(f"Missing timing columns {missing_cols}, skipping timing plots")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Prepare data
    batch_indices = range(len(stats_df))
    
    # Linear scale subplot
    axes[0].plot(batch_indices, stats_df['total_batch_s'], label="Total batch time", linewidth=1.5)
    axes[0].plot(batch_indices, stats_df['gpu_inference_s'], label="GPU inference time", linewidth=1.5)
    axes[0].plot(batch_indices, stats_df['cpu_to_gpu_s'], label="CPU→GPU transfer", linewidth=1.5)
    axes[0].plot(batch_indices, stats_df['gpu_to_cpu_s'], label="GPU→CPU transfer", linewidth=1.5)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Batch Timing Breakdown (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale subplot
    axes[1].plot(batch_indices, stats_df['total_batch_s'], label="Total batch time", linewidth=1.5)
    axes[1].plot(batch_indices, stats_df['gpu_inference_s'], label="GPU inference time", linewidth=1.5)
    axes[1].plot(batch_indices, stats_df['cpu_to_gpu_s'], label="CPU→GPU transfer", linewidth=1.5)
    axes[1].plot(batch_indices, stats_df['gpu_to_cpu_s'], label="GPU→CPU transfer", linewidth=1.5)
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Time (s, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Batch Timing Breakdown (Log Scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timing_plot = profile_dir / "batch_timing_plot.png"
    plt.savefig(timing_plot, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    logging.info(f"Batch timing plot saved to {timing_plot}")

