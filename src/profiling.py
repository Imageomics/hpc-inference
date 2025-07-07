import psutil
import pynvml
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import json

def log_computing_specs(profile_dir, batch_size, num_workers, extra_info=None):
    os.makedirs(profile_dir, exist_ok=True)
    specs = {}

    # CPU info
    specs["cpu_count"] = psutil.cpu_count(logical=True) # total logical CPUs on the machine.
    specs['cpu_count_available'] = os.cpu_count()       # CPUs available to the current process.
    specs["cpu_freq"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    specs["memory_GB"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    # GPU info
    if torch.cuda.is_available():
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
                "name": name,
                "total_mem_GB": round(mem.total / (1024 ** 3), 2)
            })
        pynvml.nvmlShutdown()
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
    with open(os.path.join(profile_dir, "computing_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

def start_usage_logging(usage_log, stop_event, interval=0.5, gpu_index=0):
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    while not stop_event.is_set():
        timestamp = time.time()
        cpu_percent = psutil.cpu_percent(interval=None)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6  # MB
        usage_log.append((timestamp, cpu_percent, gpu_util, gpu_mem))
        time.sleep(interval)
    pynvml.nvmlShutdown()
    
def save_batch_stats(all_batch_stats, profile_dir):
    os.makedirs(profile_dir, exist_ok=True)
    
    stats_df = pd.DataFrame(all_batch_stats)
    stats_df.to_csv(
        os.path.join(profile_dir, "profile_log.csv"), index=False
    )
    return stats_df

def save_usage_log(usage_log, profile_dir):
    os.makedirs(profile_dir, exist_ok=True)
    
    usage_df = pd.DataFrame(usage_log, columns=["timestamp", "cpu_percent", "gpu_util", "gpu_mem_MB"])
    usage_df["timestamp"] -= usage_df["timestamp"].iloc[0]
    
    usage_df.to_csv(
        os.path.join(profile_dir, "usage_log.csv"), index=False
    )
    return usage_df

def save_usage_plots(usage_df, profile_dir):
    os.makedirs(profile_dir, exist_ok=True)
    
    # Plot CPU & GPU Utilization vs. Time
    plt.figure(figsize=(12, 5))
    plt.plot(usage_df["timestamp"], usage_df["cpu_percent"], label="CPU Util (%)")
    plt.plot(usage_df["timestamp"], usage_df["gpu_util"], label="GPU Util (%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 100) 
    plt.legend()
    plt.title("CPU and GPU Utilization Over Time")
    plt.tight_layout()
    plt.savefig(
        os.path.join(profile_dir, "cpu_gpu_usage_plot.png")
    )
    plt.show()
    

    # Plot GPU memory usage
    plt.figure(figsize=(12, 3))
    plt.plot(usage_df["timestamp"], usage_df["gpu_mem_MB"], label="GPU Memory (MB)")
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Memory (MB)")
    plt.legend()
    plt.title("GPU Memory Usage Over Time")
    plt.tight_layout()
    plt.savefig(
        os.path.join(profile_dir, "gpu_mem_usage_plot.png")
    )
    plt.show()

def save_batch_timings_plot(stats_df, profile_dir):
    os.makedirs(profile_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Linear scale subplot
    axes[0].plot(stats_df['total_batch_s'], label="Total batch time")
    axes[0].plot(stats_df['gpu_inference_s'], label="GPU inference time")
    axes[0].plot(stats_df['cpu_to_gpu_s'], label="CPU→GPU transfer")
    axes[0].plot(stats_df['gpu_to_cpu_s'], label="GPU→CPU transfer")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Batch Timing Breakdown (Linear Scale)")
    axes[0].legend()
    
    # Log scale subplot
    axes[1].plot(stats_df['total_batch_s'], label="Total batch time")
    axes[1].plot(stats_df['gpu_inference_s'], label="GPU inference time")
    axes[1].plot(stats_df['cpu_to_gpu_s'], label="CPU→GPU transfer")
    axes[1].plot(stats_df['gpu_to_cpu_s'], label="GPU→CPU transfer")
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Time (s, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Batch Timing Breakdown (Log Scale)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(profile_dir, "batch_timing_plot.png")
    )
    plt.show()