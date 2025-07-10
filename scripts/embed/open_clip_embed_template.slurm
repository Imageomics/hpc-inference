#!/bin/bash
#SBATCH --job-name=embed_job_name             # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                     # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE      # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                     # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK         # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME            # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                       # [REQUIRED] Walltime limit (e.g., 6:00:00)
#SBATCH --output=logs/embed_job_name%j.out    # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/embed_job_name%j.err     # [OPTIONAL] Stderr log file
#SBATCH --account=ACCOUNT_NAME                # [REQUIRED] Project account for allocation
#SBATCH --mail-type=ALL                       # [OPTIONAL] Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@domain.edu     # [OPTIONAL] Email address for notifications

# === Load modules and activate environment ===
module load cuda/VERSION                      # [REQUIRED] Load CUDA module (e.g., cuda/12.4.1)
source /path/to/your/venv/bin/activate        # [REQUIRED] Activate your Python virtual environment

# === Ensure package is installed ===
# Make sure hpc-inference package is installed in your environment
which python                                  # Print Python path for debugging

# === Set data paths ===
TARGET_DIR="/path/to/your/input_data"         # [REQUIRED] Directory containing input Parquet files
OUTPUT_DIR="/path/to/your/output_dir"         # [REQUIRED] Directory to save output embeddings

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/config.yaml"       # Path to YAML config file
FILE_LIST="/path/to/your/file_list"           # [OPTIONAL] File with list of Parquet files

srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --config "${CONFIG_FILE}" \
    --file_list "${FILE_LIST}" # [OPTIONAL] Use this if you have a specific list of files to process

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.embed.open_clip_embed \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --model_name "ViT-B-32" \
#     --pretrained "openai" \
#     --batch_size 32 \
#     --num_workers 28 \
#     --prefetch_factor 32 \
#     --read_batch_size 128 \
#     --max_rows_per_file 10000 \
#     --out_prefix "embed_results" \
#     --read_columns uuid original_size resized_size image


# -------------------------------
# SLURM Template Field Explanations
# -------------------------------
# --job-name:          Name for your job in the queue/monitoring system.
# --nodes:             Number of nodes to allocate for the job.
# --gpus-per-task:     Number of GPUs per task (set to 1 unless using model parallelism).
# --cpus-per-task:     Number of CPU cores per task (should match or exceed your data loader workers).
# --ntasks-per-node:   Number of parallel tasks per node.
#                      Set this based on how many GPUs and CPUs are available per node.
#                      For example, if one task requires 1 GPU and 48 CPUs, and the node has 2 GPUs and 50 CPUs,
#                      then ntasks-per-node should be 1 (to avoid oversubscription).
#                      If your workload can use less resources per task, you could increase this to 2 or more,
#                      as long as you do not exceed the available GPUs and CPUs per node.
#                      The total number of tasks will be nodes * ntasks-per-node.
# --partition:         Cluster partition/queue to submit to (e.g., gpu, gpu-exp).
# --time:              Maximum walltime for the job (format: HH:MM:SS).
# --output:            Path for standard output log file (use %j for job ID).
# --error:             Path for standard error log file.
# --account:           Your allocation/project account for resource usage.
# --mail-type:         When to send email notifications (BEGIN, END, FAIL, ALL).
# --mail-user:         Email address for notifications.
#
# CONFIGURATION OPTIONS:
# 1. CONFIG FILE (Recommended): Use --config to specify a YAML configuration file
#    See configs/config_embed_template.yaml for template and examples
# 2. COMMAND LINE: Use individual arguments (--model_name, --batch_size, etc.)


# TARGET_DIR:          Directory containing input Parquet files.
# OUTPUT_DIR:          Directory where output embeddings will be written.
# FILE_LIST:           [Optional] Path to a file listing specific Parquet files to process.
#                      If provided, only files in this list will be processed; otherwise, all Parquet files in TARGET_DIR are used.
#
# PACKAGE SETUP:       Ensure the hpc-inference package is installed:
#                      cd /path/to/sci_hpc_batch_inference && pip install -e .
#                      This installs the package in development mode with all dependencies.

# -------------------------------
# Common Command Line Arguments
# -------------------------------
# --model_name:        OpenCLIP model name (e.g., ViT-B-32, ViT-L-14, ViT-bigG-14)
# --pretrained:        Pretrained weights (e.g., openai, laion2b_s32b_b82k)
# --batch_size:        Batch size for inference (adjust based on GPU memory)
# --num_workers:       Number of dataloader workers (should match --cpus-per-task)
# --prefetch_factor:   Number of batches prefetched by each worker
# --read_batch_size:   Number of rows read from Parquet at a time
# --max_rows_per_file: Maximum rows per output Parquet file
# --out_prefix:        Prefix for output files
# --read_columns:      Columns to read from Parquet files (space-separated list)
#                      Default: uuid original_size resized_size image
#                      Example: --read_columns uuid image (for minimal columns)