#!/bin/bash
#SBATCH --job-name=embed_parquet_job           # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 6:00:00)
#SBATCH --output=logs/embed_parquet_%j.out     # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/embed_parquet_%j.err      # [OPTIONAL] Stderr log file
#SBATCH --account=ACCOUNT_NAME                 # [REQUIRED] Project account for allocation
#SBATCH --mail-type=ALL                        # [OPTIONAL] Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@domain.edu      # [OPTIONAL] Email address for notifications

# === Load modules and activate environment ===
module load cuda/VERSION                       # [REQUIRED] Load CUDA module (e.g., cuda/12.4.1)
source /path/to/your/venv/bin/activate         # [REQUIRED] Activate your Python virtual environment

# === Ensure package is installed ===
# Make sure hpc-inference package is installed in your environment
which python                                   # Print Python path for debugging

# === Set data paths ===
TARGET_DIR="/path/to/your/parquet_files"       # [REQUIRED] Directory containing input Parquet files
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save output embeddings

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/parquet_config.yaml"  # Path to YAML config file
FILE_LIST="/path/to/your/file_list.txt"        # [OPTIONAL] File with list of Parquet files

srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type parquet \
    --config "${CONFIG_FILE}" \
    --file_list "${FILE_LIST}"  # [OPTIONAL] Use this if you have a specific list of files to process

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.embed.open_clip_embed \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type parquet \
#     --model_name "ViT-B-32" \
#     --pretrained "openai" \
#     --batch_size 32 \
#     --num_workers 28 \
#     --prefetch_factor 32 \
#     --read_batch_size 128 \
#     --max_rows_per_file 10000 \
#     --out_prefix "embed_results" \
#     --read_columns uuid original_size resized_size image \
#     --evenly_distribute \
#     --stagger


# -------------------------------
# PARQUET-SPECIFIC PARAMETERS
# -------------------------------
# --input_type parquet:    Tells the script to process Parquet files containing encoded images
# --file_list:             [Optional] Path to text file with list of specific Parquet files to process
#                          Format: one file path per line
#                          If not provided, all .parquet files in TARGET_DIR will be processed
# --read_batch_size:       Number of rows to read from each Parquet file at a time
#                          Larger values = more memory usage but potentially faster I/O
# --read_columns:          Specific columns to read from Parquet files
#                          Default: uuid original_size resized_size image
#                          Reduce columns to save memory if you don't need all data

# PARQUET FILE REQUIREMENTS:
# - Must contain 'uuid' column with unique identifiers
# - Must contain 'image' column with encoded image bytes (JPEG, PNG, etc.)
# - Optional: 'original_size', 'resized_size' columns for metadata
# - Images should be encoded as bytes (e.g., from PIL.Image.save to BytesIO)

# -------------------------------
# SLURM Template Field Explanations
# -------------------------------
# --job-name:          Name for your job in the queue/monitoring system.
# --nodes:             Number of nodes to allocate for the job.
# --gpus-per-task:     Number of GPUs per task (set to 1 unless using model parallelism).
# --cpus-per-task:     Number of CPU cores per task (should match or exceed your data loader workers).
# --ntasks-per-node:   Number of parallel tasks per node.
#                      For Parquet processing, consider I/O bottlenecks and set conservatively.
# --partition:         Cluster partition/queue to submit to (e.g., gpu, gpu-exp).
# --time:              Maximum walltime for the job (format: HH:MM:SS).
# --output:            Path for standard output log file (use %j for job ID).
# --error:             Path for standard error log file.
# --account:           Your allocation/project account for resource usage.

# PERFORMANCE TIPS FOR PARQUET:
# - Use --evenly_distribute for better load balancing across ranks
# - Adjust --read_batch_size based on available memory (128-1000 works well)
# - Use --stagger to avoid I/O congestion when multiple workers start
# - Consider using --file_list to process specific files or resume jobs