#!/bin/bash
#SBATCH --job-name=embed_job_name             # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                     # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE      # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                     # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK         # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME            # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                       # [REQUIRED] Walltime limit (e.g., 6:00:00)
#SBATCH --output=logs/embed_%j.out            # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/embed_%j.err             # [OPTIONAL] Stderr log file
#SBATCH --account=ACCOUNT_NAME                # [REQUIRED] Project account for allocation
#SBATCH --mail-type=ALL                       # [OPTIONAL] Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@domain.edu     # [OPTIONAL] Email address for notifications

# === Load modules and activate environment ===
module load cuda/VERSION                      # [REQUIRED] Load CUDA module (e.g., cuda/12.4.1)
source /path/to/your/venv/bin/activate        # [REQUIRED] Activate your Python virtual environment

# === Ensure package is installed ===
# Make sure hpc-inference package is installed in your environment

which python                                  # Print Python path for debugging

# === Set config and data paths ===
CONFIG_FILE="/path/to/your/config.yaml"       # [REQUIRED] Path to YAML config file
TARGET_DIR="/path/to/your/input_data"         # [REQUIRED] Directory containing input Parquet files
OUTPUT_DIR="/path/to/your/output_dir"         # [REQUIRED] Directory to save output embeddings
FILE_LIST="/path/to/your/file_list"           # [OPTIONAL] File with list of Parquet files to process (overrides TARGET_DIR)

# === Launch the embedding script ===
# The -m flag runs the Python module as a script. Arguments are:
#   1. CONFIG_FILE: Path to YAML config file
#   2. TARGET_DIR:  Directory with input data
#   3. OUTPUT_DIR:  Directory for output embeddings
#   4. (optional) --file_list: File with list of Parquet files to process (overrides TARGET_DIR)
#
# Alternative: If you have installed the package with entry points, you can use:
# srun hpc-embed "${CONFIG_FILE}" "${TARGET_DIR}" "${OUTPUT_DIR}" --file-list "${FILE_LIST}"
srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${CONFIG_FILE}" \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --file_list "${FILE_LIST}"  # Optional: specify if you have a file list

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
# CONFIG_FILE:         Path to your YAML configuration file (see configs/config_embed_template.yaml).
# TARGET_DIR:          Directory containing input Parquet files.
# OUTPUT_DIR:          Directory where output embeddings will be written.
# FILE_LIST:           [Optional] Path to a file listing specific Parquet files to process.
#                      If provided, only files in this list will be processed; otherwise, all Parquet files in TARGET_DIR are used.
#
# PACKAGE SETUP:       Ensure the hpc-inference package is installed:
#                      cd /path/to/sci_hpc_batch_inference && pip install -e .
#                      This installs the package in development mode with all dependencies.