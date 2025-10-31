#!/bin/bash
#SBATCH --job-name=classify_parquet_job        # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 6:00:00)
#SBATCH --output=logs/classify_parquet_%j.out  # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/classify_parquet_%j.err   # [OPTIONAL] Stderr log file
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
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save classification results
CLASS_EMBEDDINGS_DICT="/path/to/class_embeddings.pt"  # [REQUIRED] Path to class embeddings dictionary file

# === Optional parameters ===
FILE_LIST="/path/to/your/file_list.txt"        # [OPTIONAL] File with list of Parquet files to process

# === Run classification inference ===
srun python -m hpc_inference.inference.classify.classify \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    "${CLASS_EMBEDDINGS_DICT}" \
    --file_list "${FILE_LIST}" \
    --batch_size 8 \
    --num_workers 24 \
    --prefetch_factor 4 \
    --read_batch_size 640 \
    --max_rows_per_file 10000 \
    --out_prefix "classify_results" \
    --profile_dir "profile_results"


# -------------------------------
# CLASSIFICATION PARAMETERS
# -------------------------------
# TARGET_DIR:              Directory containing Parquet files with embeddings to classify
# OUTPUT_DIR:              Directory to save classification results
# CLASS_EMBEDDINGS_DICT:   Path to PyTorch (.pt) file containing class embeddings dictionary
#                          Format: {"class_name": embedding_tensor, ...}
# --file_list:             [Optional] Path to text file with list of specific Parquet files to process
#                          Format: one file path per line
#                          If not provided, all .parquet files in TARGET_DIR will be processed
# --batch_size:            Number of embeddings to process per batch (default: 8)
# --num_workers:           Number of workers for DataLoader (default: 24)
# --prefetch_factor:       Prefetch factor for DataLoader (default: 4)
# --read_batch_size:       Number of rows to read from each Parquet file at a time (default: 640)
# --max_rows_per_file:     Maximum rows per output file before creating a new file (default: 10000)
# --out_prefix:            Prefix for output files (default: "clip_embed_results")
# --profile_dir:           Directory to save profiling results (default: "profile_results")

# PARQUET FILE REQUIREMENTS:
# - Must contain 'uuid' column with unique identifiers
# - Must contain 'embedding' column with embedding vectors (as tensors or arrays)
# - Embeddings should be compatible with the class embeddings for similarity computation

# OUTPUT FORMAT:
# - Parquet files containing: uuid, pred_label, pred_score
# - Files are split when max_rows_per_file is reached
# - Named as: {out_prefix}_rank_{rank}_{file_idx}.parquet

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

# PERFORMANCE TIPS FOR CLASSIFICATION:
# - Adjust batch_size based on GPU memory and embedding dimensions
# - Use appropriate num_workers based on CPU cores and I/O characteristics
# - Adjust read_batch_size based on available memory (640 works well for most cases)
# - Consider using file_list to process specific files or resume interrupted jobs
# - Monitor GPU utilization and adjust batch_size accordingly
# - Class embeddings are loaded once and kept on GPU for efficiency