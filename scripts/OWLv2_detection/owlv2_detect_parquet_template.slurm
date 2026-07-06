#!/bin/bash
#SBATCH --job-name=owlv2_detect_parquet_job    # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 8:00:00)
#SBATCH --output=logs/owlv2_detect_parquet_%j.out  # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/owlv2_detect_parquet_%j.err   # [OPTIONAL] Stderr log file
#SBATCH --account=ACCOUNT_NAME                 # [REQUIRED] Project account for allocation
#SBATCH --mail-type=ALL                        # [OPTIONAL] Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@domain.edu      # [OPTIONAL] Email address for notifications

# === Load modules and activate environment ===
module load cuda/VERSION                       # [REQUIRED] Load CUDA module (e.g., cuda/12.4.1)
source /path/to/your/venv/bin/activate         # [REQUIRED] Activate your Python virtual environment

# === Ensure package is installed ===
# Make sure hpc-inference package with transformers dependencies is installed
# pip install 'hpc-inference[transformers]' or pip install transformers
which python                                   # Print Python path for debugging

# === Set data paths ===
TARGET_DIR="/path/to/your/parquet_files"       # [REQUIRED] Directory containing input Parquet files
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save detection results

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/owlv2_detect_parquet_config.yaml"  # Path to YAML config file
FILE_LIST="/path/to/your/file_list.txt"        # [OPTIONAL] File with list of Parquet files

srun python -m hpc_inference.inference.detection.OWLv2_detect \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type parquet \
    --config "${CONFIG_FILE}" \
    --file_list "${FILE_LIST}"  # [OPTIONAL] Use this if you have a specific list of files to process

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.detection.OWLv2_detect \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type parquet \
#     --text_labels fish shark whale \
#     --confidence_threshold 0.1 \
#     --batch_size 16 \
#     --num_workers 20 \
#     --prefetch_factor 16 \
#     --read_batch_size 128 \
#     --max_uuids_per_file 10000 \
#     --out_prefix "OWLv2_detection_results" \
#     --read_columns uuid image original_size resized_size \
#     --evenly_distribute


# -------------------------------
# OWLv2 DETECTION PARQUET-SPECIFIC PARAMETERS
# -------------------------------
# --input_type parquet:    Tells the script to process Parquet files containing encoded images
# --text_labels:           List of object names to detect (zero-shot detection)
#                          Examples: fish, shark, whale, dolphin, turtle, coral
#                          Can specify any object names in natural language
# --confidence_threshold:  Minimum confidence score for detections (0.0-1.0)
#                          OWLv2 typically uses 0.1 as default threshold
#                          Lower = more detections (including false positives)
#                          Higher = fewer, more confident detections
# --read_batch_size:       Number of rows to read from Parquet files at once
# --read_columns:          Columns to read from Parquet files (space-separated)
#                          REQUIRED: uuid, image
#                          OPTIONAL: original_size, resized_size, etc.
# --file_list:             [OPTIONAL] Path to text file containing list of Parquet files to process
#                          Useful for processing specific subsets of data
# --evenly_distribute:     Distribute files evenly based on size for better load balancing
# --stagger:               [OPTIONAL] Stagger worker start times to reduce I/O congestion

# PARQUET DATA REQUIREMENTS:
# Your Parquet files must contain:
# 1. 'uuid' column: Unique string identifier for each image
# 2. 'image' column: Image data encoded as bytes (from PIL Image.save() to BytesIO)
# 3. Optional metadata columns as specified in --read_columns

# OUTPUT FORMAT:
# The script outputs Parquet files with schema:
# - image_uuid: Unique identifier for each image (from input Parquet)
# - detection_id: Detection index within the image (null if no detections)
# - score: Detection confidence score (null if no detections)
# - text_label: Detected object label from text_labels list (null if no detections)
# - box: Bounding box coordinates [x1, y1, x2, y2] in pixels (null if no detections)
# Files are saved in: {OUTPUT_DIR}/detections/rank_{rank}/

# -------------------------------
# SLURM Template Field Explanations
# -------------------------------
# --job-name:          Name for your job in the queue/monitoring system.
# --nodes:             Number of nodes to allocate for the job.
# --gpus-per-task:     Number of GPUs per task (set to 1).
# --cpus-per-task:     Number of CPU cores per task (should match or exceed your data loader workers).
# --ntasks-per-node:   Number of parallel tasks per node.
# --partition:         Cluster partition/queue to submit to (e.g., gpu, gpu-exp).
# --time:              Maximum walltime for the job (format: HH:MM:SS).
# --output:            Path for standard output log file (use %j for job ID).
# --error:             Path for standard error log file.
# --account:           Your allocation/project account for resource usage.

# PERFORMANCE TIPS FOR OWLv2 DETECTION ON PARQUET:
# - OWLv2 is memory-intensive due to vision-language model architecture
# - Model uses google/owlv2-base-patch16-ensemble (approx. 500MB)
# - Batch size of 16 works well for most GPUs with 16GB+ memory
# - Reduce batch size if encountering OOM errors
# - Use --evenly_distribute for better load balancing across workers
# - Adjust --read_batch_size for optimal I/O performance
# - Use --file_list to process specific subsets of your data efficiently
# - Consider --stagger if experiencing I/O bottlenecks during startup

# OWLv2 MODEL NOTES:
# - Zero-shot object detection using natural language queries
# - No training required - can detect any object described in text
# - Model: google/owlv2-base-patch16-ensemble
# - Automatically downloaded from Hugging Face on first use
# - Requires internet connectivity or pre-cached model
# - Works well on diverse objects and scenes
# - Performance depends on text prompt quality
# - More flexible than trained detectors but may be less accurate for specialized domains

# MODEL DOWNLOAD AND CACHING:
# - Model will be automatically downloaded to ~/.cache/huggingface/ on first use
# - Model size: ~500MB for weights + ~100MB for processor
# - Consider pre-downloading before job execution:
#   python -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; \
#              Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); \
#              Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')"

# FILE_LIST FORMAT (if using --file_list):
# Create a text file with one Parquet file path per line:
# /path/to/data/file1.parquet
# /path/to/data/file2.parquet
# /path/to/data/file3.parquet

# EXAMPLE DETECTION OUTPUT:
# For each image, the output will have one row per detection:
# image_uuid          | detection_id | score | text_label | box
# -------------------|-------------|-------|------------|------------------
# img_001            | 0           | 0.85  | fish       | [120.5, 80.2, 340.8, 290.1]
# img_001            | 1           | 0.72  | shark      | [450.0, 200.0, 600.0, 400.0]
# img_002            | null        | null  | null       | null
#
# Images with no detections will have a single row with null values

# ZERO-SHOT DETECTION TIPS:
# - Use clear, specific object names (e.g., "shark" not "big fish")
# - Can use compound terms (e.g., "blue whale", "sea turtle")
# - Multiple related terms may improve recall (e.g., "fish", "fish swimming")
# - Model works best with common objects from its training distribution
# - May struggle with very specific or rare object categories
# - Consider multiple detection passes with different text prompts for comprehensive coverage
