#!/bin/bash
#SBATCH --job-name=animal_detect_parquet_job   # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 8:00:00)
#SBATCH --output=logs/animal_detect_parquet_%j.out  # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/animal_detect_parquet_%j.err   # [OPTIONAL] Stderr log file
#SBATCH --account=ACCOUNT_NAME                 # [REQUIRED] Project account for allocation
#SBATCH --mail-type=ALL                        # [OPTIONAL] Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@domain.edu      # [OPTIONAL] Email address for notifications

# === Load modules and activate environment ===
module load cuda/VERSION                       # [REQUIRED] Load CUDA module (e.g., cuda/12.4.1)
source /path/to/your/venv/bin/activate         # [REQUIRED] Activate your Python virtual environment

# === Ensure package is installed ===
# Make sure hpc-inference package with YOLO dependencies is installed
# pip install 'hpc-inference[yolo]' or pip install ultralytics
which python                                   # Print Python path for debugging

# === Set data paths ===
TARGET_DIR="/path/to/your/parquet_files"       # [REQUIRED] Directory containing input Parquet files
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save animal detection results

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/animal_detect_parquet_config.yaml"  # Path to YAML config file
FILE_LIST="/path/to/your/file_list.txt"        # [OPTIONAL] File with list of Parquet files

srun python -m hpc_inference.inference.detection.animal_detect \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type parquet \
    --config "${CONFIG_FILE}" \
    --file_list "${FILE_LIST}"  # [OPTIONAL] Use this if you have a specific list of files to process

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.detection.animal_detect \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type parquet \
#     --model_weights "md_v5a.0.0.pt" \
#     --confidence_threshold 0.2 \
#     --image_size 1280 \
#     --batch_size 8 \
#     --num_workers 24 \
#     --prefetch_factor 8 \
#     --read_batch_size 64 \
#     --max_rows_per_file 5000 \
#     --out_prefix "animal_detection_results" \
#     --read_columns uuid image original_size resized_size \
#     --evenly_distribute


# -------------------------------
# ANIMAL DETECTION PARQUET-SPECIFIC PARAMETERS
# -------------------------------
# --input_type parquet:    Tells the script to process Parquet files containing encoded images
# --model_weights:         Animal detection model file:
#                          - "md_v5a.0.0.pt" (MegaDetector v5a, recommended for wildlife)
#                          - "md_v5b.0.0.pt" (MegaDetector v5b, alternative)
#                          - "yolov8n.pt" (YOLOv8 nano, fastest)
#                          - "yolov8s.pt" (YOLOv8 small, balanced)
#                          - "yolov8m.pt" (YOLOv8 medium, more accurate)
#                          - "yolov8l.pt" (YOLOv8 large, most accurate)
# --confidence_threshold:  Minimum confidence score for animal detections (0.0-1.0)
#                          MegaDetector typically uses 0.2 as optimal threshold
#                          Lower = more detections (including false positives)
#                          Higher = fewer, more confident detections
# --image_size:            Input image size for the model (square format)
#                          MegaDetector v5 typically uses 1280
#                          Larger sizes improve accuracy but slow processing
# --read_batch_size:       Number of rows to read from Parquet files at once
#                          Smaller than face detection due to larger output per image
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
# The script outputs Parquet files containing:
# - uuid: Unique identifier for each image (from input Parquet)
# - max_detection_score: Maximum confidence score across all detections (0.0 if no animals detected)
# - num_detections: Total number of detections above threshold
# - detections: JSON string with detailed detection information including:
#   * bbox: Absolute pixel coordinates [x1, y1, x2, y2]
#   * bbox_normalized: Normalized coordinates [0-1]
#   * confidence: Detection confidence score
#   * class_id: Numeric class ID (0=animal, 1=person, 2=vehicle for MegaDetector)
#   * class_name: Human-readable class name
# Files are saved in: {OUTPUT_DIR}/detections/rank_{rank}/

# -------------------------------
# SLURM Template Field Explanations
# -------------------------------
# --job-name:          Name for your job in the queue/monitoring system.
# --nodes:             Number of nodes to allocate for the job.
# --gpus-per-task:     Number of GPUs per task (set to 1 unless using model parallelism).
# --cpus-per-task:     Number of CPU cores per task (should match or exceed your data loader workers).
# --ntasks-per-node:   Number of parallel tasks per node.
#                      For animal detection, balance between available GPUs and I/O capacity.
# --partition:         Cluster partition/queue to submit to (e.g., gpu, gpu-exp).
# --time:              Maximum walltime for the job (format: HH:MM:SS).
#                      Animal detection can be slower than face detection, allow more time.
# --output:            Path for standard output log file (use %j for job ID).
# --error:             Path for standard error log file.
# --account:           Your allocation/project account for resource usage.

# PERFORMANCE TIPS FOR ANIMAL DETECTION ON PARQUET:
# - Animal detection (especially MegaDetector) is more memory-intensive than face detection
# - Start with smaller batch sizes (8 instead of 16) and increase if memory allows
# - Use --evenly_distribute for better load balancing across workers
# - Adjust --read_batch_size for optimal I/O performance (64 is conservative)
# - Use --file_list to process specific subsets of your data efficiently
# - MegaDetector confidence_threshold of 0.2 is typically optimal for wildlife images
# - Choose model based on use case:
#   * MegaDetector: Best for wildlife/camera trap images
#   * YOLOv8: General purpose, good for various animal contexts
# - Consider --stagger if experiencing I/O bottlenecks during startup

# MEGADETECTOR MODEL NOTES:
# - MegaDetector is specifically designed for wildlife camera trap images
# - Trained on millions of camera trap images from around the world
# - Detects animals, people, and vehicles with high accuracy
# - Works best on outdoor/natural settings
# - May not perform as well on indoor pets or zoo animals
# - Models will be automatically downloaded on first use
# - Ensure internet connectivity or pre-download models to avoid delays

# MODEL DOWNLOAD AND CACHING:
# - Models will be automatically downloaded to ~/.cache/ultralytics/ on first use
# - MegaDetector models are larger than standard YOLO models (~400MB vs ~50MB)
# - Consider pre-downloading models before job execution:
#   python -c "from ultralytics import YOLO; YOLO('md_v5a.0.0.pt')"

# FILE_LIST FORMAT (if using --file_list):
# Create a text file with one Parquet file path per line:
# /path/to/data/file1.parquet
# /path/to/data/file2.parquet
# /path/to/data/file3.parquet

# EXAMPLE DETECTION OUTPUT:
# For each image, the detections JSON will contain:
# [
#   {
#     "bbox": [120.5, 80.2, 340.8, 290.1],
#     "bbox_normalized": [0.118, 0.078, 0.333, 0.284],
#     "confidence": 0.85,
#     "class_id": 0,
#     "class_name": "animal"
#   },
#   {
#     "bbox": [450.0, 200.0, 600.0, 400.0],
#     "bbox_normalized": [0.440, 0.195, 0.586, 0.391],
#     "confidence": 0.72,
#     "class_id": 1,
#     "class_name": "person"
#   }
# ]

# CAMERA TRAP WORKFLOW CONSIDERATIONS:
# - MegaDetector is optimized for camera trap scenarios
# - Handles various lighting conditions and weather
# - Good at distinguishing animals from vegetation movement
# - Reduces false positives from shadows, leaves, etc.
# - Provides both detection and rough classification
# - Ideal for wildlife monitoring and ecological research
