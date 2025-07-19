#!/bin/bash
#SBATCH --job-name=animal_detect_images_job    # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 8:00:00)
#SBATCH --output=logs/animal_detect_images_%j.out  # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/animal_detect_images_%j.err   # [OPTIONAL] Stderr log file
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
TARGET_DIR="/path/to/your/image_directory"     # [REQUIRED] Directory containing input images
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save animal detection results

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/animal_detect_config.yaml"  # Path to YAML config file

srun python -m hpc_inference.inference.detection.animal_detect \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type images \
    --config "${CONFIG_FILE}"

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.detection.animal_detect \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type images \
#     --model_weights "md_v5a.0.0.pt" \
#     --confidence_threshold 0.2 \
#     --image_size 1280 \
#     --batch_size 8 \
#     --num_workers 24 \
#     --prefetch_factor 8 \
#     --max_rows_per_file 5000 \
#     --out_prefix "animal_detection_results" \
#     --uuid_mode filename \
#     --evenly_distribute \
#     --validate_images


# -------------------------------
# ANIMAL DETECTION IMAGE FOLDER-SPECIFIC PARAMETERS
# -------------------------------
# --input_type images:     Tells the script to process image files from a directory
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
# --uuid_mode:             How to generate unique IDs from image paths:
#                          - "filename": Use just the filename (image001.jpg)
#                          - "relative": Use relative path from TARGET_DIR (subfolder/image001.jpg)
#                          - "fullpath": Use full absolute path (/full/path/to/image001.jpg)
#                          - "hash": Use MD5 hash of the full path (a1b2c3d4e5f6g7h8)
# --validate_images:       [OPTIONAL] Validate that all images can be opened with PIL
#                          Slower but safer - catches corrupted files before processing
# --file_list:             NOT applicable for image folders (will cause error)

# SUPPORTED IMAGE FORMATS:
# .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
# Images are automatically converted to RGB mode for animal detection

# DIRECTORY STRUCTURE:
# TARGET_DIR can contain:
# - Flat structure: /images/img1.jpg, /images/img2.jpg, ...
# - Nested structure: /images/site1/img1.jpg, /images/site2/img2.jpg, ...
# All .jpg, .jpeg, .png, etc. files will be found recursively

# OUTPUT FORMAT:
# The script outputs Parquet files containing:
# - uuid: Unique identifier for each image (based on uuid_mode)
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

# PERFORMANCE TIPS FOR ANIMAL DETECTION ON IMAGE FOLDERS:
# - Animal detection (especially MegaDetector) is more memory-intensive than face detection
# - Start with smaller batch sizes (8 instead of 16) and increase if memory allows
# - Use --evenly_distribute for better load balancing when file sizes vary
# - Use --validate_images if you suspect corrupted files (adds startup time)
# - Consider --uuid_mode based on your downstream analysis needs
# - MegaDetector confidence_threshold of 0.2 is typically optimal for wildlife images
# - Choose model based on use case:
#   * MegaDetector: Best for wildlife/camera trap images
#   * YOLOv8: General purpose, good for various animal contexts
# - For large datasets, consider converting to Parquet format first for better I/O performance

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
