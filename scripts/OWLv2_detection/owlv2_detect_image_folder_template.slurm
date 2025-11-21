#!/bin/bash
#SBATCH --job-name=owlv2_detect_images_job     # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 8:00:00)
#SBATCH --output=logs/owlv2_detect_images_%j.out  # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/owlv2_detect_images_%j.err   # [OPTIONAL] Stderr log file
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
TARGET_DIR="/path/to/your/image_directory"     # [REQUIRED] Directory containing input images
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save detection results

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/owlv2_detect_config.yaml"  # Path to YAML config file

srun python -m hpc_inference.inference.detection.OWLv2_detect \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type images \
    --config "${CONFIG_FILE}"

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.detection.OWLv2_detect \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type images \
#     --text_labels fish shark whale \
#     --confidence_threshold 0.1 \
#     --batch_size 16 \
#     --num_workers 20 \
#     --prefetch_factor 16 \
#     --max_uuids_per_file 10000 \
#     --out_prefix "OWLv2_detection_results" \
#     --uuid_mode filename \
#     --evenly_distribute


# -------------------------------
# OWLv2 DETECTION IMAGE FOLDER-SPECIFIC PARAMETERS
# -------------------------------
# --input_type images:     Tells the script to process image files from a directory
# --text_labels:           List of object names to detect (zero-shot detection)
#                          Examples: fish, shark, whale, dolphin, turtle, coral
#                          Can specify any object names in natural language
# --confidence_threshold:  Minimum confidence score for detections (0.0-1.0)
#                          OWLv2 typically uses 0.1 as default threshold
#                          Lower = more detections (including false positives)
#                          Higher = fewer, more confident detections
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
# Images are automatically converted to RGB mode for detection

# DIRECTORY STRUCTURE:
# TARGET_DIR can contain:
# - Flat structure: /images/img1.jpg, /images/img2.jpg, ...
# - Nested structure: /images/site1/img1.jpg, /images/site2/img2.jpg, ...
# All .jpg, .jpeg, .png, etc. files will be found recursively

# OUTPUT FORMAT:
# The script outputs Parquet files with schema:
# - image_uuid: Unique identifier for each image (based on uuid_mode)
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

# PERFORMANCE TIPS FOR OWLv2 DETECTION ON IMAGE FOLDERS:
# - OWLv2 is memory-intensive due to vision-language model architecture
# - Model uses google/owlv2-base-patch16-ensemble (approx. 500MB)
# - Batch size of 16 works well for most GPUs with 16GB+ memory
# - Reduce batch size if encountering OOM errors
# - Use --evenly_distribute for better load balancing when file sizes vary
# - Consider --uuid_mode based on your downstream analysis needs

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
# - Consider pre-downloading before job execution:
#   python -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; \
#              Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); \
#              Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')"
