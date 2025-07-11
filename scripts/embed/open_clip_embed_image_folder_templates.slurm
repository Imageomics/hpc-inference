#!/bin/bash
#SBATCH --job-name=embed_images_job            # [REQUIRED] Set a descriptive job name
#SBATCH --nodes=NUM_NODES                      # [REQUIRED] Number of nodes to use
#SBATCH --ntasks-per-node=TASKS_PER_NODE       # [RECOMMENDED] Number of tasks per node
#SBATCH --gpus-per-task=1                      # [REQUIRED] Number of GPUs per task (set to 1)
#SBATCH --cpus-per-task=CPUS_PER_TASK          # [RECOMMENDED] Number of CPU cores per task (e.g., 48)
#SBATCH --partition=PARTITION_NAME             # [REQUIRED] Partition/queue name (e.g., gpu, gpu-exp)
#SBATCH --time=HH:MM:SS                        # [REQUIRED] Walltime limit (e.g., 6:00:00)
#SBATCH --output=logs/embed_images_%j.out      # [OPTIONAL] Stdout log file (%j = job ID)
#SBATCH --error=logs/embed_images_%j.err       # [OPTIONAL] Stderr log file
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
TARGET_DIR="/path/to/your/image_directory"     # [REQUIRED] Directory containing input images
OUTPUT_DIR="/path/to/your/output_dir"          # [REQUIRED] Directory to save output embeddings

# === Choose your configuration method ===
# Option 1: Use config file (RECOMMENDED for production)
CONFIG_FILE="/path/to/your/images_config.yaml" # Path to YAML config file

srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${TARGET_DIR}" \
    "${OUTPUT_DIR}" \
    --input_type images \
    --config "${CONFIG_FILE}"

# Option 2: Use command line arguments (for quick testing)
# Uncomment and modify the lines below, comment out the config version above
#
# srun python -m hpc_inference.inference.embed.open_clip_embed \
#     "${TARGET_DIR}" \
#     "${OUTPUT_DIR}" \
#     --input_type images \
#     --model_name "ViT-B-32" \
#     --pretrained "openai" \
#     --batch_size 32 \
#     --num_workers 28 \
#     --prefetch_factor 32 \
#     --max_rows_per_file 10000 \
#     --out_prefix "embed_results" \
#     --uuid_mode filename \
#     --evenly_distribute \
#     --stagger \
#     --validate_images


# -------------------------------
# IMAGE FOLDER-SPECIFIC PARAMETERS
# -------------------------------
# --input_type images:     Tells the script to process image files from a directory
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
# Images are automatically converted to RGB mode for model processing

# DIRECTORY STRUCTURE:
# TARGET_DIR can contain:
# - Flat structure: /images/img1.jpg, /images/img2.jpg, ...
# - Nested structure: /images/class1/img1.jpg, /images/class2/img2.jpg, ...
# All .jpg, .jpeg, .png, etc. files will be found recursively

# -------------------------------
# SLURM Template Field Explanations
# -------------------------------
# --job-name:          Name for your job in the queue/monitoring system.
# --nodes:             Number of nodes to allocate for the job.
# --gpus-per-task:     Number of GPUs per task (set to 1 unless using model parallelism).
# --cpus-per-task:     Number of CPU cores per task (should match or exceed your data loader workers).
# --ntasks-per-node:   Number of parallel tasks per node.
#                      For image processing, balance between available GPUs and I/O capacity.
# --partition:         Cluster partition/queue to submit to (e.g., gpu, gpu-exp).
# --time:              Maximum walltime for the job (format: HH:MM:SS).
# --output:            Path for standard output log file (use %j for job ID).
# --error:             Path for standard error log file.
# --account:           Your allocation/project account for resource usage.

# PERFORMANCE TIPS FOR IMAGE FOLDERS:
# - Use --evenly_distribute for better load balancing when file sizes vary
# - Use --validate_images if you suspect corrupted files (adds startup time)
# - Consider --uuid_mode based on your downstream analysis needs
# - Use --stagger to reduce file system stress during startup
# - For large datasets, consider converting to Parquet format first for better performance