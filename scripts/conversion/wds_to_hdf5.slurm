#!/bin/bash
#SBATCH --job-name=wds_to_hdf5
#SBATCH --nodes=1
#SBATCH --ntasks=16                            # Number of parallel workers
#SBATCH --cpus-per-task=1
#SBATCH --partition=PARTITION_NAME
#SBATCH --time=12:00:00
#SBATCH --output=logs/wds_to_hdf5_%j.out
#SBATCH --error=logs/wds_to_hdf5_%j.err
#SBATCH --account=ACCOUNT_NAME
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@domain.edu

# === Environment Setup ===
module load python/3.10  # Adjust as needed
source /path/to/your/venv/bin/activate

# === Configuration ===
WDS_DIR="/path/to/wds/shards"
OUTPUT_DIR="/path/to/output"
MAX_ITEMS=50000  # Items per HDF5 file (omit for no limit)
WEBP_QUALITY=100
SHARD_PATTERN="*.tar"

# === Create logs directory ===
mkdir -p logs

# === Run Conversion ===
srun python src/hpc_inference/utils/wds_to_hdf5.py \
    --wds-dir "${WDS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --shard-pattern "${SHARD_PATTERN}" \
    --max-items ${MAX_ITEMS} \
    --webp-quality ${WEBP_QUALITY} \
    --disable-progress \
    --verbose

echo "Conversion completed. Output in: ${OUTPUT_DIR}"
echo "  - Data: ${OUTPUT_DIR}/data/"
echo "  - Metadata: ${OUTPUT_DIR}/metadata/"


