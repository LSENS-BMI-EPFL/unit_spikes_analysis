#!/bin/bash
#SBATCH --job-name=single_unit_analysis
#SBATCH --output=/home/%u/logs/single_units_analysis_%A_%a.out
#SBATCH --error=/home/%u/logs/single_units_analysis_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myriam.hamon@epfl.ch

source /home/mhamon/anaconda3/etc/profile.d/conda.sh
conda activate unit_spike_env_2

FILE_LIST="$1"
RESULTS_PATH="$2"

NWB_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST")

echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing file: $NWB_FILE"
echo "Results path: $RESULTS_PATH"

if [ ! -f "$NWB_FILE" ]; then
    echo "Error: File $NWB_FILE does not exist"
    exit 1
fi

mkdir -p "$RESULTS_PATH"

echo "Starting analysis for NWB file: $NWB_FILE"
start_time=$(date +%s)


# Use full path to the Python script
python3 /home/mhamon/Github/unit_spikes_analysis/glm_utils.py --nwb "$NWB_FILE" --out "$RESULTS_PATH"


end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Analysis completed successfully in ${elapsed}s"
