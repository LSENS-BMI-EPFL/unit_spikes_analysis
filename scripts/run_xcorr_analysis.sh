#!/bin/bash
#SBATCH --job-name=xcorr_analysis       # Job name
#SBATCH --output=/home/%u/logs/xcorr_analysis_%j.out  # Output log file (%j = job ID)
#SBATCH --error=/home/%u/logs/xcorr_analysis_%j.err   # Error log file
#SBATCH --time=7-24:00:00                 # Max runtime (adjust based on expected runtime)
#SBATCH --qos=parallel                    # QOS
#SBATCH --nodes=32                       # Single node
#SBATCH --ntasks=74                      # Single task
#SBATCH --cpus-per-task=1               # Use 8 CPU cores (adjust as needed)
#SBATCH --mem=32G                       # Allocate 16GB memory (adjust as needed)
#SBATCH --mail-type=END,FAIL            # Email notifications for job completion or failure
#SBATCH --mail-user=axel.bisi@epfl.ch  # Replace with your email address

# Load modules (if necessary for your cluster)
#module load python3.8 neo  # Replace with your Python version
module load gcc python openmpi py-mpi4py

# Activate virtual environment
source /home/bisi/venvs/xcorr/bin/activate

# Input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <path_to_nwb_file> <path_to_results>"
    exit 1
fi

NWB_FILE="$1"
RESULTS_PATH="$2"

# Ensure the results directory exists
mkdir -p "$RESULTS_PATH"

# Start the Python script
echo "Starting xcorr_analysis for NWB file: $NWB_FILE"
start_time=$(date +%s)

srun -N 32 -n 74 -q parallel python3 xcorr_utils.py "$NWB_FILE" "$RESULTS_PATH"

#python3 -c "
#import sys
#from xcorr_utils import xcorr_analysis_mpi

#xcorr_analysis_mpi('$NWB_FILE', '$RESULTS_PATH')
#" || {
#    echo "Error: xcorr_analysis failed!"
#    exit 1
#}

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "xcorr_analysis completed successfully in ${elapsed}s. Results saved in: $RESULTS_PATH"


