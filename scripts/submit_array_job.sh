#!/bin/bash

# Directory containing files
input_dir="/scratch/mhamon/data"
output_dir="/scratch/mhamon/results/"

# Create an array of file names
files=("$input_dir"/*)

# Write file list to a temporary file
file_list="nwb_list.txt"
printf "%s\n" "${files[@]}" > "$file_list"

# Submit Slurm array job
sbatch --array=0-$((${#files[@]}-1))%30 run_single_units_analysis.sh "$file_list" "$output_dir"
