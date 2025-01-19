#!/bin/bash

# Directory containing files
input_dir="/home/bisi/data/"
output_dir="/home/bisi/results/"

# Create an array of file names
files=("$input_dir"/*)

# Write file list to a temporary file
file_list="nwb_list.txt"
printf "%s\n" "${files[@]}" > "$file_list"

# Submit Slurm array job
sbatch --array=0-$((${#files[@]}-1))%10 run_xcorr_analysis.sh "$file_list" "$output_dir"
