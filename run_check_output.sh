#!/bin/bash

# This script checks the output of the transcription process and prints a summary of the results.
echo "----------------------------------------"
echo "Start of script"

# Specify the input directory containing the .mp3 files
input_dir="/input"

# Specify the output directory containing the .json files
output_dir="/output"

# Find all .mp3 files recursively in the input directory and list them
mp3_files=$(find "$input_dir" -type f -name "*.mp3")

# Find all .json files recursively in the output directory and list them
json_files=$(find "$output_dir" -type f -name "*.json")

# Count total number of .mp3 files found
total_files=$(echo "$mp3_files" | wc -l)

# Count total number of .json files found
total_json_files=$(echo "$json_files" | wc -l)

# Print out the total number of .mp3 files found
echo "Found $total_files .mp3 files"

# Print out the total number of .json files found
echo "Found $total_json_files .json files"

# Print out the total number of .mp3 files for which a corresponding .json file was not found
echo "Found $(($total_files - $total_json_files)) .mp3 files for which a corresponding .json file was not found"

# List .mp3 files for which a corresponding .json file was not found
for file in $mp3_files; do
    filename=$(basename "$file")
    filename_no_ext="${filename%.*}"
    output_subdir=$(dirname "${output_dir}${file#$input_dir}")
    if [ ! -f "${output_subdir}/${filename_no_ext}.json" ]; then
        echo "No corresponding .json file found for $filename"
    fi
done

# Print message indicating the end of transcription for the current file
echo "End of script"
echo "----------------------------------------"
