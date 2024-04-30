#!/bin/bash

# This script transcribes all .mp3 files in the input directory using the whisperx command-line tool.

echo "----------------------------------------"
echo "Start of script"
echo "Transcribing .mp3 files in the input directory..."

# Activate Conda environment
source /opt/conda/bin/activate whisperx

# Specify the input directory containing the .mp3 files
input_dir="/input"

# Specify the output directory
output_dir="/output"

# Arguments for whisperx
model="large-v2"
device="cuda"
batch_size=8
compute_type="float16"
output_format="json"
HF_TOKEN="" # For diarization

# Find all .mp3 files recursively in the input directory and list them
# Exclude files for which a corresponding .json file already exists in the output directory
mp3_files=$(find "$input_dir" -type f -name "*.mp3")

# Count total number of .mp3 files found
total_files=$(echo "$mp3_files" | wc -l)

# Print out the total number of .mp3 files found
echo "Found $total_files .mp3 files"

# Counter for tracking progress
processed_files=0

# Loop through each .mp3 file
for file in $mp3_files; do
    # Increment processed files counter
    ((processed_files++))

    # Extract the filename with extension
    filename=$(basename "$file")

    # Extract the filename without extension
    filename_no_ext="${filename%.*}"

    # Determine the corresponding subdirectory in the output directory (to maintain the same directory structure as the input directory)
    output_subdir=$(dirname "${output_dir}${file#$input_dir}")

    # Check if a corresponding .json file already exists in the output directory
    if [ -f "${output_subdir}/${filename_no_ext}.json" ]; then
        # If a corresponding .json file already exists, skip the current file
        echo "Skipping $filename ($processed_files/$total_files); JSON file already exists in the output directory"
        continue
    fi

    # Create the corresponding subdirectory in the output directory
    mkdir -p "$output_subdir"

    # Print message indicating the start of transcription for the current file
    echo "----------------------------------------"
    echo "Transcribing $filename ($processed_files/$total_files)"
    echo

    # Run the whisperx command on each file, saving output in the corresponding directory
    whisperx "$file" \
        --model "$model" \
        --device "$device" \
        --batch_size "$batch_size" \
        --compute_type "$compute_type" \
        --output_format "$output_format" \
        --output_dir "$output_subdir" \
        --suppress_numerals \
    # --diarize --hf_token "$HF_TOKEN"

    # Print message indicating the completion of transcription for the current file
    echo
    echo "Transcription completed for $filename"
    echo "----------------------------------------"
done

# Deactivate Conda environment
conda deactivate

# Print message indicating the end of transcription for all files
echo "End of script"
echo "----------------------------------------"
