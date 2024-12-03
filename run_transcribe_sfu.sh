#!/bin/bash
# This script transcribes all .mp3 files in the input directory using the NeMo ASR model.
echo "----------------------------------------"
echo "ASR Transcription using NVIDIA NeMo Framework with Buffering"
echo "Start of script"
echo "Transcribing .mp3 files in the input directory..."

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_path> <output_path>"
    exit 1
fi

# Set the paths
input_dir="$1"
output_dir="$2"
CONFIG_PATH="/nemo_asr_root/transcribe.cfg"
ASR_MODEL_PATH="/nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo"

# Set Python to unbuffered mode for real-time logging
export PYTHONUNBUFFERED=1

# Find all .mp3 files recursively in the input directory and list them
# Use -print0 to handle filenames with spaces and special characters
while IFS= read -r -d '' file; do
    mp3_files+=("$file")
done < <(find "$input_dir" -type f -name "*.mp3" -print0)

# Count total number of .mp3 files found
total_files=${#mp3_files[@]}

# Print out the total number of .mp3 files found
echo "Found $total_files .mp3 files"

# Counter for tracking progress
processed_files=0

# Record overall start time
START_TIME=$(date +%s)

# Loop through each .mp3 file
for file in "${mp3_files[@]}"; do
    # Increment processed files counter
    ((processed_files++))

    # Extract the filename with extension
    filename=$(basename "$file")

    # Extract the filename without extension
    filename_no_ext="${filename%.*}"

    # Determine the corresponding subdirectory in the output directory
    output_subdir=$(dirname "${output_dir}${file#$input_dir}")

    # Check if a corresponding .txt file already exists in the output directory
    if [ -f "${output_subdir}/${filename_no_ext}_output.txt" ]; then
        # If a corresponding .txt file already exists, skip the current file
        echo "Skipping '$filename' ($processed_files/$total_files); TXT file already exists in the output directory"
        continue
    fi

    # Create the corresponding subdirectory in the output directory
    mkdir -p "$output_subdir"

    # Print message indicating the start of transcription for the current file
    echo "----------------------------------------"
    echo "Transcribing '$filename' ($processed_files/$total_files)"
    echo

    # Run NeMo transcription on the current file
    python transcribe.py "$file" "$output_subdir" "$CONFIG_PATH" "$ASR_MODEL_PATH"

    # Print message indicating the completion of transcription for the current file
    echo
    echo "Transcription completed for '$filename'"
    echo "----------------------------------------"
done

echo "Checking output files..."
./run_check_output.sh

# Calculate and display total execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print final statistics
echo "End of script"
echo "Total files processed: $processed_files/$total_files"
echo "Total transcription time: $DURATION seconds"
echo "----------------------------------------"