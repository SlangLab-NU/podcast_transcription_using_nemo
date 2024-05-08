#!/bin/bash

# This script transcribes all .mp3 files in the input directory using the whisperx command-line tool.

echo "----------------------------------------"
echo "Start of script"
echo "Transcribing .mp3 files in the input directory..."

# Specify the input directory containing the .mp3 files
input_dir="/input"

# Specify the output directory
output_dir="/output"

# Set the paths to the manifest file, ASR model, and output directory
MANIFEST_FILE="/nemo_asr_root/manifest.json"
ASR_MODEL_PATH="/nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo"
OUTPUT_DIR="/nemo_asr_root/output"

# Start the Python script in the background
python transcribe.py "$MANIFEST_FILE" "$ASR_MODEL_PATH" "$OUTPUT_DIR" &
PYTHON_PID=$!

# Start py-spy to record the Python process
py-spy record --pid $PYTHON_PID --output transcribe.svg &

# Wait for the Python script to complete
wait $PYTHON_PID

# Print message indicating the end of transcription for all files
echo "End of script"
echo "----------------------------------------"
