#!/bin/bash

# This script transcribes all .mp3 files in the input directory using the ASR model.

echo "----------------------------------------"
echo "ASR Transcription using NVIDIA NeMo Framework with Buffering"
echo ""
echo "Start of script"

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_path> <output_path>"
    exit 1
fi

# Set the paths to audio files, ASR model, and output directory
AUDIO_PATH=$1
OUTPUT_DIR=$2
CONFIG_PATH="/nemo_asr_root/transcribe.cfg"
ASR_MODEL_PATH="/nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo"

export PYTHONUNBUFFERED=1

# Record start time
START_TIME=$(date +%s)

# Start the Python script
python transcribe.py "$AUDIO_PATH" "$OUTPUT_DIR" "$CONFIG_PATH" "$ASR_MODEL_PATH"

# Record end time
END_TIME=$(date +%s)

# Calculate duration
DURATION=$(( END_TIME - START_TIME ))

# Print message indicating the end of transcription for all files
echo "End of script"
echo "Total transcription time: $DURATION seconds."
echo "----------------------------------------"
