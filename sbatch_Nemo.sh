#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output="%j.out"  # Output filename based on the job ID
#SBATCH --error="%j.err"   # Error filename based on the job ID

# Check if at least one configuration file is provided as an argument
if [ "$#" -lt 1 ]; then
    echo "No configuration files provided. Usage: sbatch sbatch_Nemo.sh <config_file1> <config_file2> ..."
    exit 1
fi

# Combine all the configuration files into a single string of arguments
config_files="$@"

# Check if config files exist
for config_file in $config_files; do
    if [ ! -f "$config_file" ]; then
        echo "Configuration file not found: $config_file"
        exit 1
    fi
done

# Load Singularity
module load singularity

# Pull the Docker image and convert it to a Singularity image if not already done

# New path here 
image_path=""
if [ ! -f "$image_path" ]; then
    echo "Pulling Docker image from Docker Hub and converting to Singularity image..."
    # New docker image here
    singularity pull docker: pull docker://haoniu08/nemo_asr:latest
fi

echo "Running experiment with configuration files: $config_files"

# Bind necessary directories and run the Singularity container
singularity run --nv \
    --bind /work/van-speech-nlp
    --pwd /app "$image_path" /bin/bash -c "
        
    echo 'Starting transcrbing with configuration files: $config_files'
    python3 src/training/main.py $config_files
"

echo "Completed transcription with mp3 files: $config_files"