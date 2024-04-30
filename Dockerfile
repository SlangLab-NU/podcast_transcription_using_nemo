# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Create Python 3.10 environment and install PyTorch
RUN conda create --name whisperx python=3.10 && \
    /bin/bash -c "source activate whisperx && \
    conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda clean -afy && \
    pip install git+https://github.com/m-bain/whisperx.git pyannote.audio && \
    pip install huggingface-hub pyannote.audio && \
    apt-get update && apt-get install -y ffmpeg vim && \
    rm -rf /var/lib/apt/lists/*"

# Set the working directory in the container
WORKDIR /whisperx_root

# Copy the current directory contents into the container at /app
COPY . /whisperx_root

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["whisperx"]
