# Use Python 3.8 as the base image
FROM python:3.8

# Install required system dependencies
RUN apt-get update && apt-get install -y git sox ffmpeg wget unzip

# Install Python dependencies
RUN pip install numpy omegaconf scipy torch torchvision torchaudio Flask gunicorn

# Install Cython for NeMo
RUN pip install Cython

# Clone NeMo repository and install NeMo ASR
ARG BRANCH=r1.13.0
RUN pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

# Download and unzip the ASR model
RUN mkdir -p /nemo_asr_root/model && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_xlarge/versions/1.10.0/zip -O stt_en_conformer_ctc_xlarge_1.10.0.zip && \
    unzip stt_en_conformer_ctc_xlarge_1.10.0.zip -d /nemo_asr_root/model && \
    rm stt_en_conformer_ctc_xlarge_1.10.0.zip

# Create a output directory
RUN mkdir -p /nemo_asr_root/output
RUN mkdir -p /nemo_asr_root/sample

# Set the working directory
WORKDIR /nemo_asr_root

# Copy your scripts and configuration files into the container
COPY run_transcribe.sh /nemo_asr_root/run_transcribe.sh
COPY transcribe.py /nemo_asr_root/transcribe.py
COPY run_check_output.sh /nemo_asr_root/run_check_output.sh
COPY manifest.json /nemo_asr_root/manifest.json
COPY sample/ac001_2006-09-10.wav /nemo_asr_root/sample/ac001_2006-09-10.wav

# Expose the port the nemo_asr_root runs on
EXPOSE 5000

# Define the command to run the nemo_asr_rootlication
CMD ["bash", "run_transcribe.sh"]