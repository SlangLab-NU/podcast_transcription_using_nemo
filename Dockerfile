# Use Python 3.8 as the base image
FROM python:3.8

# Install required system dependencies
RUN apt-get update && apt-get install -y git sox libsndfile1 ffmpeg wget unzip

# Install Python dependencies
RUN pip install numpy flask-cors omegaconf pytest scipy torch torchvision torchaudio Flask gunicorn unidecode Cython wheel ffmpeg-python

# Clone NeMo repository and install NeMo ASR
ARG BRANCH=r1.13.0
RUN pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@$BRANCH"
RUN pip install huggingface_hub==0.22.0

# Download and unzip the ASR model
RUN mkdir -p /nemo_asr_root/model && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_xlarge/versions/1.10.0/zip -O stt_en_conformer_ctc_xlarge_1.10.0.zip && \
    unzip stt_en_conformer_ctc_xlarge_1.10.0.zip -d /nemo_asr_root/model && \
    rm stt_en_conformer_ctc_xlarge_1.10.0.zip

# Create output directory
RUN mkdir -p /nemo_asr_root/output
RUN mkdir -p /nemo_asr_root/sample

# Set working directory
WORKDIR /nemo_asr_root

# Copy scripts and configuration files into the container
COPY run_transcribe_sfu.sh /nemo_asr_root/run_transcribe_sfu.sh
COPY transcribe.cfg /nemo_asr_root/transcribe.cfg
COPY AudioBuffersDataLayer.py /nemo_asr_root/AudioBuffersDataLayer.py
COPY AudioChunkIterator.py /nemo_asr_root/AudioChunkIterator.py
COPY ChunkBufferDecoder.py /nemo_asr_root/ChunkBufferDecoder.py
COPY transcribe.py /nemo_asr_root/transcribe.py
COPY audio.py /nemo_asr_root/audio.py

# Make the run_transcribe_sfu.sh script executable
RUN chmod +x /nemo_asr_root/run_transcribe_sfu.sh
 
# Entrypoint to run the bash script
ENTRYPOINT ["/nemo_asr_root/run_transcribe_sfu.sh"]