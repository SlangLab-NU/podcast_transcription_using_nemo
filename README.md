# Data Preparation for the SFUSED Database

--- README is UNDER CONSTRUCTION ---

This repository contains a Dockerfile that can be used on the Cluster to transcript podcast using the WhisperX model. The Dockerfile is used to create a Docker image that can be run on the Cluster. The image contains all the necessary dependencies to run the application.

Link to the WhisperX model: https://github.com/m-bain/whisperX

If you do not wish to you use the Docker and you want to run the transcription on your local machine, you can follow the installation instructions in [Running on transcription on local machines](#running-on-local-machines)

The following files are included in the repository:

- Dockerfile: This file contains the instructions to build the Docker image.
- run_transcribe.sh: This bash script runs the transcription on all the files in the input directory.

To build the Dockerfile on your local machine, you need to have Docker installed. You can download Docker from the following link: https://docs.docker.com/desktop/

Alternatively, a Docker image has already been built, and it is available on this [Docker Hub page](https://hub.docker.com/repository/docker/macarious/whisperx/)

## Building Docker on Local Machine

First, start Docker on your local machine and log in to Docker Hub.

Run the following command in the root directory to build the dockerfile:

`docker build -t [docker_user_name]/whisperx .`

For example, to build the Docker image with the user name `macarious`, use the following command:

`docker build -t macarious/whisperx .`

Push the Docker to Docker Hub so that it can be pulled to the Cluster:

`docker push [docker_user_name]/whisperx:latest`

For example, to push the Docker image with the user name `macarious`, use the following command:

`docker push macarious/whisperx:latest`

## Pull Docker to Cluster from Docker Hub

Log in to the Cluster (`@xfer`).

`ssh [user_name]@xfer.discovery.neu.edu`

To pull the Docker image from Docker Hub to the Cluster, load the Singularity module and run the following command:

`module load singularity/3.5.3`

`singularity pull docker://[docker_user_name]/whisperx:latest`

For example, to pull the pre-built Docker image with the user name `macarious`, use the following command:

`singularity pull docker://macarious/whisperx:latest`

This creates a file `whisperx_latest.sif` on the Cluster.

## Requesting GPU on the Cluster

Switch to (`@login`) and use GPU from Cluster (see https://github.com/SlangLab-NU/links/wiki/Working-with-sbatch-and-srun-on-the-cluster-with-GPU-nodes):

`[user_name]@login.discovery.neu.edu`

Check the status of the GPU nodes:

`sinfo -p gpu --Format=nodes,cpus,memory,features,statecompact,nodelist,gres`

Request for GPU (the following command requests for the t4 GPU for 8 hours):

`srun --partition=gpu --nodes=1 --gres=gpu:t4:1 --time=08:00:00 --pty /bin/bash`

## Running the Docker Image on the GPU Node

Load singularity on the GPU node:

`module load singularity/3.5.3`

Execute the Docker image using Singularity. The `/input` and `/output` directories are the input and output directories, respectively, and they need to be mounted to the Docker image using the `--bind` option. The following command mounts the input and output directories and runs the Docker image:

```
singularity run --nv --bind [input_path]:/input,[output_path]:/output, --pwd /whisperx_root /work/van-speech-nlp/hui.mac/whisperx_latest.sif /bin/bash
```

For example, the following command mounts the input to `/work/van-speech-nlp/data/sfused/data` and the output to `/work/van-speech-nlp/hui.mac/sfused/transcription` and runs the Docker image:

```
singularity run --nv --bind /work/van-speech-nlp/data/sfused/data:/input,/work/van-speech-nlp/hui.mac/sfused/transcription:/output, --pwd /whisperx_root /work/van-speech-nlp/hui.mac/whisperx_latest.sif /bin/bash
```

## Running WhisperX with the Docker Image

Log in to Hugging Face (not necessary if you do not plan to use the diariation model):

`huggingface-cli login`

Add execute permission and run bash script. This script runs the transcription on all the files in the input directory:

`chmod +x run_transcribe.sh | ./run_transcribe.sh`

Alternatively, to run whisperx on an individual file, use the following command:

`whisperx /input/[path_to_mp3] --output_dir /output --output_format json --suppress_numerals`

For example, to run whisperx on the file `ac001_2006-09-10.mp3` in the input directory, use the following command:

`whisperx /input/ac/ac001_2006-09-10.mp3 --output_dir /output --output_format json --suppress_numerals`

The `--output_format json` option specifies the output format as JSON. The `--suppress_numerals` option suppresses the numerals in the output.

## Diariation

--- Instructions for diariation will be added soon. ---

## Checking the output files

Run the following script to check all the output files and see if they match the input files:

`chmod +x run_check_output.sh | ./run_check_output.sh`

The script checks if any output files are missing and lists the missing files.

## Running on transcription on local machines {#running-on-local-machines}

Follow the installation instructions on the [WhisperX GitHub page](https://github.com/m-bain/whisperX).

Without a GPU, when running the whipserx on local machines, instead of `--compute_type float32`, the `--compute_type int8` is used to run the model on CPU. Use the following command to run whisperx on an individual file:

`whisperx /input/ac/ac001_2006-09-10.mp3 --output_dir /output --output_format json --suppress_numerals --compute_type int8`

