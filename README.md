# Data Preparation for the SFUSED Database

--- README is UNDER CONSTRUCTION ---

This repository contains a Dockerfile that can be used on the Cluster to transcript podcast using the NVIDIA NeMo Framework. The Dockerfile is used to create a Docker image that can be run on the Cluster. The image contains all the necessary dependencies to run the application.

Link to the NeMo model: https://github.com/NVIDIA/NeMo

If you do not wish to you use the Docker and you want to run the transcription on your local machine, you can follow the installation instructions in [Running on transcription on local machines](#running-on-local-machines)

The following files are included in the repository:

- Dockerfile: This file contains the instructions to build the Docker image.
- run_transcribe.sh: This bash script runs the transcription on all the files in the input directory.

To build the Dockerfile on your local machine, you need to have Docker installed. You can download Docker from the following link: https://docs.docker.com/desktop/

Alternatively, a Docker image has already been built, and it is available on this [Docker Hub page](https://hub.docker.com/repository/docker/macarious/nemo_asr/)

## Building Docker on Local Machine

First, start Docker on your local machine and log in to Docker Hub.

Run the following command in the root directory to build the dockerfile:

`docker build -t [docker_user_name]/nemo_asr .`

For example, to build the Docker image with the user name `macarious`, use the following command:

`docker build -t macarious/nemo_asr .`


## Running Docker on Local Machine

Run the Docker image on your local machine using the following command:

`docker run -it [docker_user_name]/nemo_asr`

For example, to run the Docker image with the user name `macarious`, use the following command:

`docker run -it macarious/nemo_asr`

The following is the output of the Docker image:

```
----------------------------------------
Start of script
Transcribing .mp3 files in the input directory...
[NeMo W 2024-05-01 10:23:30 optimizers:55] Apex was not found. Using the lamb or fused_adam optimizer will error out.
Manifest file: /nemo_asr_root/manifest.json
ASR model path: /nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo
Output directory: /nemo_asr_root/output
[NeMo I 2024-05-01 10:23:41 mixins:170] Tokenizer SentencePieceTokenizer initialized with 128 tokens
[NeMo W 2024-05-01 10:23:41 modelPT:142] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath:
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket1/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket2/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket3/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket4/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket5/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket6/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket7/tarred_audio_manifest.json
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket8/tarred_audio_manifest.json
    sample_rate: 16000
    batch_size: 1
    shuffle: true
    num_workers: 4
    pin_memory: true
    use_start_end_token: false
    trim_silence: false
    max_duration: 20
    min_duration: 0
    is_tarred: true
    tarred_audio_filepaths:
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket1/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket2/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket3/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket4/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket5/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket6/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket7/audio__OP_0..8191_CL_.tar
    - - /data/NeMo_ASR_SET/English/v3.0/train_bucketed/bucket8/audio__OP_0..8191_CL_.tar
    shuffle_n: 2048
    bucketing_strategy: synced_randomized
    bucketing_batch_size:
    - 64
    - 64
    - 32
    - 32
    - 16
    - 16
    - 16
    - 16

[NeMo W 2024-05-01 10:23:41 modelPT:149] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath:
    - /data/librispeech_withsp2/manifests/librivox-dev-other.json
    - /data/librispeech_withsp2/manifests/librivox-dev-clean.json
    - /data/librispeech_withsp2/manifests/librivox-test-other.json
    - /data/librispeech_withsp2/manifests/librivox-test-clean.json
    sample_rate: 16000
    batch_size: 4
    shuffle: false
    num_workers: 4
    pin_memory: true
    use_start_end_token: false
    is_tarred: false
    tarred_audio_filepaths: na
    min_duration: 0

[NeMo W 2024-05-01 10:23:41 modelPT:155] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath:
    - /data/librispeech_withsp2/manifests/librivox-test-other.json
    - /data/librispeech_withsp2/manifests/librivox-dev-clean.json
    - /data/librispeech_withsp2/manifests/librivox-dev-other.json
    - /data/librispeech_withsp2/manifests/librivox-test-clean.json
    sample_rate: 16000
    batch_size: 4
    shuffle: false
    num_workers: 4
    pin_memory: true
    use_start_end_token: false
    is_tarred: false
    tarred_audio_filepaths: na

[NeMo I 2024-05-01 10:23:41 features:225] PADDING: 0
[NeMo I 2024-05-01 10:23:48 save_restore_connector:243] Model EncDecCTCModelBPE was successfully restored from /nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo.
Transcribing:   0%|          | 0/1 [00:00<?, ?it/s][NeMo W 2024-05-01 10:23:48 nemo_logging:349] /usr/local/lib/python3.8/site-packages/torch/functional.py:665: UserWarning: stft with return_complex=False is deprecated. In a future pytorch release, stft will return complex tensors for all inputs, and return_complex=False will raise an error.
    Note: you can still call torch.view_as_real on the complex output to recover the old return format. (Triggered internally at ../aten/src/ATen/native/SpectralOps.cpp:873.)
      return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]

Transcribing: 100%|██████████| 1/1 [00:04<00:00,  4.24s/it]
Transcription:  astronomy cast episode one for monday september eleventh two thousand and six pluto and planet hood welcome to the astronomy cast my name is fraser kane and i am the webmaster of universe today and i also have with me pamela gay also from slacker astronomy fame and i
End of script
----------------------------------------
```

<!-- 
Push the Docker to Docker Hub so that it can be pulled to the Cluster:

`docker push [docker_user_name]/nemo_asr:latest`

For example, to push the Docker image with the user name `macarious`, use the following command:

`docker push macarious/nemo_asr:latest`

## Pull Docker to Cluster from Docker Hub

Log in to the Cluster (`@xfer`).

`ssh [user_name]@xfer.discovery.neu.edu`

To pull the Docker image from Docker Hub to the Cluster, load the Singularity module and run the following command:

`module load singularity/3.5.3`

`singularity pull docker://[docker_user_name]/nemo_asr:latest`

For example, to pull the pre-built Docker image with the user name `macarious`, use the following command:

`singularity pull docker://macarious/nemo_asr:latest`

This creates a file `nemo_asr_latest.sif` on the Cluster.

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
singularity run --nv --bind [input_path]:/input,[output_path]:/output, --pwd /app /work/van-speech-nlp/hui.mac/nemo_asr_latest.sif /bin/bash
```

For example, the following command mounts the input to `/work/van-speech-nlp/data/sfused/data` and the output to `/work/van-speech-nlp/hui.mac/sfused/transcription` and runs the Docker image:

```
singularity run --nv --bind /work/van-speech-nlp/data/sfused/data:/input,/work/van-speech-nlp/hui.mac/sfused/transcription:/output, --pwd /app /work/van-speech-nlp/hui.mac/nemo_asr_latest.sif /bin/bash
```

## Running Nemo with the Docker Image

Add execute permission and run bash script. This script runs the transcription on all the files in the input directory:

`chmod +x run_transcribe.sh | ./run_transcribe.sh`

Alternatively, to run NeMo on an individual file, use the following command:

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

Follow the installation instructions on the [NeMo GitHub page](https://github.com/NVIDIA/NeMoX).

Without a GPU, when running the whipserx on local machines, instead of `--compute_type float32`, the `--compute_type int8` is used to run the model on CPU. Use the following command to run whisperx on an individual file:

`whisperx /input/ac/ac001_2006-09-10.mp3 --output_dir /output --output_format json --suppress_numerals --compute_type int8` -->
