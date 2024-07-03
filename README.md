# ASR Transcription using NVIDIA NeMo Framework with Buffering

--- README is UNDER CONSTRUCTION ---

This repository contains a Dockerfile that can be used to transcribe audio files using the NVIDIA NeMo Framework.

Link to the NeMo model: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)

## Files Included in the Repository

- `Dockerfile`: Instructions to build the Docker image.
- `run_transcribe.sh`: Bash script to run the transcription on all the files in the input directory.
- `transcribe.py`: Python script to transcribe the audio files using the NeMo model.
- `transcribe.cfg`: Configuration file required to run the `transcribe.py` script.
- `AudioBuffersDataLayer.py`: Buffers the audio files before transcription.
- `AudioChunkIterator.py`: Iterates over the audio chunks.
- `ChunkBufferDecoder.py`: Decodes the buffered chunks.

## 1. Pulling the Docker Image

To pull the Docker image from Docker Hub, use the following command:

```
docker pull macarious/nemo_asr:latest
```

## 2. Running Docker

Run the Docker image on your local machine using the following command:

```
docker run --rm -v [path_to_audio_files]:/input -v [path_to_output_dir]:/output macarious/nemo_asr /input /output
```

The `[path_to_audio_files]` should be replaced with the path to the directory containing the audio files or a specific audio file. The `[path_to_output_dir]` should be replaced with the path to the directory where the output transcripts will be saved.

For example, if the audio files are located in the directory `./data/audio` and you want the output transcripts to be saved in the directory `./transcripts`, you can run the following command:

```
docker run --rm -v "./data/audio:/input" -v "./transcripts:/output" macarious/nemo_asr /input /output
```

## Alternatively, Building and Running the Docker Image Locally

To build the Docker image locally, navigate to the directory containing the `Dockerfile` and run the following command:

```
docker build -t nemo_asr .
```

After building the Docker image, you can run it using the following command:

```
docker run --rm -v [path_to_audio_files]:/input -v [path_to_output_dir]:/output nemo_asr /input /output
```

The `[path_to_audio_files]` should be replaced with the path to the directory containing the audio files or a specific audio file. The `[path_to_output_dir]` should be replaced with the path to the directory where the output transcripts will be saved.

For example, if the audio files are located in the directory `./data/audio` and you want the output transcripts to be saved in the directory `./transcripts`, you can run the following command:

```
docker run --rm -v "./data/audio:/input" -v "./transcripts:/output" nemo_asr /input /output
```

To push the Docker image to Docker Hub, you need to tag the image with your Docker Hub username and push it using the following commands:

```
docker tag nemo_asr [docker_hub_username]/nemo_asr
docker push [docker_hub_username]/nemo_asr
```

Replace `[docker_hub_username]` with your Docker Hub username

For example, if your Docker Hub username is `macarious`, you would run the following commands:

```
docker tag nemo_asr macarious/nemo_asr
docker push macarious/nemo_asr
```

---

### Additional Notes:

1. Ensure the paths `[path_to_audio_files]` and `[path_to_output_dir]` are correctly replaced with the actual paths on your local machine.
2. The Docker image name should match the one you built or pulled from Docker Hub.

## Troubleshooting

If you encounter any issues, ensure that:

1. Docker is running on your machine.
2. The volume paths are correctly specified.
3. You have the necessary permissions to access the specified directories.

For further assistance, refer to the Docker documentation: [Docker Documentation](https://docs.docker.com/)

---

Feel free to add any additional instructions or information that might be helpful for users of your repository.
