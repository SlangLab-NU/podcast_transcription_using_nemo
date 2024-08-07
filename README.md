# ASR Transcription using NVIDIA NeMo Framework with Buffering

--- README is UNDER CONSTRUCTION ---

This repository contains a Dockerfile that can be used to transcribe audio files using the NVIDIA NeMo Framework.

Link to the NeMo model: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)

## Files Included in the Repository

- `Dockerfile-api`: Instructions to build the Docker image to run the code as an API.
- `Dockerfile`: Instructions to build the Docker image.
- `run_transcribe.sh`: Bash script to run the transcription on all the files in the input directory.
- `transcribe.py`: Python script to transcribe the audio files using the NeMo model.
- `transcribe.cfg`: Configuration file required to run the `transcribe.py` script.
- `AudioBuffersDataLayer.py`: Buffers the audio files before transcription.
- `AudioChunkIterator.py`: Iterates over the audio chunks.
- `ChunkBufferDecoder.py`: Decodes the buffered chunks.

## 1. Pulling the Docker Image (for running file based transcription using the Dockerfile)

To pull the Docker image from Docker Hub, use the following command:

```
docker pull macarious/nemo_asr:latest
```

## 2. Running Docker (for running file based transcription using the Dockerfile)

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

## 3. Running the API using Dockerfile-api 

Build the Docker image locally like so:
```
docker build -f Dockerfile-api -t flask-gunicorn-app .
```

Run the Docker container to start up the API, this will be available at `http://localhost:8000` :
```
docker run -p 8000:8000 flask-gunicorn-app
```

Once the container is up and running, to verify that it is running navigate to `http://localhost:8000` in your browser window where you should see the default route displaying, "Testing Flask!"

<img width="278" alt="Screen Shot 2024-08-07 at 10 16 36 AM" src="https://github.com/user-attachments/assets/0a098f7f-b461-4159-99b2-5c518e76480a">

To test the API with Postman, first encode an audio file like so to turn it into a base64 string (inspired by the [Google speech-to-text API](https://cloud.google.com/speech-to-text/docs/base64-encoding)):
```
# Import the base64 encoding library.
import base64

# Pass the audio data to an encoding function.
input_audio_file = os.path.join(cwd, 'input.wav')
with open(input_audio_file, 'rb') as wav_file:
  wav_data = wav_file.read()
  base64_encoded = base64.b64encode(wav_data).decode('utf-8')
```

Use the base64_encoded string to compose a JSON object like so:

```
{
  "config": {
    "sample_rate": 16000,
  },
  "audio": {
    "content": "ZkxhQwAAACIQABAAAAUJABtAA+gA8AB+W8FZndQvQAyjv..."
  }
}

```
Right now (as of Aug 7, 2024) `sample_rate` does not do much. It assumes that the string can be converted to raw bytes which contains a WAV header. There is internal resampling on a per-chunk basis, but this needs to be tested with varying sampling rates input via the API. 

Once composed use Postman to hit the `/predict` endpoint for the container running locally by setting the Body parameter to `raw` and `JSON` like the screenshot below:
![Screen Shot 2024-08-07 at 10 25 57 AM](https://github.com/user-attachments/assets/031cd2b2-e367-4853-9f66-0922b1915fbe)

You should receive a `200 OK` response back with the transcription.

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
