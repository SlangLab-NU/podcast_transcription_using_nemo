'''
This script is used to transcribe audio files using the NeMo ASR model using buffer decoding.

The script uses the following classes:
    - ChunkBufferDecoder: A class to decode audio buffers using a sliding window approach.
    - AudioChunkIterator: A simple iterator class to return successive chunks of samples.
    - AudioBuffersDataLayer: A simple iterable dataset class to return a single buffer of samples.

The script takes the following command line arguments:
    - audio_path: Path to the audio file or directory containing audio files.
    - output_dir: Directory to save the txt files with transcriptions.
    - config_path: Path to the configuration file.
    - asr_model_path: Path to the ASR model.
    
Example usage:
    python transcribe.py <audio_path> <output_dir> <config_path> <asr_model_path>
'''
import os
import argparse
import configparser
import logging
import torch
import contextlib
import gc
import soundfile as sf
import scipy.signal
import numpy as np
import warnings
import nemo.collections.asr as nemo_asr

from AudioChunkIterator import AudioChunkIterator
from ChunkBufferDecoder import ChunkBufferDecoder

logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.getLogger('nemo.core').setLevel(logging.ERROR)
logging.getLogger('nemo.collections').setLevel(logging.ERROR)
logging.getLogger('nemo.collections.asr').setLevel(logging.ERROR)
logging.getLogger('nemo.utils').setLevel(logging.ERROR)
logging.getLogger('nemo').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Apex was not found.*")


class AudioTranscriber:
    def __init__(self, asr_model_path, config_path, device='cpu'):
        self.model_path = asr_model_path
        self.config = config_path
        self.init_config()

        # Check if device is available if 'cuda' is selected
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = 'cpu'

        self.device = torch.device(device)
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=asr_model_path)
        self.model = model.to(self.device)

    def init_config(self):
        """
        Initialize the configuration for transcribing audio
        """
        config_path = self.config

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config path {config_path} does not exist")

        config = configparser.ConfigParser()
        config.read(config_path)

        transcribe_config = {}
        if 'transcribe' not in config:
            print(
                "Config file does not contain the 'transcribe' section. Default configuration will be used.")
        else:
            transcribe_config = config['transcribe']

        self.sample_rate = int(transcribe_config.get('sr', 16000))
        self.chunk_len_in_sec = int(transcribe_config.get('chunk_len', 30))
        self.context_len_in_sec = int(transcribe_config.get('context_len', 5))

    def get_samples(self, audio_file: str, target_sr: int = 16000, chunk_size: int = 1024):
        """
        Load audio file in chunks and resample if necessary
        """
        with sf.SoundFile(audio_file, 'r') as f:
            sample_rate = f.samplerate
            num_frames = f.frames
            data = []

            # Process the audio file in chunks to save memory
            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                chunk = f.read(frames=end-start, dtype='float32')
                if sample_rate != target_sr:
                    chunk = scipy.signal.resample(chunk, int(
                        len(chunk) * target_sr / sample_rate))
                data.append(chunk)

        samples = np.concatenate(data, axis=0)

        # Check if the audio is mono or multi-channel
        if samples.ndim == 2:  # Multi-channel (e.g., stereo)
            samples = np.mean(samples, axis=1)  # Convert to mono by averaging

        return samples

    def transcribe_samples(self, samples: np.array):
        """
        Transcribe audio samples using the ASR model
        """
        model = self.model
        sample_rate = self.sample_rate
        chunk_len_in_sec = self.chunk_len_in_sec
        context_len_in_sec = self.context_len_in_sec

        # Buffer = context + chunk + context
        buffer_len_in_sec = chunk_len_in_sec + 2 * context_len_in_sec

        n_buffers = int(
            np.ceil(len(samples) / (sample_rate * chunk_len_in_sec)))
        buffer_len = int(sample_rate * buffer_len_in_sec)
        sampbuffer = np.zeros([buffer_len], dtype='float32')

        # Initialize the decoder to transcribe audio buffers
        chunk_reader = AudioChunkIterator(
            samples, sample_rate, chunk_len_in_sec)
        chunk_len = int(sample_rate * chunk_len_in_sec)
        count = 0
        buffer_list = []
        for chunk in chunk_reader:
            count += 1
            chunk_len = len(chunk)
            sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
            sampbuffer[-chunk_len:] = chunk

            buffer_list.append(np.array(sampbuffer))

            if count >= n_buffers:
                break

        stride = 4

        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        decoder = ChunkBufferDecoder(
            model, stride, chunk_len_in_sec, buffer_len_in_sec)

        transcription = decoder.transcribe_buffers(buffer_list, plot=False)

        return transcription

    def save_to_file(self, transcription: str, output_dir: str, audio_file: str):
        """
        Save transcription to a text file
        """
        # Check if output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output directory created: {output_dir}")

        file_name = os.path.basename(audio_file)
        file_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{file_name}.txt")

        with open(output_file, "w") as f:
            f.write(transcription)

    def transcribe_audio(self, audio_path: str, output_dir: str):
        """
        Transcribe audio using the ASR model
        """
        sample_rate = self.sample_rate
        audio_list = []

        # Check if audio path exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path {audio_path} does not exist")

        # Check if audio path is a file or directory
        if os.path.isfile(audio_path):
            audio_list.append(audio_path)
        elif os.path.isdir(audio_path):
            for root, _, files in os.walk(audio_path):
                for file in files:
                    if file.endswith('.wav'):
                        audio_list.append(os.path.join(root, file))
        else:
            raise ValueError(f"Invalid audio path: {audio_path}")

        file_count = len(audio_list)
        print(f"{file_count} audio files found.")

        for index, audio_file in enumerate(audio_list):
            index_message = f"Audio File {index + 1} out of {file_count} ({audio_file}):"
            print(index_message, "Sampling audio...")
            samples = self.get_samples(audio_file, sample_rate)

            print(index_message, "Transcribing audio...")
            transcription = self.transcribe_samples(samples)

            print(index_message, "Saving transcription...")
            self.save_to_file(transcription, output_dir, audio_file)

            print(index_message, "Completed.")


if __name__ == "__main__":

    print("Running transcribe.py")

    parser = argparse.ArgumentParser(
        description="Transcribe audio using NeMo ASR with buffer decoding")
    parser.add_argument("audio_path", type=str,
                        help="Path to the audio file or directory containing audio files")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save the txt files with transcriptions")
    parser.add_argument("config_path", type=str,
                        help="Path to the configuration file")
    parser.add_argument("asr_model_path", type=str,
                        help="Path to the ASR model")

    args = parser.parse_args()
    asr_model_path = args.asr_model_path
    config_path = args.config_path
    audio_path = args.audio_path
    output_dir = args.output_dir

    # Check if audio_path exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Invalid audio path: {audio_path}")

    # Check if config_path exists
    if not os.path.exists(config_path):
        print("Config path does not exist. Default configuration will be used.")

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    transcriber = AudioTranscriber(asr_model_path, config_path)
    transcriber.transcribe_audio(audio_path, output_dir)

    print("Completed transcribe.py")
