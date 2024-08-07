import warnings
import nemo.collections.asr as nemo_asr
from ChunkBufferDecoder import ChunkBufferDecoder
from AudioChunkIterator import AudioChunkIterator
from typing import List, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
import gc
import torch
from contextlib import contextmanager
import configparser
import argparse
import logging
import os
from io import BytesIO
from typing import Union
warnings.filterwarnings("ignore", category=UserWarning,
                        message="stft with return_complex=False is deprecated")
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

@contextmanager
def open_audio(input_data: Union[str, bytes]):
    if isinstance(input_data, str):
        # input_data is a file path
        audio_file = sf.SoundFile(input_data, 'r')
    elif isinstance(input_data, bytes):
        # input_data is bytes data
        audio_buffer = BytesIO(input_data)
        audio_file = sf.SoundFile(audio_buffer, 'rb')
    else:
        raise ValueError("input_data must be a file path (str) or bytes data")

    try:
        yield audio_file
    finally:
        audio_file.close()

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

    def get_samples(self, input_data: Union[str, bytes], target_sr: int = 16000, chunk_size: int = 1024):
        """
        Load audio file in chunks and resample if necessary.
        Returns samples for both left and right channels separately if stereo, or only left if mono.
        """
        with open_audio(input_data) as f:
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
            left_channel = samples[:, 0]
            right_channel = samples[:, 1]
        else:
            left_channel = samples
            right_channel = None  # Mono audio

        # If both channels are the same, return only one channel
        if np.array_equal(left_channel, right_channel):
            right_channel = None

        return left_channel, right_channel

    def transcribe_samples(self, samples: Tuple[np.array, np.array]):
        """
        Transcribe audio samples for both left and right channels using the ASR model.
        """
        left_samples, right_samples = samples
        transcriptions = []
        channels = ['left', 'right']

        for channel, samples in zip(channels, [left_samples, right_samples]):
            # Skip the right channel if it's the same as the left (mono audio)
            if left_samples is not None and right_samples is not None and np.array_equal(left_samples, right_samples) and channel == 'right':
                continue

            try:
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
                buffer_offsets = []
                for chunk in chunk_reader:
                    count += 1
                    chunk_len = len(chunk)
                    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
                    sampbuffer[-chunk_len:] = chunk

                    buffer_list.append(np.array(sampbuffer))
                    # Offset by chunk length
                    buffer_offsets.append((count - 1) * chunk_len_in_sec)

                    if count >= n_buffers:
                        break

                stride = 4

                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                decoder = ChunkBufferDecoder(
                    model, stride, chunk_len_in_sec, buffer_len_in_sec)

                for buffer, buffer_offset in zip(buffer_list, buffer_offsets):
                    transcription, timestamps = decoder.transcribe_buffers(
                        [buffer], merge=False, buffer_offset=buffer_offset)
                    for t, ts in zip(transcription, timestamps):
                        transcriptions.append((t, ts, channel))
            except Exception as e:
                print(f"An error occurred: {e}")
                
        return transcriptions

    def save_to_file(self, transcriptions: List[Tuple[str, Tuple[float, float], str]], output_dir: str, audio_file: str):
        """
        Save transcription to a text file.
        """
        # Check if output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output directory created: {output_dir}")

        file_name = os.path.basename(audio_file)
        file_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{file_name}_output.txt")

        # Sort transcriptions by start time
        transcriptions.sort(key=lambda x: x[1][0])

        with open(output_file, "w") as f:
            for (transcript, timestamps, channel) in transcriptions:
                start_time, end_time = timestamps
                if start_time is None:
                    start_time = 0.0
                if end_time is None:
                    end_time = 0.0
                f.write(
                    f"{transcript} {start_time:.2f} {end_time:.2f} {channel} 1.00\n")

    def transcribe_api(self, input_data: Union[str, bytes], sample_rate=16000):
        sample_rate = self.sample_rate
        samples = self.get_samples(input_data, sample_rate)
        transcriptions = self.transcribe_samples(samples)
        return transcriptions

    def transcribe_audio(self, audio_path: str, output_dir: str):
        """
        Transcribe audio using the ASR model.
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
            transcriptions = self.transcribe_samples(samples)

            print(index_message, "Saving transcription...")
            self.save_to_file(transcriptions, output_dir, audio_file)

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
