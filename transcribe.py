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
from audio import convert_mp3_to_wav

warnings.filterwarnings("ignore", category=UserWarning,
                        message="stft with return_complex=False is deprecated")
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


@contextmanager
def open_audio(input_data: Union[str, bytes, BytesIO]) -> sf.SoundFile:
    """
    Open audio file using soundfile library.

    Args:
        input_data: Path to the audio file, bytes data of the audio file, or BytesIO object.

    Returns:
        SoundFile object for reading audio data.
    """
    if isinstance(input_data, str):
        # input_data is a file path
        audio_file = sf.SoundFile(input_data, 'r')
    elif isinstance(input_data, bytes):
        # input_data is bytes data
        audio_buffer = BytesIO(input_data)
        audio_file = sf.SoundFile(audio_buffer, 'rb')
    elif isinstance(input_data, BytesIO):
        # input_data is already a BytesIO object
        audio_file = sf.SoundFile(input_data, 'rb')
    else:
        raise ValueError("input_data must be a file path (str), bytes data, or BytesIO object")

    try:
        yield audio_file
    finally:
        audio_file.close()


class AudioTranscriber:
    def __init__(self, asr_model_path: str, config_path: str, device: str = 'cpu'):
        """
        Initialize the AudioTranscriber object with the ASR model and configuration.

        Args:
            asr_model_path (str): Path to the ASR model.
            config_path (str): Path to the configuration file.
            device (str): Device to use for inference. Default is 'cpu'.

        Returns:
            None
        """
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
        Initialize the configuration for transcribing audio.

        The configuration file should contain the following parameters:
        - sr: Sample rate for audio. Default is 16000.
        - chunk_len: Length of audio chunks in seconds. Default is 30.
        - context_len: Length of context in seconds. Default is 5.
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
        self.stride = int(transcribe_config.get('stride', 4))
        
        logging.debug(f"Sample rate: {self.sample_rate}, Chunk length: {self.chunk_len_in_sec}, Context length: {self.context_len_in_sec}, Stride: {self.stride}")


    def compare_channels(self, left_samples: np.array, right_samples: np.array, tolerance: float = 1e-4) -> bool:
        """
        Compare left and right audio channels to check if they are the same.

        Args:
            left_samples (np.array): Left audio samples.
            right_samples (np.array): Right audio samples.
            tolerance (float): Tolerance level for comparing the two channels. Default is 1e-4.

        Returns:
            Boolean value indicating if the two channels are the same within the tolerance level.
        """
        if left_samples is None or right_samples is None:
            return False

        mean_absolute_diff = np.mean(np.abs(left_samples - right_samples))
        return mean_absolute_diff < tolerance

    def get_samples(self, input_data: Union[str, bytes], original_sr: int = None, target_sr: int = 16000, chunk_size: int = 1024):
        """
        Load audio file in chunks and resample if necessary.

        Args:
            input_data: Path to the audio file or bytes data of the audio file.
            original_sr: Original sample rate of the audio file.
            target_sr: Target sample rate for resampling. Default is 16000.
            chunk_size: Number of frames to read at a time. Default is 1024.

        Returns:
            Tuple of left and right audio samples. If the audio is mono, right channel is None.
        """
        # Convert MP3 to WAV if necessary
        if (isinstance(input_data, str) and input_data.endswith('.mp3') 
            or isinstance(input_data, bytes)):
            input_data = convert_mp3_to_wav(input_data)

        with open_audio(input_data) as f:
            logging.info(f"Resampling from {f.samplerate} to {target_sr}")
            sample_rate = f.samplerate if original_sr is None else original_sr
            num_frames = f.frames
            data = []

            # Process the audio file in chunks to save memory
            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                chunk = f.read(frames=end-start, dtype='float32')
                data.append(chunk)

        samples = np.concatenate(data, axis=0)

        # Check if the audio is mono or multi-channel
        if samples.ndim == 2:  # Multi-channel (e.g., stereo)
            logging.info("Multi-channel audio detected.")
            left_channel = samples[:, 0]
            right_channel = samples[:, 1]

            # Resample both channels separately if necessary
            logging.info(
                f"sample_rate: {sample_rate}, target_sr: {target_sr}, sample_rate != target_sr: {sample_rate != target_sr}")
            if sample_rate != target_sr:
                logging.info("Resampling left and right channels...")
                left_channel = self.resample_channel(
                    left_channel, sample_rate, target_sr)
                right_channel = self.resample_channel(
                    right_channel, sample_rate, target_sr)
        else:
            logging.info("Mono audio detected.")
            left_channel = samples
            right_channel = None  # Mono audio

            # Resample the mono channel if necessary
            if sample_rate != target_sr:
                left_channel = self.resample_channel(
                    left_channel, sample_rate, target_sr)

        # If both channels are the same (within tolerance), return only one channel
        if self.compare_channels(left_channel, right_channel):
            right_channel = None

        return left_channel, right_channel

    def resample_channel(self, channel_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample a single audio channel to the target sample rate.

        Args:
            channel_data (np.ndarray): The audio data to be resampled.
            original_sr (int): The original sample rate of the audio data.
            target_sr (int): The target sample rate for the resampled audio.

        Returns:
            np.ndarray: The resampled audio data.
        """
        num_samples = int(len(channel_data) * target_sr / original_sr)
        logging.info(
            f"Original Samples: {len(channel_data)}, Resampled Samples: {num_samples}")
        resampled_data = scipy.signal.resample(channel_data, num_samples)
        return resampled_data

    def transcribe_samples(self, samples: Tuple[np.array, np.array]) -> List[Tuple[str, Tuple[float, float], str]]:
        """
        Transcribe audio samples using the ASR model. If the audio is mono, only the left channel is transcribed.

        Args:
            samples [Tuple[np.array, np.array]]: Tuple of left and right audio samples.

        Returns:
            List of transcriptions with timestamps and channel information.
        """
        left_samples, right_samples = samples
        transcriptions = []
        channels = ['left', 'right']

        for channel, samples in zip(channels, [left_samples, right_samples]):
            # Skip the right channel if it's the same as the left (mono audio)
            logging.info(f"Transcribing {channel} channel...")
            if samples is None or (channel == 'right' and self.compare_channels(left_samples, right_samples)):
                logging.info(f"Skipping {channel} channel.")
                continue
            
            logging.info("")

            # try:
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

            # Keep below (chunk_len / 6); smaller the value the better the resolution
            stride = self.stride

            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            decoder = ChunkBufferDecoder(
                model, stride, chunk_len_in_sec, buffer_len_in_sec, context_len_in_sec)

            count = 1
            for buffer, buffer_offset in zip(buffer_list, buffer_offsets):
                decoder.reset()
                transcription, timestamps = decoder.transcribe_buffers(
                    [buffer], merge=False, buffer_offset=buffer_offset)
                for t, ts in zip(transcription, timestamps):
                    transcriptions.append((t, ts, channel))
            # except Exception as e:
            #     print(f"An error occurred: {e}")

        return transcriptions

    def save_to_file(self, transcriptions: List[Tuple[str, Tuple[float, float], str]], output_dir: str, audio_file: str):
        """
        Save transcription to a text file. The text file will contain the transcriptions with timestamps and channel information.

        Args:
            transcriptions (List[Tuple[str, Tuple[float, float], str]]): List of transcriptions with timestamps and channel information.
            output_dir (str): Directory to save the output text files.
            audio_file (str): Path to the audio file.

        Returns:
            None
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

    def transcribe_api(self, input_data: Union[str, bytes], sample_rate=16000) -> List[Tuple[str, Tuple[float, float], str]]:
        """
        Transcribe audio using the ASR model.

        Args:
            input_data (Union[str, bytes]): Path to the audio file or bytes data of the audio file.
            sample_rate (int): Sample rate of the audio file. Default is 16000.

        Returns:
            List of transcriptions with timestamps and channel information.
        """
        samples = self.get_samples(input_data)
        transcriptions = self.transcribe_samples(samples)
        return transcriptions

    def transcribe_audio(self, audio_path: str, output_dir: str):
        """
        Transcribe audio using the ASR model. The audio can be a single file or a directory containing multiple audio files.

        Args:
            audio_path (str): Path to the audio file or directory containing audio files.
            output_dir (str): Directory to save the output text files.

        Returns:
            None
        """
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
                    if file.endswith('.wav') or file.endswith('.mp3'):
                        audio_list.append(os.path.join(root, file))
        else:
            raise ValueError(f"Invalid audio path: {audio_path}")

        file_count = len(audio_list)
        print(f"{file_count} audio files found.")

        for index, audio_file in enumerate(audio_list):
            index_message = f"Audio File {index + 1} out of {file_count} ({audio_file}):"
            print(index_message, "Sampling audio...")
            samples = self.get_samples(audio_file)

            print(index_message, "Transcribing audio...")
            transcriptions = self.transcribe_samples(samples)
            
            for t, ts, channel in transcriptions:
                print(f"{t} {ts} {channel}")

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
