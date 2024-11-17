import ffmpeg
import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Union

def convert_mp3_to_wav(input_data: Union[str, bytes]) -> BytesIO:
    """
    Convert MP3 file to WAV format.

    Args:
        input_data: Path to the MP3 file or bytes data of the MP3 file.

    Returns:
        BytesIO object containing WAV data.
    """
    if isinstance(input_data, str):
        # input_data is a file path
        input_data = open(input_data, 'rb').read()

    input_buffer = BytesIO(input_data)
    output_buffer = BytesIO()

    process = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    wav_data, _ = process.communicate(input=input_buffer.read())
    output_buffer.write(wav_data)
    output_buffer.seek(0)

    return output_buffer