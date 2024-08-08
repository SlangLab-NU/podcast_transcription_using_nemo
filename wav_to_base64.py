import base64
import os
import argparse


def convert_wav_to_base64(input_audio_file):
    """
    Convert the input audio file to base64 format.

    Args:
        input_audio_file: The path to the input audio file.

    Raises:
        FileNotFoundError: The input audio file was not found.

    Returns:
        base64_encoded: The base64 encoded audio file.
    """
    # Read the audio file.
    cwd = os.getcwd()
    input_audio_file = os.path.join(cwd, input_audio_file)

    if not os.path.exists(input_audio_file):
        raise FileNotFoundError(f"File not found: {input_audio_file}")
    with open(input_audio_file, 'rb') as wav_file:
        wav_data = wav_file.read()
        base64_encoded = base64.b64encode(wav_data).decode('utf-8')
        return base64_encoded


def save_base64_to_text(base64_encoded, output_text_file):
    """
    Save the base64 encoded audio file to a text file.

    Args:
        base64_encoded: The base64 encoded audio file.
        output_text_file: The path to the output text file.

    Raises:
        Exception: An error occurred saving the base64 encoded audio file.
    """
    try:
        with open(output_text_file, 'w') as text_file:
            text_file.write(base64_encoded)
            print(f"Base64 encoded audio file saved to {output_text_file}")
    except Exception as e:
        print(
            f"Error saving base64 encoded audio file to {output_text_file}: {e}")
        raise e


if __name__ == '__main__':
    # Parse the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('input_audio_file', type=str,
                        help='The path to the input audio file.')
    args = parser.parse_args()
    input_audio_file = args.input_audio_file

    # Convert the audio file to base64.
    base64_encoded = convert_wav_to_base64(input_audio_file)
    save_base64_to_text(base64_encoded, 'base64_encoded_audio.txt')
