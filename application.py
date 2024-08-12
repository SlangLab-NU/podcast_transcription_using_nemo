from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from transcribe import AudioTranscriber
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)


def create_transcriber():
    """
    Create an instance of AudioTranscriber
    """
    cwd = os.getcwd()
    CONFIG_PATH = os.path.join(cwd, 'transcribe.cfg')
    ASR_MODEL_PATH = os.path.join(
        cwd, "model/stt_en_conformer_ctc_xlarge.nemo")
    transcriber = AudioTranscriber(ASR_MODEL_PATH, CONFIG_PATH)
    return transcriber


def create_app():
    """
    Create a Flask app
    """
    app_obj = Flask(__name__)
    # app_obj = CORS(app_obj)
    return app_obj


app = create_app()
transciber = create_transcriber()


@app.route("/")
def index():
    """
    Test the Flask app. This is the default route.
    """
    logging.debug("GET request received")
    return "Testing, Flask!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the transcription of the audio file. The audio file is sent as a base64 encoded string.

    Sample request:
    {
    "config": {
        "sample_rate": 16000
        },
    "audio": {
        "content": "ZkxhQwAAACIQABAAAAUJABtAA+gA8AB+W8FZndQvQAyjv..."
        }
    }
    """
    logging.debug("POST request received")
    json_data = request.get_json()
    audio_encoded = json_data['audio']['content']
    sample_rate = json_data['config']['sample_rate']
    wav_data = base64.b64decode(audio_encoded)
    transcription = transciber.transcribe_api(wav_data, sample_rate)
    return jsonify(transcription)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
