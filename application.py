from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from transcribe import AudioTranscriber
import os
import json


def create_transcriber():
    cwd = os.getcwd()
    CONFIG_PATH = os.path.join(cwd, 'transcribe.cfg')
    ASR_MODEL_PATH = os.path.join(cwd, "model/stt_en_conformer_ctc_xlarge.nemo")
    transcriber = AudioTranscriber(ASR_MODEL_PATH, CONFIG_PATH)
    return transcriber


def create_app():
    app_obj = Flask(__name__)
    # app_obj = CORS(app_obj)
    return app_obj


app = create_app()
transciber = create_transcriber()

@app.route("/")
def index():
    return "Testing, Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    audio_encoded = json_data['audio']['content']
    sample_rate = json_data['config']['sample_rate']
    wav_data = base64.b64decode(audio_encoded)
    output_audio_file = os.path.join(os.getcwd(), 'output.wav')
    with open(output_audio_file, 'wb') as wav_file:
        wav_file.write(wav_data)
    transcription = transciber.transcribe_api(output_audio_file, sample_rate)
    return jsonify(transcription)


if __name__ == '__main__':
    app.run()