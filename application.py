from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from transcribe import AudioTranscriber
import os


def create_transcriber():
    cwd = os.getcwd()
    CONFIG_PATH = os.path.join(cwd, 'transcribe.cfg')
    ASR_MODEL_PATH = os.path.join(cwd, "model/stt_en_conformer_ctc_xlarge.nemo")
    transcriber = AudioTranscriber(ASR_MODEL_PATH, CONFIG_PATH)
    return transcriber


app = Flask(__name__)
transciber = create_transcriber()

@app.route("/")
def index():
    return "Testing, Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = '/Users/aanchan/work/podcast_transcription_using_nemo/test/input.wav'
    transcription = transciber.transcribe_api(audio_file)
    return jsonify(transcription)

