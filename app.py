from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from transcribe import AudioTranscriber


def create_transcriber():
    CONFIG_PATH = "/nemo_asr_root/transcribe.cfg"
    ASR_MODEL_PATH = "/nemo_asr_root/model/stt_en_conformer_ctc_xlarge.nemo"
    transcriber = AudioTranscriber(ASR_MODEL_PATH, CONFIG_PATH)
    return transcriber

def create_app():
    app = Flask(__name__)
    return app

app = create_app()

@app.route("/")
def index():
    return "Testing, Flask!"

#@app.route('/predict', methods=['POST'])
#def predict():
