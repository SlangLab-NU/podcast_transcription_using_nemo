import pytest
from application import app
from application import create_transcriber
import base64, json
import os, io
import soundfile as sf

@pytest.fixture(scope="module")
def audio_buffer():
    cwd = os.getcwd()
    input_audio_file = os.path.join(cwd, 'input.wav')
    with open(input_audio_file, 'rb') as wav_file:
        wav_data = wav_file.read()
    base64_encoded = base64.b64encode(wav_data).decode('utf-8')
    #wav_data = base64.b64decode(base64_encoded)
    #waveform, samplerate = sf.read(io.BytesIO(wav_data))
    yield base64_encoded


@pytest.fixture(scope="module")
def request_data(audio_buffer):
    request_dict = {
        'config': {
          'sample_rate': 16000
        },
        'audio': {
            'content': audio_buffer
        }
    }
    yield request_dict


def test_load_transcriber():
    transcriber = create_transcriber()
    assert transcriber is not None

@pytest.fixture(scope="module")
def testing_client():
    yield app.test_client()


def test_predict_route(testing_client, request_data):
    request_json = json.dumps(request_data)
    response = testing_client.post('/predict', json=request_data)
    print(response.get_data(as_text=True))
    assert response


def test_index_route(testing_client):
    response = testing_client.get('/')
    print(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'Testing, Flask!'