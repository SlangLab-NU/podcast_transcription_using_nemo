import pytest
from application import app
from application import create_transcriber
import base64, soundfile as sf
import os


@pytest.fixture(scope="module")
def create_audio_buffer():
    cwd = os.getcwd()
    input_audio_file = os.path.join(cwd, 'input.wav')
    output_audio_file = os.path.join(cwd, 'output.wav')
    with open(input_audio_file, 'rb') as wav_file:
        wav_data = wav_file.read()
    base64_encoded = base64.b64encode(wav_data).decode('utf-8')
    yield base64_encoded
    #wav_data = base64.b64decode(base64_encoded)
    #with open(output_audio_file, 'wb') as wav_file:
    #    wav_file.write(wav_data)
    #assert os.path.isfile(output_audio_file)
def test_load_transcriber():
    transcriber = create_transcriber()
    assert transcriber is not None

def test_predict_route():
    client = app.test_client()
    response = client.post('/predict')
    print(response.get_data(as_text=True))
    assert response
def test_index_route():
    client = app.test_client()
    response = client.get('/')
    print(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'Testing, Flask!'