import pytest
from application import app
from application import create_transcriber
import base64, json
import os


@pytest.fixture(scope="module")
def audio_buffer():
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
    response = testing_client.post('/predict', json=request_data)
    print(response.get_data(as_text=True))
    assert response


def test_index_route(testing_client):
    response = testing_client.get('/')
    print(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'Testing, Flask!'