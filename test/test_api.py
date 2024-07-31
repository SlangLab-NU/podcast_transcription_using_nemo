import pytest
from app import create_app


@pytest.fixture(scope='session')
def app_obj():
    app_context=create_app()
    yield app_context
def test_index_route(app_obj):
    response = app_obj.test_client().get('/')
    print(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'Testing, Flask!'