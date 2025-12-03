import pytest, shutil
from pathlib import Path
from fastapi.testclient import TestClient
from python_scripts.server.main import app
@pytest.fixture(scope="session")
def test_audio_sample():
    p = Path("samples/audio/data/short_english.wav")
    if not p.exists(): pytest.skip("Add short_english.wav")
    return str(p)
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    shutil.rmtree("temp_uploads", ignore_errors=True)
