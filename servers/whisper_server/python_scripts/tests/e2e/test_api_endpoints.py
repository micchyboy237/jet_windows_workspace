import pytest
def test_transcribe_endpoint(client, test_audio_sample):
    with open(test_audio_sample, "rb") as f:
        r = client.post("/transcribe?model_size=base^&compute_type=int8", files={"file": ("test.wav", f, "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["detected_language"] == "^<|en|^>"
    assert len(j["transcription"]) > 10

def test_translate_endpoint_returns_translation(client, test_audio_sample):
    with open(test_audio_sample, "rb") as f:
        r = client.post("/translate?model_size=base^&compute_type=int8", files={"file": ("test.wav", f, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["translation"] is not None
