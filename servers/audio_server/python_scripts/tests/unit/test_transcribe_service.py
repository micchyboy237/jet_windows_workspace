import pytest
from python_scripts.server.transcribe_service import get_transcriber
from .helpers.audio.whisper_ct2_transcriber import QuantizedModelSizes
def test_get_transcriber_returns_same_instance():
    t1 = get_transcriber("small","int8","cpu")
    t2 = get_transcriber("small","int8","cpu")
    assert t1 is t2
def test_get_transcriber_different_configs_different_instances():
    t1 = get_transcriber("base","int8","cpu")
    t2 = get_transcriber("base","int8_float16","cpu")
    assert t1 is not t2
