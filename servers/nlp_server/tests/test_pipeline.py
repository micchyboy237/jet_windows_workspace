# tests/test_pipeline.py
import pytest
from app.pipeline import NLPProcessor

class TestNLPProcessor:
    def setup_method(self):
        self.processor = NLPProcessor()

    def test_annotate_single(self):
        # Given a single short text
        text = "Barack Obama was born in Hawaii."
        # When we annotate
        results = self.processor.annotate([text])
        # Then we get one result and expected structure
        assert isinstance(results, list)
        assert len(results) == 1
        res = results[0]
        assert "sentences" in res
        assert len(res["sentences"]) >= 1
        # Check token words structure
        sent0 = res["sentences"][0]
        assert "tokens" in sent0
        assert "words" in sent0
        first_word = sent0["words"][0]
        assert "text" in first_word and "lemma" in first_word and "pos" in first_word

    def test_annotate_empty_list(self):
        # Given an empty list
        with pytest.raises(IndexError):
            _ = self.processor.annotate([])  # expecting maybe internal error or empty result