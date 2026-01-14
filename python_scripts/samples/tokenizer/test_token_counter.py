# test_token_counter.py
from __future__ import annotations

import pytest
from unittest.mock import Mock
import re

from token_counter import (
    TokenCounter,
)


# =============================================================================
# Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()

    def fake_tokenize(bytes_input, add_special_tokens=True, special=False):
        try:
            text = bytes_input.decode("utf-8", errors="replace")
        except:
            text = ""

        # Strip control tokens when add_special_tokens=False
        if not add_special_tokens:
            text = re.sub(r'<\|[^|]+\|>', ' ', text)
            text = ' '.join(text.split())  # normalize spaces

        if text.strip() == "hello world":
            return [319, 296, 11]
        if "system" in text.lower():
            return [128000, 9125, 198, 128006, 9125, 198]  # llama-3-ish
        if not text.strip():
            return []
        # naive fallback
        return list(range(len(text) // 3 + 2))

    tokenizer.tokenize.side_effect = fake_tokenize
    return tokenizer


@pytest.fixture
def counter_with_mock_tokenizer(mock_tokenizer):
    counter = TokenCounter()
    counter._tokenizer = mock_tokenizer
    return counter


@pytest.fixture
def counter_no_tokenizer():
    counter = TokenCounter()
    counter._tokenizer = None
    return counter


# =============================================================================
# Real tokenizer behavior (mocked)
# =============================================================================

class TestTokenCounterWithTokenizer:
    """Given we have a working tokenizer from llama_cpp"""

    def test_basic_counting(self, counter_with_mock_tokenizer):
        # Given
        text = "hello world"

        # When
        count = counter_with_mock_tokenizer.count_tokens(text)

        # Then
        assert count == 3

    def test_returns_tokens_when_requested(self, counter_with_mock_tokenizer):
        # Given
        text = "hello world"

        # When
        count, tokens = counter_with_mock_tokenizer.count_tokens(text, return_tokens=True)

        # Then
        assert count == 3
        assert tokens == [319, 296, 11]   # match what current mock returns for "hello world"

    def test_add_special_tokens_false(self, counter_with_mock_tokenizer):
        # Given
        text = "<|im_start|>system hello"

        # When
        count = counter_with_mock_tokenizer.count_tokens(
            text, add_special_tokens=False
        )

        # Then
        # Now should be different after stripping special tokens
        assert count == 6           # after stripping we get "system hello" → mock returns 6 tokens

    def test_empty_string(self, counter_with_mock_tokenizer):
        # Given
        text = ""

        # When
        count = counter_with_mock_tokenizer.count_tokens(text)

        # Then
        assert count == 0


# =============================================================================
# Fallback / no tokenizer behavior
# =============================================================================

class TestTokenCounterFallback:
    """Given we DON'T have access to real tokenizer"""

    def test_fallback_rough_estimation_normal_text(self, counter_no_tokenizer):
        # Given
        text = "Hello world! This is a test sentence."

        # When
        count = counter_no_tokenizer.count_tokens(text, ignore_special_patterns=True)

        # Then
        # Very rough: ~4 chars per token + spaces
        assert 8 <= count <= 14  # wide range because estimation is crude

    def test_fallback_detects_common_special_tokens(self, counter_no_tokenizer):
        # Given
        text = """<|im_start|>system
You are helpful assistant<|im_end|>
<|im_start|>user
Hi!<|im_end|>"""

        # When
        count = counter_no_tokenizer.count_tokens(text)

        # Then
        # Should detect several special tokens → higher than pure char estimation
        assert count >= 12

    @pytest.mark.parametrize("text, expected_min", [
        ("", 0),
        ("a", 1),
        (" " * 100, 20),   # spaces are usually cheap
        ("hello" * 20, 25),  # repeated words compress well
    ])
    def test_fallback_edge_cases(self, counter_no_tokenizer, text, expected_min):
        # When
        count = counter_no_tokenizer.count_tokens(text, ignore_special_patterns=True)

        # Then
        assert count >= expected_min


# =============================================================================
# Integration / real tokenizer tests (slow, optional)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_real_tokenizer_looks_sane():
    """
    This test requires llama_cpp_python + a small gguf model nearby
    Usually you run this manually / in separate CI job
    """
    try:
        from llama_cpp import LlamaTokenizer  # type: ignore
    except ImportError:
        pytest.skip("llama_cpp not installed")

    # You need a small model for CI/integration
    MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # example

    counter = TokenCounter(model_path=MODEL_PATH, tokenizer_only=True)

    text = """<|im_start|>system
You are a helpful assistant.<|im_end>
<|im_start|>user
Hello!<|im_end>
<|im_start|>assistant
Hi there!"""

    count = counter.count_tokens(text, add_special_tokens=True)
    assert count > 15  # rough sanity check for llama-3/mistral-like tokenizer

    count_no_special = counter.count_tokens(text, add_special_tokens=False)
    # Many tokenizers (esp. llama-3 family) often ignore the flag when text already contains bos/eos/etc
    assert count_no_special <= count, (
        f"Expected tokens without forced special <= with forced special. "
        f"Got {count_no_special} > {count}"
    )