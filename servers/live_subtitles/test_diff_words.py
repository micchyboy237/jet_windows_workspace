import pytest

from diff_words import count_newly_appended_words


@pytest.mark.parametrize(
    "a, b, word_sep, expected",
    [
        
        # Basic append cases
        ("", "hello world", " ", 2),
        ("hello", "hello world", " ", 1),
        ("hello world", "hello world extra", " ", 1),
        ("a b c", "a b c d e f", " ", 3),
        
        # Changes in the middle + append at the end
        ("The quick brown", "The quick blue fox jumps", " ", 2),   # "fox", "jumps"
        ("Python is great", "Python is powerful and awesome", " ", 2),  # "and", "awesome"
        
        # No new words appended
        ("hello world", "hello world", " ", 0),
        ("hello world", "hello universe", " ", 0),
        ("hello world", "hello world!!!", " ", 0),
        
        # Edge cases (early changes ignored)
        ("", "", " ", 0),
        (" ", "word", " ", 1),
        ("word", " ", " ", 0),
        ("a b c d", "x y z a b c d e", " ", 1),   # only "e" is appended
        
        # Whitespace handling
        ("hello world", "hello world extra", " ", 1),
        
        # Japanese example (first word replaced + 1 appended)
        ("えあうめ楽しか", "えあうん楽しか 追加の単語です", " ", 1),
        
        # Custom separators
        ("apple,banana,cherry", "apple,banana,cherry,date", ",", 1),
        ("1|2|3", "1|2|3|4|5", "|", 2),
    ],
)
def test_count_newly_appended_words(a: str, b: str, word_sep: str, expected: int):
    assert count_newly_appended_words(a, b, word_sep) == expected


def test_complex_changes():
    """Heavy changes early, but new words appended at the end."""
    # Full rewrite (no exact word matches) + net 1 extra word appended
    assert count_newly_appended_words(
        "The quick brown fox jumps over the lazy dog",
        "A completely different sentence that ends with new words here"
    ) == 1

    assert count_newly_appended_words(
        "Old version of the document",
        "Totally rewritten content with many changes finally appended"
    ) == 3  # net 3 extra words appended after full rewrite


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
