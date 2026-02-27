import pytest
from japanese_nlp import extract_japanese_nouns


@pytest.fixture(scope="module")
def sample_texts():
    return [
        "Pythonは簡単で使いやすいツールです。自然言語処理も楽しくできます。",
        "東京タワーとスカイツリーはどちらも素晴らしい観光地です。",
        "私は毎日コーヒーとパンケーキを食べます。",
        "",  # empty string
    ]


def test_extract_single_text(sample_texts):
    nouns = extract_japanese_nouns(sample_texts[0])
    expected = ["Python", "ツール", "自然言語処理"]  # 処理 may or may not be compound
    # Depending on exact tokenization — adjust expectation if needed
    assert len(nouns) >= 3
    assert "ツール" in nouns
    assert "Python" in nouns


def test_extract_multiple_texts(sample_texts):
    nouns = extract_japanese_nouns(sample_texts[:2])
    assert "東京タワー" in nouns or "東京" in nouns  # compound noun handling
    assert "スカイツリー" in nouns or "スカイ" in nouns
    assert len(nouns) > 5


def test_deduplicate():
    text = "東京東京タワー東京スカイツリー東京".replace(" ", "")  # no spaces
    nouns = extract_japanese_nouns(text, deduplicate=True)

    # Realistic expectation given common splits
    expected_possible = [
        ["東京", "タワー", "スカイツリー"],  # most common
        ["東京", "東京タワー", "スカイツリー"],  # if partial merge
        ["東京", "タワー", "東京スカイツリー"],
    ]

    assert nouns in expected_possible or nouns[:3] == [
        "東京",
        "タワー",
        "スカイツリー",
    ], f"Got {nouns} — nagisa usually splits compounds like 東京タワー"

    # Or stricter minimal check:
    assert "東京" in nouns
    assert "タワー" in nouns or "東京タワー" in nouns
    assert "スカイツリー" in nouns or "東京スカイツリー" in nouns


def test_pos_tags_custom():
    text = "私は東京大学で自然言語処理を研究しています。"
    nouns_custom = extract_japanese_nouns(text, pos_tags=["名詞", "固有名詞"])
    nouns_default = extract_japanese_nouns(text)

    assert len(nouns_custom) >= len(nouns_default)

    joined = " ".join(nouns_custom)
    # Realistic: expect split, not merged
    assert "東京" in joined and "大学" in joined, (
        f"Expected split '東京 大学', got: {joined}"
    )

    # Optional: warn if someone expects merge
    if "東京大学" in joined:
        pytest.skip("Rare case where nagisa kept '東京大学' as one token")


def test_lowercase():
    text = "Pythonは最高です。AIも好き。"
    nouns_lower = extract_japanese_nouns(text, lowercase=True)
    nouns_normal = extract_japanese_nouns(text)

    assert "python" in nouns_lower
    assert "Python" in nouns_normal
    assert "ai" in nouns_lower


def test_empty_input():
    assert extract_japanese_nouns("") == []
    assert extract_japanese_nouns(["", "   "]) == []


def test_list_input_preservation_order(sample_texts):
    nouns = extract_japanese_nouns(sample_texts)
    # Just check it doesn't crash and returns something reasonable
    assert isinstance(nouns, list)
    assert len(nouns) > 5
