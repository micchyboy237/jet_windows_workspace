from __future__ import annotations

import pytest

from html_text_extractor import extract_all_text, extract_text_lines


@pytest.mark.parametrize(
    "html, expected",
    [
        pytest.param(
            """
            <html>
              <head><title>Test</title></head>
              <body>
                <h1>Main Title</h1>
                <p>First paragraph with <b>bold</b> text.</p>
                <p>Second paragraph.<br>With a line break.</p>
                <script>alert("bad")</script>
                <style>body {color:red}</style>
              </body>
            </html>
            """,
            "Main Title\nFirst paragraph with bold text.\nSecond paragraph. With a line break.",
            id="basic_structure",
        ),
        pytest.param(
            "<div><p>Hello</p>   <p>World</p></div>",
            "Hello\nWorld",
            id="multiple_paragraphs_collapse_whitespace",
        ),
        pytest.param(
            "<ul><li>One</li><li>Two</li></ul>",
            "One\nTwo",
            id="list_items_as_lines",
        ),
        pytest.param(
            "<div>Before<script>bad()</script>After</div>",
            "Before After",
            id="script_removed",
        ),
        pytest.param(
            "<p>  Lots   of    spaces   </p>",
            "Lots of spaces",
            id="internal_whitespace_normalized",
        ),
        pytest.param(
            "<article><h2>News</h2><p>Content here.</p></article>",
            "News\nContent here.",
            id="article_structure",
        ),
    ],
)
def test_extract_all_text(html: str, expected: str) -> None:
    # Given
    # When
    result = extract_all_text(html, separator=" ", strip=True)

    # Then
    assert result == expected


@pytest.mark.parametrize(
    "html, expected_lines",
    [
        pytest.param(
            """
            <h1>Title</h1>
            <p>Intro text</p>
            <ul>
              <li>Item A</li>
              <li>Item B</li>
            </ul>
            """,
            ["Title", "Intro text", "Item A", "Item B"],
            id="semantic_blocks",
        ),
        pytest.param(
            "<div><span>inline</span> <strong>also inline</strong></div>",
            ["inline also inline"],
            id="inline_elements_merged",
        ),
    ],
)
def test_extract_text_lines(html: str, expected_lines: list[str]) -> None:
    # Given
    # When
    result = extract_text_lines(html, min_length=1)

    # Then
    assert result == expected_lines


def test_empty_html() -> None:
    # Given
    html = ""

    # When
    result_all = extract_all_text(html)
    result_lines = extract_text_lines(html)

    # Then
    assert result_all == ""
    assert result_lines == []


def test_only_comments_and_scripts() -> None:
    # Given
    html = """
    <!-- secret -->
    <script>console.log("hi")</script>
    <style>body{}</style>
    """

    # When
    result = extract_all_text(html)

    # Then
    assert result == ""