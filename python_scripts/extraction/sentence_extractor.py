# sentence_extractor.py

import logging

import torch
from tqdm import tqdm
from wtpsplit import SaT

# ----------------------------------------------------------------------
# DEBUG LOGGING (toggle with `debug=True`)
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global cache for a single SaT model instance
_model_cache = {"model": None, "key": None}


def _get_model_key(
    model_name: str, style_or_domain: str | None, language: str
) -> tuple:
    return (model_name, style_or_domain, language)


def _load_model(model_name: str, style_or_domain: str | None, language: str) -> SaT:
    model_key = _get_model_key(model_name, style_or_domain, language)
    if _model_cache["key"] != model_key:
        try:
            if style_or_domain:
                sat = SaT(
                    model_name, style_or_domain=style_or_domain, language=language
                )
            else:
                sat = SaT(model_name)
            _model_cache["model"] = sat
            _model_cache["key"] = model_key
        except Exception as e:
            raise ValueError(
                f"Failed to load model '{model_name}' with style_or_domain '{style_or_domain}': {e}"
            )
    return _model_cache["model"]


def group_by_empty_split(segments: list[str]) -> list[list[str]]:
    paragraphs, current = [], []
    for seg in segments:
        if seg.strip():
            current.append(seg)
        else:
            if current:
                paragraphs.append(current)
                current = []
    if current:
        paragraphs.append(current)
    return paragraphs


# ----------------------------------------------------------------------
# Joining helpers
# ----------------------------------------------------------------------
def _flatten_list(lst: list) -> list[str]:
    result: list[str] = []
    for el in lst:
        if isinstance(el, str):
            result.append(el)
        elif isinstance(el, list):
            result.extend(_flatten_list(el))
    return result


def _join_paragraph(para: str | list[str]) -> str:
    if isinstance(para, str):
        return para.strip()
    flat = _flatten_list(para)
    return "\n".join(s.strip() for s in flat if s.strip())


def strip_trailing_whitespace_after_final_newline(text: str) -> str:
    """
    Strip trailing whitespace only if it comes after the final newline.
    Preserve all line-internal trailing whitespace and the final newline.
    """
    if not text:
        return text

    # Find the last newline
    last_newline_idx = text.rfind("\n")
    if last_newline_idx == -1:
        # No newline → nothing to strip at end
        return text

    # Split: content up to and including last '\n', and trailing part
    content = text[: last_newline_idx + 1]
    trailing = text[last_newline_idx + 1 :]

    # Only strip whitespace from the trailing part
    return content + trailing.rstrip(" \t")


# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def is_valid_sentence(text: str) -> bool:
    """
    Quick heuristic check whether a string should be considered a valid sentence.

    Returns True for things that look like proper sentences or meaningful fragments.
    Returns False for junk, headers, very short garbage, etc.
    """
    if not text:
        return False

    # Strip leading/trailing whitespace (but don't modify original for display)
    cleaned = text.strip()

    if not cleaned:
        return False

    # --- 1. Too short → usually noise ---
    # < 2 chars is almost never a sentence
    # < 5 chars → very rarely useful unless it's clearly a sentence
    if len(cleaned) <= 2:
        return False

    # Allow very short but clearly sentence-like fragments
    if len(cleaned) <= 6:
        # Examples we usually want to keep:
        # "Yes.", "No.", "Okay.", "Hi.", "A.M.", "Dr.", "etc."
        if cleaned.endswith((".", "!", "?")):
            return True
        # Otherwise very short → probably noise
        return False

    # --- 2. Very long continuous text without space → likely error / token garbage ---
    # (common when model fails to split)
    words = cleaned.split()
    if len(words) == 1 and len(cleaned) > 40:
        return False

    # --- 3. Looks like a heading / label / UI text / metadata ---
    # Common false positives from web text, subtitles, etc.
    if cleaned.isupper() and len(cleaned) < 60:  # ALL CAPS TITLES
        return False

    # Very common false positives
    if cleaned.lower() in {
        "abstract",
        "introduction",
        "conclusion",
        "references",
        "table of contents",
        "acknowledgements",
        "appendix",
        "chapter",
        "figure",
        "table",
        "section",
        "page",
        "isbn",
        "doi",
    }:
        return False

    # Starts with common list/heading patterns
    if cleaned.lstrip().startswith(("•", "○", "→", "▪", "*", "-", "–", "—")):
        if len(cleaned.split()) <= 5:
            return False

    # --- 4. Contains many special characters in a row → likely junk ---
    import re

    if re.search(r"[-_=]{4,}", cleaned):  # ----, ====, etc.
        return False
    if re.search(r"\W{6,}", cleaned):  # many non-word chars
        return False

    # --- 5. Has sentence-ending punctuation → strong signal ---
    if cleaned.rstrip().endswith((".", "!", "?", '."', '."', ".”", "…", '"', "'")):
        return True

    # --- 6. Has at least one space and some letters → good enough ---
    has_space = " " in cleaned
    has_letter = any(c.isalpha() for c in cleaned)

    if has_letter and has_space:
        return True

    # Default: be conservative
    return False


def extract_sentences(
    text: str | list[str],
    model_name: str = "sat-12l-sm",
    use_gpu: bool = False,
    do_paragraph_segmentation: bool = False,
    paragraph_threshold: float = 0.5,
    style_or_domain: str | None = None,
    language: str = "en",
    valid_only: bool = False,
    verbose: bool = False,
    debug: bool = False,  # ← NEW
) -> list[str]:
    """
    Extract sentences (or paragraphs) using SaT.
    """
    if debug:
        log.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Model loading & device
    # ------------------------------------------------------------------
    sat = _load_model(model_name, style_or_domain, language)

    device = "cpu"
    if use_gpu:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

    if sat.device != device:
        sat.to(device)
        if device == "cuda":
            sat.half()
    log.debug(f"Model on device: {device}")

    # ------------------------------------------------------------------
    # 2. Run SaT
    # ------------------------------------------------------------------
    raw_output = sat.split(
        text,
        do_paragraph_segmentation=do_paragraph_segmentation,
        paragraph_threshold=paragraph_threshold,
        verbose=verbose,
    )

    # Always materialize generator → list
    segmented = list(raw_output)
    log.debug(f"SaT output materialized: {len(segmented)} items")
    if debug:
        # Show first few items (avoid huge logs)
        preview = str(segmented)[:500]
        log.debug(f"SaT raw preview: {preview}")

    # ------------------------------------------------------------------
    # 3. Normalise SaT output → List[List[str]]
    # ------------------------------------------------------------------
    inner_segmented = _flatten_list(segmented)
    # Check each item if it contains newline characters at the end
    # If an item does, insert empty spaces (based on number of ending newlines) after it
    processed = []
    for s in inner_segmented:
        processed.append(s)
        if s.endswith("\n"):
            newline_count = len(s) - len(s.rstrip("\n"))
            processed.extend([""] * newline_count)
    inner_segmented = processed

    # For batched input texts
    if do_paragraph_segmentation and isinstance(text, list):
        inner_segmented = _flatten_list(inner_segmented)
        # Insert empty spaces in between items
        inner_segmented = [
            s
            for i, s in enumerate(inner_segmented)
            for s in ([s, ""] if i < len(inner_segmented) - 1 else [s])
        ]

    # Clean out trailing whitespaces after newline
    inner_segmented = [
        strip_trailing_whitespace_after_final_newline(s) for s in inner_segmented
    ]

    # ------------------------------------------------------------------
    # 4. Build final paragraphs
    # ------------------------------------------------------------------
    grouped = group_by_empty_split(inner_segmented)

    log.debug("Flattened mixed SaT output")
    paragraphs: list[str] = [_join_paragraph(group) for group in grouped]
    sentences = paragraphs
    log.debug(f"Before validation: {len(sentences)} sentences")

    # ------------------------------------------------------------------
    # 5. Optional validation
    # ------------------------------------------------------------------
    if valid_only:
        before = len(sentences)
        sentences = [
            s
            for s in tqdm(sentences, desc="Filtering valid sentences")
            if is_valid_sentence(s)
        ]
        log.debug(f"Validation filtered {before - len(sentences)} sentences")

    log.debug(f"Final output length: {len(sentences)}")
    return sentences


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from html_text_extractor import extract_html_text

    console = Console()

    def print_example_header(title: str):
        console.print(
            Panel.fit(
                Text(title, style="bold cyan"),
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    # Only the HTML example retained
    HTML_SAMPLE = """<div class="article-body">
    <h1>Understanding Sentence Segmentation</h1>
    <p>Modern NLP models can split text very accurately. <strong>SaT</strong> is one of the best open-source options in 2025.</p>
    <p>However, real-world content often contains:</p>
    <ul>
        <li>Headings and subheadings</li>
        <li>Lists and bullets</li>
        <li>Tables and captions</li>
        <li>References and footnotes</li>
    </ul>
    <p>Good sentence extractors should filter out most of these noise elements when <code>valid_only=True</code>.</p>
    <blockquote>
        "Clean segmentation makes downstream tasks much easier." — NLP researcher, 2024
    </blockquote>
    <p>That's why we love using wtpsplit + SaT!</p>
</div>
<footer>Last updated: Feb 2026</footer>"""

    html_text = extract_html_text(HTML_SAMPLE)

    print_example_header("HTML document example")
    console.print(
        "[yellow]Input contains typical article HTML structure[/yellow]\n"
    )

    console.print("[bold]Extracted sentences (valid_only=False):[/bold]")
    sentences = extract_sentences(
        html_text,
        model_name="sat-12l-sm",
        language="en",
        use_gpu=False,
        valid_only=False,
    )
    for i, s in enumerate(sentences, 1):
        console.print(f"[dim]{i:2d}.[/dim] {s}")

    # console.print("\n[bold]Valid sentences only:[/bold]")
    # valid_sentences = extract_sentences(
    #     html_text,
    #     model_name="sat-12l-sm",
    #     language="en",
    #     use_gpu=True,
    #     valid_only=True,
    # )
    # for i, s in enumerate(valid_sentences, 1):
    #     console.print(f"[dim]{i:2d}.[/dim] {s}")

    # console.print("\n[bold]With paragraph segmentation:[/bold]")
    # paragraphs = extract_sentences(
    #     html_text,
    #     model_name="sat-12l-sm",
    #     do_paragraph_segmentation=True,
    #     paragraph_threshold=0.5,
    #     use_gpu=True,
    #     valid_only=False,
    # )
    # for i, para in enumerate(paragraphs, 1):
    #     console.print(f"[bold cyan]Paragraph {i}:[/bold cyan]")
    #     console.print(para)
    #     console.print()

    console.print("\n[italic dim]Done.[/italic dim]")
