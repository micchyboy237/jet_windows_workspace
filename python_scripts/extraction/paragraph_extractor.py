import torch
from wtpsplit import SaT

def extract_paragraphs(text: str, model_name: str = "sat-12l-sm", style_or_domain: str = "ud", language: str = "en", use_gpu: bool = False, paragraph_threshold: float = 0.5) -> list[str]:
    """
    Extracts paragraphs from unstructured text without relying on newline delimiters.
    
    This function uses the SaT model from wtpsplit to perform semantic segmentation.
    It detects paragraph boundaries based on newline probability predictions,
    making it suitable for noisy or concatenated text (e.g., from PDFs or web scrapes).
    
    Args:
        text (str): The input text to segment.
        model_name (str, optional): The SaT model to use (e.g., "sat-12l-sm" for high accuracy,
                                    "sat-3l-sm" for faster inference). Defaults to "sat-12l-sm".
        use_gpu (bool, optional): Whether to use GPU if available. Defaults to False.
        paragraph_threshold (float, optional): Threshold for paragraph boundary detection
                                               (higher = more conservative). Defaults to 0.5.
    
    Returns:
        list[str]: A list of extracted paragraphs as strings.
    
    Raises:
        ValueError: If the model fails to load or text is empty.
    
    Example:
        >>> text = "This is the first paragraph. It has multiple sentences. This is the second paragraph without newlines."
        >>> extract_paragraphs(text)
        ['This is the first paragraph. It has multiple sentences. ', 'This is the second paragraph without newlines.']
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    
    # Load the model
    try:
        sat =  SaT(
            model_name, style_or_domain=style_or_domain, language=language
        )
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")
    
    # Move to GPU if available and requested
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        sat.half().to(device)
    
    # Perform segmentation with paragraph support
    segmented = sat.split(text, do_paragraph_segmentation=True, paragraph_threshold=paragraph_threshold)
    
    # Flatten paragraphs (each is a list of sentences) into strings
    paragraphs = [' '.join(sent.strip() for sent in para) for para in segmented]
    
    return paragraphs

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

    print("HTML document example")
    console.print(
        "[yellow]Input contains typical article HTML structure[/yellow]\n"
    )

    console.print("[bold]Extracted paragraphs:[/bold]")
    paragraphs = extract_paragraphs(
        html_text,
        model_name="sat-12l-sm",
        language="en",
        use_gpu=False,
        paragraph_threshold=0.7,
    )
    for i, s in enumerate(paragraphs, 1):
        console.print(f"[dim]{i:2d}.[/dim] {s}")