import torch
from wtpsplit import SaT

def extract_paragraphs(text: str, model_name: str = "sat-12l-sm", use_gpu: bool = True, paragraph_threshold: float = 0.5) -> list[str]:
    """
    Extracts paragraphs from unstructured text without relying on newline delimiters.
    
    This function uses the SaT model from wtpsplit to perform semantic segmentation.
    It detects paragraph boundaries based on newline probability predictions,
    making it suitable for noisy or concatenated text (e.g., from PDFs or web scrapes).
    
    Args:
        text (str): The input text to segment.
        model_name (str, optional): The SaT model to use (e.g., "sat-12l-sm" for high accuracy,
                                    "sat-3l-sm" for faster inference). Defaults to "sat-12l-sm".
        use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
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
        sat = SaT(model_name)
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