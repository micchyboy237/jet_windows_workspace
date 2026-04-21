#!/usr/bin/env python3
"""
Reusable function to add punctuation using sherpa-onnx OfflinePunctuation.

Supports both English and Chinese (and mixed) text.
"""

from pathlib import Path
from typing import List, Union, Optional

import sherpa_onnx


def create_punctuation_model(
    model_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> sherpa_onnx.OfflinePunctuation:
    """
    Create and return a sherpa-onnx OfflinePunctuation model.

    Args:
        model_dir: Directory containing the extracted punctuation model
                  (e.g., "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12")
        model_path: Direct path to model.onnx file (overrides model_dir if provided)

    Returns:
        An initialized OfflinePunctuation object

    Raises:
        ValueError: If the model file is not found
        FileNotFoundError: If model directory/path is invalid
    """
    if model_path is None:
        if model_dir is None:
            # Default location (customize this for your environment)
            base_dir = Path.home() / ".cache" / "pretrained_models" / "sherpa-onnx"
            model_dir = (
                base_dir / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
            )

        model_path = Path(model_dir) / "model.onnx"

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Punctuation model not found at: {model_path}\n"
            f"Please download it from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models"
        )

    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(
            ct_transformer=str(model_path)
        ),
    )

    return sherpa_onnx.OfflinePunctuation(config)


def add_punctuation(
    texts: Union[str, List[str]],
    punctuator: Optional[sherpa_onnx.OfflinePunctuation] = None,
    model_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    space_replacement: Optional[str] = None,
) -> Union[str, List[str]]:
    """
    Add punctuation to one or more text strings using sherpa-onnx.

    This is the main reusable function.

    Args:
        texts: A single string or a list of strings to punctuate
        punctuator: Optional pre-created OfflinePunctuation instance (for reuse)
        model_dir: Directory containing the punctuation model (if punctuator not provided)
        model_path: Direct path to model.onnx (if punctuator not provided)
        space_replacement: Optional character to replace spaces with in each
                           punctuated output string (e.g. "_" or "·")

    Returns:
        Punctuated text(s) with the same type as input:
        - str  → str
        - list → list

    Example:
        >>> punctuated = add_punctuation("hello how are you")
        >>> print(punctuated)
        "Hello, how are you?"

        >>> results = add_punctuation(["i love you", "what time is it"])
        >>> print(results)
        ["I love you.", "What time is it?"]

        >>> add_punctuation("see you soon", space_replacement="_")
        "See_you_soon."
    """
    # Create punctuator if not provided
    if punctuator is None:
        punctuator = create_punctuation_model(model_dir=model_dir, model_path=model_path)

    # Handle single string vs list
    was_single = isinstance(texts, str)
    if was_single:
        text_list: List[str] = [texts]
    else:
        text_list = texts  # type: ignore[assignment]

    # Add punctuation with optional space replacement
    punctuated_list = [
        punctuator.add_punctuation(text.replace(" ", space_replacement))
        if space_replacement is not None
        else punctuator.add_punctuation(text)
        for text in text_list
    ]

    # Return same type as input
    return punctuated_list[0] if was_single else punctuated_list


# ======================
# Example Usage
# ======================


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Add punctuation to English/Chinese/Japanese text via sherpa-onnx."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=["おはようございます 今日はいい天気ですね 何か予定がありますか"],
        help="One or more text strings to punctuate. For multi-word use quotes. (Ex.: 'hello world how are you today')",
    )
    parser.add_argument(
        "-s", "--space-replacement",
        default=None,
        help="Character to replace spaces with in punctuated output (e.g. '_' or '·').",
    )
    args = parser.parse_args()

    input_texts = args.texts

    # Reuse the punctuation model instance for all calls (better perf)
    punctuator = create_punctuation_model()

    if len(input_texts) == 1:
        punctuated = add_punctuation(input_texts[0], punctuator=punctuator, space_replacement=args.space_replacement)
        print(punctuated)
    else:
        results = add_punctuation(input_texts, punctuator=punctuator, space_replacement=args.space_replacement)
        for orig, punct in zip(input_texts, results):
            print(f"Input : {orig}")
            print(f"Output: {punct}")
            print("-" * 40)
