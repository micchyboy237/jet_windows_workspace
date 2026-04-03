#!/usr/bin/env python3

"""
This script shows how to add punctuations to text using sherpa-onnx Python API.

Please download the model from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

The following is an example

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
"""

from pathlib import Path
import argparse

import sherpa_onnx

def main():
    parser = argparse.ArgumentParser(
        description="Add punctuation to a list of texts using sherpa-onnx OfflinePunctuation."
    )
    parser.add_argument(
        "text_list",
        nargs="+",
        help="List of texts to punctuate. Example: 'text1' 'text2' ...",
    )
    args = parser.parse_args()

    base_dir = Path(r"C:\Users\druiv\.cache\pretrained_models\sherpa-onnx")
    model = base_dir / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12" / "model.onnx"

    if not model.is_file():
        raise ValueError(f"{model} does not exist")

    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=str(model)),
    )

    punct = sherpa_onnx.OfflinePunctuation(config)

    for text in args.text_list:
        text_with_punct = punct.add_punctuation(text)
        print("----------")
        print(f"input: {text}")
        print(f"output: {text_with_punct}")

    print("----------")

if __name__ == "__main__":
    main()
