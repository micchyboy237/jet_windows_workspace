#!/usr/bin/env python3

"""
This script shows how to add punctuations to text using sherpa-onnx Python API.

Please download the model from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

The following is an example

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
"""

from pathlib import Path
import argparse

import sherpa_onnx

def main():
    parser = argparse.ArgumentParser(
        description="Add punctuation and case to texts using sherpa-onnx OnlinePunctuation."
    )
    parser.add_argument(
        "texts",
        nargs="+",
        help="Text(s) to punctuate. Example: 'text1' 'text2' ..."
    )
    args = parser.parse_args()

    base_dir = Path(r"C:\Users\druiv\.cache\pretrained_models\sherpa-onnx")
    model_dir = base_dir / "sherpa-onnx-online-punct-en-2024-08-06"
    
    model = model_dir / "model.onnx"
    bpe = model_dir / "bpe.vocab"

    if not model.is_file():
        raise ValueError(f"{model} does not exist. Please download the online model first.")
    if not bpe.is_file():
        raise ValueError(f"{bpe} does not exist.")

    model_config = sherpa_onnx.OnlinePunctuationModelConfig(
        cnn_bilstm=str(model), 
        bpe_vocab=str(bpe)
    )
    config = sherpa_onnx.OnlinePunctuationConfig(model_config=model_config)
    punct = sherpa_onnx.OnlinePunctuation(config)

    for text in args.texts:
        text_with_punct = punct.add_punctuation_with_case(text)
        print("----------")
        print(f"input : {text}")
        print(f"output: {text_with_punct}")
    print("----------")

if __name__ == "__main__":
    main()
