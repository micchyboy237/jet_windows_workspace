# audio_tagger.py

"""
This script shows how to use audio tagging Python APIs to tag a file.

Please read the code to download the required model files and test wave file.
"""

import logging
import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
AUDIO_TAGGING_MODEL = BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
CLASS_LABELS_INDICES_CSV = BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
TEST_WAVE = BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/1.wav"


def read_test_wave(test_wave):
    # Please download the model files and test wave files from
    # https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models

    if not Path(test_wave).is_file():
        raise ValueError(
            f"Please download {test_wave} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    # See https://python-soundfile.readthedocs.io/en/0.11.0/#soundfile.read
    data, sample_rate = sf.read(
        test_wave,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)

    # samples is a 1-d array of dtype float32
    # sample_rate is a scalar
    return samples, sample_rate


def create_audio_tagger():
    # Please download the model files and test wave files from
    # https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
    model_file = AUDIO_TAGGING_MODEL
    label_file = CLASS_LABELS_INDICES_CSV

    if not Path(model_file).is_file():
        raise ValueError(
            f"Please download {model_file} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    if not Path(label_file).is_file():
        raise ValueError(
            f"Please download {label_file} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    # NOTE: str() is required — sherpa_onnx is a C++ pybind11 extension that
    # does strict type checking and will not accept pathlib.Path objects.
    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                model=str(model_file),
            ),
            num_threads=1,
            debug=True,
            provider="cpu",
        ),
        labels=str(label_file),
        top_k=5,
    )

    if not config.validate():
        raise ValueError(f"Please check the config: {config}")

    print(config)

    return sherpa_onnx.AudioTagging(config)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run audio tagging on a provided audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        default=str(TEST_WAVE),
        help=f"Path to input wave file. Defaults to {str(TEST_WAVE)!r}"
    )
    args = parser.parse_args()

    logging.info("Create audio tagger")
    audio_tagger = create_audio_tagger()

    logging.info(f"Read test wave from: {args.audio_path}")
    samples, sample_rate = read_test_wave(args.audio_path)

    logging.info("Computing")

    start_time = time.time()

    stream = audio_tagger.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    result = audio_tagger.compute(stream)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    s = "\n"
    for i, e in enumerate(result):
        s += f"{i}: {e}\n"
    logging.info(s)

    # Save results to OUTPUT_DIR/results.json
    output_path = OUTPUT_DIR / "results.json"

    def to_dict(event):
        return {
            "index": None,  # filled below
            "name": getattr(event, "name", None),
            "class_index": getattr(event, "index", None),
            "prob": getattr(event, "prob", None),
        }

    structured_result = []
    for i, e in enumerate(result):
        item = to_dict(e)
        item["index"] = i
        structured_result.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved results to: {output_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
