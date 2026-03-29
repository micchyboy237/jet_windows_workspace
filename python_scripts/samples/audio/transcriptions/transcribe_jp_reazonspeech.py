# Install once:
# Follow https://github.com/reazon-research/ReazonSpeech for sherpa-onnx

# from reazonspeech.k2.asr import load_model, transcribe, audio_from_path
# from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import argparse
from reazonspeech.espnet.asr import load_model, transcribe, audio_from_path

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using reazonspeech ASR.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav",
        help="Path to the audio file (wav/mp3/etc.)."
    )
    args = parser.parse_args()

    audio = audio_from_path(args.audio_path)  # supports wav/mp3/etc. (auto-resamples)
    model = load_model()  # loads reazonspeech-k2-v2 by default
    ret = transcribe(model, audio)

    print("Result")
    print(ret)

if __name__ == "__main__":
    main()
