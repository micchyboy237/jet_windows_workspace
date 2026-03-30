import argparse
from reazonspeech.k2.asr import load_model, transcribe, audio_from_path

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using reazonspeech ASR.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav",
        help="Path to the audio file (wav/mp3/etc.)."
    )
    args = parser.parse_args()

    # Load ReazonSpeech model from Hugging Face
    model = load_model(device="cuda", precision="fp32", language="ja") # or language="ja-en" for bilingual model or "ja-en-mls-5k" for 5k MLS bilingual model 
    
    audio = audio_from_path(args.audio_path)  # supports wav/mp3/etc. (auto-resamples)

    ret = transcribe(model, audio)

    print("Result")
    print(ret)

if __name__ == "__main__":
    main()
