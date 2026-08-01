import argparse
from reazonspeech.espnet.asr import load_model, transcribe_live_stream_jp, audio_from_path

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using reazonspeech ASR.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav",
        help="Path to the audio file (wav/mp3/etc.)."
    )
    args = parser.parse_args()
    
    audio = audio_from_path(args.audio_path)
    model = load_model()
    
    print("Starting streaming transcription...")
    for segment in transcribe_live_stream_jp(model, audio):
        print(segment)
    
    print("Transcription complete!")

if __name__ == "__main__":
    main()