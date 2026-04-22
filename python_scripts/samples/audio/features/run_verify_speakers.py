from loader import load_audio
from pyannote.speaker_verification_utils import load_embedding_model, load_audio_tensor, verify_speakers


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Cluster speech segments and compare speaker similarity."
    )
    parser.add_argument("audio1", type=Path, help="Path to first speaker audio file (WAV)")
    parser.add_argument("audio2", type=Path, help="Path to second speaker audio file (WAV)")
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=0.7,
        help="Minimum confidence threshold for partial matches (default: 0.7)"
    )
    args = parser.parse_args()

    model = load_embedding_model()
    wav1, _ = load_audio_tensor(args.audio1, target_sample_rate=model.sample_rate)
    wav2, _ = load_audio_tensor(args.audio2, target_sample_rate=model.sample_rate)
    same, dist = verify_speakers(model, wav1, wav2, threshold=args.threshold)
    print(f"Same: {same}, Distance: {dist:.4f}")
