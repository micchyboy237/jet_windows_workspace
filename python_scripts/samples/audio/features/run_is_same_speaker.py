from loader import load_audio
from pyannote.speaker_verification_utils import is_same_speaker


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
        help="Minimum confidence threshold for partial matches (default: 0.75)"
    )
    args = parser.parse_args()

    emb1 = load_audio(args.audio1)
    emb2 = load_audio(args.audio2)

    same_speaker = is_same_speaker(emb1, emb2, args.threshold)

    print(f"Same speaker: {same_speaker}")
