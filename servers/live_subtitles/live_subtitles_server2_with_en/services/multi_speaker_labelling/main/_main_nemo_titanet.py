from nemo_titanet import detect_multi_speakers

def get_args():
    import argparse

    DEFAULT_AUDIO = r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav"
    parser = argparse.ArgumentParser(
        description="Automatic speaker labeling with NeMo TitaNet-Large embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_path", type=str, nargs="?", default=DEFAULT_AUDIO,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--model-name", type=str, default="titanet_large",
        help="NeMo pretrained speaker embedding model name"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=2.0,
        help="Window duration in seconds for embedding extraction"
    )
    parser.add_argument(
        "-s", "--step", type=float, default=0.75,
        help="Window step in seconds for sliding window"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=16,
        help="Number of windows embedded per forward pass (lower this if you hit GPU out-of-memory)"
    )
    parser.add_argument(
        "-e", "--min-energy-percentile", type=float, default=15.0,
        help="Skip the quietest N%% of windows (silence/pauses) before embedding, relative to this file's own energy range. 0 disables."
    )
    parser.add_argument(
        "-m", "--min-segment-duration", type=float, default=1.0,
        help="Minimum duration in seconds for a speaker segment to be included"
    )
    parser.add_argument(
        "-c", "--clustering-method", type=str, choices=["agglomerative", "spectral"],
        default="agglomerative", help="Clustering method to use for speaker grouping"
    )
    parser.add_argument(
        "-t", "--merge-threshold", type=float, default=0.55,
        help="Similarity threshold for merging speaker clusters (0.45-0.65 recommended for TitaNet on real conversational audio, lower = more aggressive merging)"
    )
    parser.add_argument(
        "-a", "--assign-threshold", type=float, default=0.55,
        help="Minimum similarity threshold for assigning a frame to a speaker (recalibrated from real-audio testing, NVIDIA's clean-benchmark default of 0.70 was too strict)"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    results = detect_multi_speakers(
        audio_path=args.audio_path,
        model_name=args.model_name,
        duration=args.duration,
        step=args.step,
        batch_size=args.batch_size,
        min_energy_percentile=args.min_energy_percentile,
        min_segment_duration=args.min_segment_duration,
        method=args.clustering_method,
        merge_threshold=args.merge_threshold,
        assign_threshold=args.assign_threshold
    )

if __name__ == "__main__":
    main()
