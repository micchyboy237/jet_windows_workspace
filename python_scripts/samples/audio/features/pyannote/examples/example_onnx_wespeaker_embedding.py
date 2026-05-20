import argparse
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from pyannote.audio.pipelines.speaker_verification import ONNXWeSpeakerPretrainedSpeakerEmbedding


def load_audio(audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """Load audio from file path and resample if needed."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.unsqueeze(0)  # (1, 1, samples)


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix for all embeddings."""
    normed = F.normalize(embeddings, p=2, dim=1)  # (N, 256)
    return normed @ normed.T                       # (N, N)


def print_similarity_report(paths: list[str], sim_matrix: torch.Tensor, threshold: float = 0.75):
    n = len(paths)
    print("\n── Pairwise Cosine Similarity ──────────────────────────────")
    print(f"{'':>4}", end="")
    for i in range(n):
        print(f"  [{i}]  ", end="")
    print()

    for i in range(n):
        print(f"[{i}] ", end="")
        for j in range(n):
            print(f" {sim_matrix[i, j].item():.3f}", end="")
        print(f"  ← {paths[i]}")

    print("\n── Similar Speaker Pairs (cosine ≥ {:.2f}) ─────────────────".format(threshold))
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j].item()
            if score >= threshold:
                print(f"  [MATCH] {paths[i]}  ↔  {paths[j]}  (score: {score:.3f})")
                found = True
    if not found:
        print(f"  No pairs exceed the threshold of {threshold:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Speaker embedding and similarity tool.")
    parser.add_argument("audio_paths", nargs="+", help="One or more paths to audio files.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Cosine similarity threshold for matching (default: 0.75).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = ONNXWeSpeakerPretrainedSpeakerEmbedding(
        embedding="hbredin/wespeaker-voxceleb-resnet34-LM",
        device=device,
    )

    # ── Single file ───────────────────────────────────────────────────────────
    if len(args.audio_paths) == 1:
        print(f"Single file mode: {args.audio_paths[0]}")
        audio = load_audio(args.audio_paths[0])
        embedding = embedding_model(audio)
        print(f"Embedding shape : {embedding.shape}")
        print(f"Embedding vector: {embedding[0, :8].tolist()} ... (first 8 dims)")

    # ── Batch mode ────────────────────────────────────────────────────────────
    else:
        print(f"Batch mode: {len(args.audio_paths)} files")
        waveforms = [load_audio(p).squeeze(0) for p in args.audio_paths]  # list of (1, samples)

        max_len = max(w.shape[-1] for w in waveforms)
        padded, masks = [], []
        for w in waveforms:
            pad_len = max_len - w.shape[-1]
            padded.append(F.pad(w, (0, pad_len)))
            mask = torch.zeros(max_len)
            mask[: w.shape[-1]] = 1.0
            masks.append(mask)

        batch_audio = torch.stack(padded)   # (N, 1, max_len)
        batch_masks = torch.stack(masks)    # (N, max_len)

        embeddings = embedding_model(batch_audio, masks=batch_masks)
        print(f"Batch embeddings shape: {embeddings.shape}")

        # Convert to PyTorch tensor if it's a NumPy array
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()

        sim_matrix = cosine_similarity_matrix(embeddings)
        print_similarity_report(args.audio_paths, sim_matrix, threshold=args.threshold)


if __name__ == "__main__":
    main()