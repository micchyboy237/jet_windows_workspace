# vad_speaker_labeler.py

import json
from collections import defaultdict
from pathlib import Path
from typing import List, TypedDict

import matplotlib
import numpy as np
import torch
from pyannote.audio import Inference, Model
from rich.console import Console
from vad_firered2 import extract_speech_audio, extract_speech_timestamps, save_segments

matplotlib.use("Agg")

console = Console()

SAVE_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)


# ---------------------------------------------------------------------------
# Speaker Embedding Utilities
# ---------------------------------------------------------------------------


class SpeakerGroup(TypedDict):
    speaker_label: str
    segments: List[int]


def build_embedding_inference(device: str | None = None) -> Inference:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model.from_pretrained("pyannote/embedding")
    model.to(torch.device(device))

    return Inference(model, window="whole")


def compute_embedding_from_chunk(
    inference: Inference,
    chunk: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Convert audio chunk → embedding (1, D)
    """
    waveform = torch.from_numpy(chunk).unsqueeze(0)

    emb = inference(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
    )

    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()

    if emb.ndim == 1:
        emb = emb[None, :]

    return emb


def cosine_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute NxN cosine similarity matrix.
    """
    X = np.vstack([e[0] for e in embeddings])  # (N, D)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-10)

    return np.dot(X_norm, X_norm.T)


def cluster_speakers(
    sim_matrix: np.ndarray,
    threshold: float = 0.72,
) -> List[str]:
    """
    Assign speaker labels using greedy similarity clustering.

    Parameters
    ----------
    sim_matrix:
        NxN cosine similarity matrix.

    threshold:
        Similarity threshold for assigning the same speaker label.

    Returns
    -------
    List[str]
        Speaker label for each segment.
    """
    num_segments = sim_matrix.shape[0]

    labels: List[str] = []
    speaker_centers: List[int] = []

    for idx in range(num_segments):
        assigned = False

        for speaker_idx, ref_idx in enumerate(speaker_centers):
            similarity = float(sim_matrix[idx, ref_idx])

            if similarity >= threshold:
                labels.append(f"SPEAKER_{speaker_idx + 1:02d}")
                assigned = True
                break

        if not assigned:
            speaker_centers.append(idx)
            labels.append(f"SPEAKER_{len(speaker_centers):02d}")

    return labels


if __name__ == "__main__":
    import argparse
    import shutil

    DEFAULT_AUDIO = (
        r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
    )
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    args = parser.parse_args()

    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")

    # ── Step 1: detect segments ───────────────────────────────────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        max_speech_duration_sec=8.0,
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")

    if not any(s["type"] == "speech" for s in segments):
        console.print("[red]No speech segments found.[/red]")
        raise SystemExit(0)

    # ── Step 2: extract speech audio ──────────────────────────────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=16000,
        max_speech_duration_sec=8.0,
    )

    # ── Step 2.5: Speaker Comparison ──────────────────────────────────────
    console.rule("Speaker Comparison", style="magenta")

    inference = build_embedding_inference()
    embeddings: List[np.ndarray] = []
    compared_segment_numbers: List[int] = []

    with console.status("[bold green]Computing embeddings..."):
        for idx, chunk in enumerate(audio_chunks):
            segment_number = idx + 1

            # Skip very short segments (unstable embeddings)
            if len(chunk) < 16000 * 0.5:
                console.print(
                    f"[yellow]Skipping segment "
                    f"{segment_number:03d} "
                    f"(duration < 0.5s)[/yellow]"
                )
                continue

            emb = compute_embedding_from_chunk(inference, chunk)
            embeddings.append(emb)
            compared_segment_numbers.append(segment_number)

    if len(embeddings) < 2:
        console.print("[yellow]Not enough segments for comparison.[/yellow]")
    else:
        sim_matrix = cosine_similarity_matrix(embeddings)
        speaker_labels = cluster_speakers(sim_matrix)

        # ── Speaker groups ───────────────────────────────────────────────
        console.rule("Speaker Groups", style="green")

        groups: dict[str, list[int]] = defaultdict(list)

        for idx, label in enumerate(speaker_labels):
            segment_number = compared_segment_numbers[idx]
            groups[label].append(segment_number)

        for label, segs in groups.items():
            segs_text = ", ".join(f"{seg:03d}" for seg in segs)

            console.print(f"[bold cyan]{label}[/bold cyan]")
            console.print(f"  Segments: [bold white]{segs_text}[/bold white]\n")

        # ── Top similarities ─────────────────────────────────────────────
        console.rule("Top Similarities", style="magenta")

        similarity_threshold = 0.72

        for i in range(len(sim_matrix)):
            current_segment = compared_segment_numbers[i]
            pairs: List[tuple[int, float]] = []

            for j in range(len(sim_matrix)):
                if i == j:
                    continue

                similarity = float(sim_matrix[i, j])

                if similarity >= similarity_threshold:
                    pairs.append(
                        (
                            compared_segment_numbers[j],
                            similarity,
                        )
                    )

            pairs.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            if not pairs:
                continue

            formatted_pairs = ", ".join(
                f"{seg:03d} ({score:.2f})" for seg, score in pairs[:5]
            )

            console.print(
                f"[bold yellow]Segment "
                f"{current_segment:03d}[/bold yellow] "
                f"→ {formatted_pairs}"
            )

        # Save similarity
        similarity_path = output_dir / "speaker_similarity.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(similarity_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_segments": len(embeddings),
                    "segment_numbers": compared_segment_numbers,
                    "similarity_matrix": sim_matrix.tolist(),
                },
                f,
                indent=2,
            )

        console.print(
            f"[bold green]✓ Similarity saved:[/bold green] "
            f"[link=file://{similarity_path.resolve()}]{similarity_path}[/link]"
        )

        # ── Save readable speaker groups ────────────────────────────────
        speaker_groups_path = output_dir / "speaker_groups.json"

        speaker_groups: List[SpeakerGroup] = []

        for label, segs in groups.items():
            speaker_groups.append(
                {
                    "speaker_label": label,
                    "segments": segs,
                }
            )

        with open(speaker_groups_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": similarity_threshold,
                    "groups": speaker_groups,
                },
                f,
                indent=2,
            )

        console.print(
            f"[bold green]✓ Speaker groups saved:[/bold green] "
            f"[link=file://{speaker_groups_path.resolve()}]"
            f"{speaker_groups_path}"
            f"[/link]"
        )

    # ── Step 3: save segments ─────────────────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # ── Step 4: summary files ─────────────────────────────────────────────
    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        slim = [
            {k: v for k, v in m.items() if k != "segment_probs"} for m in saved_metas
        ]
        json.dump(slim, fh, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]✓ Summary saved:[/bold green] "
        f"[link=file://{summary_path.resolve()}]{summary_path}[/link]"
    )

    console.rule("Done", style="green")
