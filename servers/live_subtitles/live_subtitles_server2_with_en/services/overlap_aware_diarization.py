"""
Overlap-Aware Speaker Diarization with ECAPA-TDNN
==================================================

Full robust pipeline integrating:
  • ECAPA-TDNN (SpeechBrain) — speaker embeddings
  • Pyannote OSD              — overlap speech detection
  • SepFormer (SpeechBrain)  — speech separation (optional)
  • Adaptive cosine thresholds by acoustic condition
  • Three overlap strategies  — nn | resegment | separate
  • RTTM export               — standard diarization output format
  • Confidence scoring        — per-turn similarity scores
  • Robust turn-building      — median-filter + min-duration merge

Install:
    pip install speechbrain pyannote.audio librosa torch scikit-learn scipy

Usage:
    python overlap_aware_diarization.py meeting.wav --strategy resegment --condition noisy
    python overlap_aware_diarization.py call.wav --strategy separate --condition phone --speakers 2
    python overlap_aware_diarization.py studio.wav --strategy nn --condition clean --rttm out.rttm
"""

import torch
import librosa
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.signal import medfilt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diarize")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """A single speaker turn, possibly overlapping with another."""
    start:    float
    end:      float
    speaker:  str
    score:    float = 0.0        # cosine similarity to centroid
    label:    str   = "speech"   # "speech" | "overlap" | "uncertain"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self):
        return (f"Turn({self.speaker}  {self.start:.2f}s→{self.end:.2f}s  "
                f"sim={self.score:.3f}  [{self.label}])")


@dataclass
class DiarizationResult:
    """Container for the full pipeline output."""
    turns:       List[Turn]
    n_speakers:  int
    audio_path:  str
    strategy:    str
    condition:   str
    thresholds:  Dict[str, float] = field(default_factory=dict)

    def clean_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "speech"]

    def overlap_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "overlap"]

    def uncertain_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "uncertain"]


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — ADAPTIVE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════

# Cosine similarity thresholds for spkrec-ecapa-voxceleb
# Sources:
#   • HuggingFace model card community: 0.70–0.75 default
#   • EEG voice signature study: same-speaker > 0.90, different < 0.65
#   • DiCLET-TTS ECAPA study: upper bound ~0.80, lower bound ~0.18
#   • DKU-DukeECE VoxSRC-2022: OSD gate = 0.85
#   • Original ECAPA paper EER on VoxCeleb1: 0.69–0.87%

THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Clean studio / controlled recording (podcast, read speech)
    "clean": {
        "same":          0.75,   # > this  → definite same speaker
        "ambiguous_low": 0.65,   # < this  → definite different speaker
        "osd_gate":      0.85,   # overlap detection confidence gate
        "hard_accept":   0.90,   # > this  → unconditional same
        "hard_reject":   0.25,   # < this  → unconditional different
    },
    # Meeting room, mild background noise, reverberation
    "noisy": {
        "same":          0.80,
        "ambiguous_low": 0.70,
        "osd_gate":      0.85,
        "hard_accept":   0.90,
        "hard_reject":   0.25,
    },
    # Phone / far-field / call-centre (channel mismatch + compression)
    "phone": {
        "same":          0.82,
        "ambiguous_low": 0.72,
        "osd_gate":      0.85,
        "hard_accept":   0.92,
        "hard_reject":   0.30,
    },
    # Forensic / high-stakes: prefer false-rejection over false-acceptance
    "forensic": {
        "same":          0.90,
        "ambiguous_low": 0.70,
        "osd_gate":      0.85,
        "hard_accept":   0.95,
        "hard_reject":   0.50,
    },
}


def classify_similarity(score: float, condition: str = "noisy") -> str:
    """
    Map a cosine similarity score to a decision label.

    Returns one of:
        "same_speaker"      — above same threshold
        "ambiguous_overlap" — in the grey zone (trigger OSD / resegment)
        "different_speaker" — below ambiguous_low threshold
    """
    t = THRESHOLDS[condition]
    if score >= t["hard_accept"]:
        return "same_speaker"
    if score >= t["same"]:
        return "same_speaker"
    if score >= t["ambiguous_low"]:
        return "ambiguous_overlap"
    return "different_speaker"


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load any audio file with librosa, resample to 16 kHz mono.
    Handles multi-channel by averaging channels.
    """
    log.info(f"Loading audio: {path}")
    
    # Load with librosa (returns numpy float32, sr)
    y, sr = librosa.load(path, sr=None, mono=False)
    
    # Convert to mono if multi-channel
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    
    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Convert to torch tensor (1, time)
    waveform = torch.from_numpy(y).unsqueeze(0).float()
    
    duration = waveform.shape[1] / target_sr
    log.info(f"Audio: {duration:.1f}s  |  {target_sr} Hz  |  mono")
    return waveform, target_sr


def validate_audio(waveform: torch.Tensor, sr: int, min_duration: float = 1.0):
    """Raise if audio is too short to diarize meaningfully."""
    duration = waveform.shape[1] / sr
    if duration < min_duration:
        raise ValueError(
            f"Audio is only {duration:.2f}s — need at least {min_duration}s."
        )


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — ECAPA-TDNN EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def load_ecapa() -> EncoderClassifier:
    """
    Load ECAPA-TDNN from a local pre-downloaded directory.
    Avoids HuggingFace Hub download and Windows symlink privilege errors.
    """
    local_model_path = r"C:\Users\druiv\.cache\pretrained_models\spkrec-ecapa-voxceleb"
    log.info(f"Loading ECAPA-TDNN from local path: {local_model_path}")
    encoder = EncoderClassifier.from_hparams(
        source=local_model_path,
        run_opts={"device": str(DEVICE)},
    )
    log.info("ECAPA-TDNN ready")
    return encoder


def extract_embeddings(
    encoder:  EncoderClassifier,
    waveform: torch.Tensor,
    sr:       int,
    seg_dur:  float = 1.5,   # window length in seconds
    seg_step: float = 0.75,  # hop size in seconds (50% overlap)
    min_chunk_samples: int = 1600,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Sliding-window ECAPA embedding extraction.

    Returns:
        segments   : list of (start_sec, end_sec) for each window
        embeddings : float32 array of shape (N, 192)
    """
    seg_samples  = int(seg_dur  * sr)
    step_samples = int(seg_step * sr)
    total        = waveform.shape[1]

    segments:   List[Tuple[float, float]] = []
    embeddings: List[np.ndarray]          = []

    start = 0
    while start + seg_samples <= total:
        end   = start + seg_samples
        chunk = waveform[:, start:end].to(DEVICE)

        if chunk.shape[1] < min_chunk_samples:
            start += step_samples
            continue

        with torch.no_grad():
            emb = encoder.encode_batch(chunk)       # (1, 1, 192)
            emb = emb.squeeze().cpu().numpy()       # (192,)

        segments.append((start / sr, end / sr))
        embeddings.append(emb)
        start += step_samples

    if not embeddings:
        raise RuntimeError("No embeddings extracted — audio may be too short.")

    log.info(f"Extracted {len(segments)} embeddings "
             f"(window={seg_dur}s, hop={seg_step}s)")
    return segments, np.vstack(embeddings).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — SPEAKER CLUSTERING
# ══════════════════════════════════════════════════════════════════════

def _eigengap_n_speakers(
    affinity: np.ndarray,
    min_spk:  int = 2,
    max_spk:  int = 8,
) -> int:
    """
    Estimate speaker count via eigengap heuristic on the cosine affinity matrix.
    Finds the index with the largest eigenvalue drop (gap) in [min, max].
    """
    eigenvalues = np.sort(np.linalg.eigvalsh(affinity))[::-1]
    # Clamp search range to available eigenvalues
    max_idx = min(max_spk, len(eigenvalues) - 1)
    gaps    = np.abs(np.diff(eigenvalues[min_spk - 1 : max_idx + 1]))
    n       = int(np.argmax(gaps) + min_spk)
    log.info(f"Eigengap auto-detected {n} speaker(s) "
             f"(search range {min_spk}–{max_idx})")
    return n


def cluster_speakers(
    embeddings:  np.ndarray,
    n_speakers:  Optional[int] = None,
    min_spk:     int = 2,
    max_spk:     int = 8,
    smooth_labels: bool = True,
    median_k:    int = 5,
) -> Tuple[np.ndarray, int]:
    """
    Spectral clustering on L2-normalised ECAPA embeddings (cosine affinity).

    Steps:
      1. L2-normalise all embeddings
      2. Build cosine affinity matrix, clip to [0, 1]
      3. Eigengap heuristic if n_speakers is None
      4. SpectralClustering with precomputed affinity
      5. Optional median-filter smoothing on labels to remove flicker

    Returns:
        labels      : int array of shape (N,) — speaker index per window
        n_speakers  : final number of speakers used
    """
    emb_norm = normalize(embeddings, norm="l2")
    affinity = np.clip(emb_norm @ emb_norm.T, 0.0, 1.0)

    if n_speakers is None:
        n_speakers = _eigengap_n_speakers(affinity, min_spk, max_spk)

    # Ensure we don't request more clusters than segments
    n_speakers = min(n_speakers, len(embeddings))

    sc = SpectralClustering(
        n_clusters=n_speakers,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(affinity)

    if smooth_labels and len(labels) >= median_k:
        # Median filter removes brief label flickers (< half a window)
        labels = medfilt(labels.astype(float), kernel_size=median_k).astype(int)

    log.info(f"Clustered into {n_speakers} speaker(s) "
             f"({'auto' if n_speakers is None else 'fixed'})")
    return labels, n_speakers


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — TURN BUILDING
# ══════════════════════════════════════════════════════════════════════

def build_turns(
    segments: List[Tuple[float, float]],
    labels:   np.ndarray,
    min_dur:  float = 0.3,
) -> List[Turn]:
    """
    Convert per-window labels into contiguous speaker turns.

    Merges adjacent windows with the same label, discards turns shorter
    than min_dur seconds (avoids one-frame artefacts).
    """
    if not segments:
        return []

    turns: List[Turn] = []
    cur_label = int(labels[0])
    cur_start = segments[0][0]

    for i in range(1, len(segments)):
        lbl = int(labels[i])
        if lbl != cur_label:
            end = segments[i][0]
            if end - cur_start >= min_dur:
                turns.append(Turn(
                    start=cur_start,
                    end=end,
                    speaker=f"SPEAKER_{cur_label:02d}",
                    label="speech",
                ))
            cur_label = lbl
            cur_start = segments[i][0]

    # Flush last turn
    last_end = segments[-1][1]
    if last_end - cur_start >= min_dur:
        turns.append(Turn(
            start=cur_start,
            end=last_end,
            speaker=f"SPEAKER_{cur_label:02d}",
            label="speech",
        ))

    log.info(f"Built {len(turns)} initial speaker turns")
    return turns


def merge_short_turns(turns: List[Turn], min_dur: float = 0.5) -> List[Turn]:
    """
    Post-processing: merge turns shorter than min_dur into the adjacent
    turn with the highest cosine similarity (or simply the nearest neighbour
    if scores are unavailable).
    """
    if len(turns) < 2:
        return turns

    merged = list(turns)
    changed = True
    while changed:
        changed = False
        result: List[Turn] = []
        i = 0
        while i < len(merged):
            t = merged[i]
            if t.duration < min_dur and len(merged) > 1:
                # Absorb into left or right neighbour (prefer same speaker)
                prev_spk = result[-1].speaker if result else None
                next_spk = merged[i + 1].speaker if i + 1 < len(merged) else None
                absorb = next_spk if prev_spk is None else (
                    prev_spk if next_spk is None else (
                        prev_spk if prev_spk == next_spk else prev_spk
                    )
                )
                if result and result[-1].speaker == absorb:
                    result[-1] = Turn(
                        start=result[-1].start,
                        end=t.end,
                        speaker=result[-1].speaker,
                        score=result[-1].score,
                        label=result[-1].label,
                    )
                else:
                    result.append(t)
                changed = True
            else:
                result.append(t)
            i += 1
        merged = result

    return merged


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — OVERLAP SPEECH DETECTION (OSD)
# ══════════════════════════════════════════════════════════════════════

def detect_overlaps_pyannote(
    audio_path: str,
    hf_token:   str,
    min_overlap_dur: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    Use pyannote's neural OSD to detect regions where ≥2 speakers overlap.
    Requires a HuggingFace token with access to pyannote/overlapped-speech-detection.

    Returns list of (start_sec, end_sec) overlap regions.
    """
    try:
        from pyannote.audio.pipelines import OverlappedSpeechDetection
    except ImportError:
        log.warning("pyannote.audio not installed — skipping OSD. "
                    "pip install pyannote.audio")
        return []

    log.info("Running pyannote Overlapped Speech Detection …")
    osd = OverlappedSpeechDetection.from_pretrained(
        "pyannote/overlapped-speech-detection",
        use_auth_token=hf_token,
    )
    osd_output = osd(audio_path)

    regions = []
    for segment, _, label in osd_output.itertracks(yield_label=True):
        if label == "OVERLAP" and segment.duration >= min_overlap_dur:
            regions.append((segment.start, segment.end))

    log.info(f"OSD found {len(regions)} overlap region(s)")
    return regions


def detect_overlaps_embedding(
    encoder:    EncoderClassifier,
    waveform:   torch.Tensor,
    sr:         int,
    turns:      List[Turn],
    centroids:  Dict[str, np.ndarray],
    condition:  str = "noisy",
    window:     float = 0.5,
    hop:        float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Fallback OSD using ECAPA embeddings — no pyannote token required.

    For each short sub-window, compute cosine similarity to ALL known
    centroids. If two or more speakers score above ambiguous_low
    simultaneously, flag the window as an overlap.

    Returns list of (start_sec, end_sec) overlap regions.
    """
    t        = THRESHOLDS[condition]
    gate     = t["ambiguous_low"]

    c_ids    = list(centroids.keys())
    c_matrix = normalize(
        np.vstack([centroids[k] for k in c_ids]), norm="l2"
    )  # (n_speakers, 192)

    win_samp = int(window * sr)
    hop_samp = int(hop    * sr)
    total    = waveform.shape[1]

    overlap_windows: List[Tuple[float, float]] = []

    start = 0
    while start + win_samp <= total:
        end   = start + win_samp
        chunk = waveform[:, start:end].to(DEVICE)

        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().cpu().numpy()

        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()   # similarity to each centroid

        # Count how many speakers score above the ambiguity gate
        active = np.sum(sims >= gate)
        if active >= 2:
            overlap_windows.append((start / sr, end / sr))

        start += hop_samp

    if not overlap_windows:
        return []

    # Merge contiguous flagged windows into regions
    regions: List[Tuple[float, float]] = []
    rs, re = overlap_windows[0]
    for (ws, we) in overlap_windows[1:]:
        if ws <= re + hop:   # contiguous or close enough
            re = we
        else:
            regions.append((rs, re))
            rs, re = ws, we
    regions.append((rs, re))

    log.info(f"Embedding-based OSD found {len(regions)} overlap region(s)")
    return regions


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — SPEAKER CENTROIDS
# ══════════════════════════════════════════════════════════════════════

def is_in_overlap(start: float, end: float,
                  overlap_regions: List[Tuple[float, float]]) -> bool:
    """True if [start, end] overlaps with any flagged region."""
    for (ov_s, ov_e) in overlap_regions:
        if start < ov_e and end > ov_s:
            return True
    return False


def build_speaker_centroids(
    encoder:         EncoderClassifier,
    waveform:        torch.Tensor,
    sr:              int,
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
    min_chunk_sec:   float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Compute per-speaker centroid embeddings using ONLY clean (non-overlapping) turns.

    Each centroid = L2-normalised mean of all clean-turn embeddings for that speaker.
    """
    log.info("Building per-speaker centroids from clean turns …")
    min_samples = int(min_chunk_sec * sr)
    per_speaker: Dict[str, List[np.ndarray]] = {}

    for t in turns:
        if is_in_overlap(t.start, t.end, overlap_regions):
            continue

        s_samp = int(t.start * sr)
        e_samp = int(t.end   * sr)
        chunk  = waveform[:, s_samp:e_samp].to(DEVICE)

        if chunk.shape[1] < min_samples:
            continue

        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().cpu().numpy()

        per_speaker.setdefault(t.speaker, []).append(emb)

    if not per_speaker:
        raise RuntimeError(
            "No clean turns found to build speaker centroids. "
            "Try reducing min_chunk_sec or check overlap detection."
        )

    centroids = {
        spk: normalize(
            np.mean(np.vstack(embs), axis=0, keepdims=True), norm="l2"
        ).squeeze()
        for spk, embs in per_speaker.items()
    }
    log.info(f"Built centroids for {len(centroids)} speaker(s): "
             f"{list(centroids.keys())}")
    return centroids


def score_turns_against_centroids(
    encoder:    EncoderClassifier,
    waveform:   torch.Tensor,
    sr:         int,
    turns:      List[Turn],
    centroids:  Dict[str, np.ndarray],
    condition:  str = "noisy",
) -> List[Turn]:
    """
    Re-score every clean turn against its own centroid and attach the
    cosine similarity score + uncertainty label.
    """
    t        = THRESHOLDS[condition]
    c_ids    = list(centroids.keys())
    c_matrix = normalize(
        np.vstack([centroids[k] for k in c_ids]), norm="l2"
    )

    scored: List[Turn] = []
    for turn in turns:
        s_samp = int(turn.start * sr)
        e_samp = int(turn.end   * sr)
        chunk  = waveform[:, s_samp:e_samp].to(DEVICE)

        if chunk.shape[1] < 1600:
            scored.append(turn)
            continue

        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().cpu().numpy()

        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()

        # Score against the assigned speaker's centroid
        if turn.speaker in c_ids:
            idx   = c_ids.index(turn.speaker)
            score = float(sims[idx])
        else:
            score = float(np.max(sims))

        decision = classify_similarity(score, condition)
        label    = "speech" if decision == "same_speaker" else (
                   "uncertain" if decision == "ambiguous_overlap"
                   else "speech")

        scored.append(Turn(
            start=turn.start,
            end=turn.end,
            speaker=turn.speaker,
            score=score,
            label=label,
        ))

    return scored


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — OVERLAP STRATEGIES
# ══════════════════════════════════════════════════════════════════════

def strategy_nearest_neighbour(
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
) -> List[Turn]:
    """
    Strategy A — Nearest-Neighbour (fastest, lowest compute).

    For each overlap region, assign the speakers immediately before and
    after it. Used by ByteDance (VoxSRC-2021, rank 2nd).
    Short overlaps (< 1s) are best served by this method.
    """
    log.info("[Strategy: nn] Assigning overlaps via nearest-neighbour …")
    extra: List[Turn] = []

    for (ov_s, ov_e) in overlap_regions:
        before = [t for t in turns if t.end   <= ov_s]
        after  = [t for t in turns if t.start >= ov_e]

        spk_before = before[-1].speaker if before else None
        spk_after  = after[0].speaker   if after  else None

        speakers = list({s for s in [spk_before, spk_after] if s})
        for spk in speakers:
            extra.append(Turn(
                start=ov_s, end=ov_e, speaker=spk,
                score=0.0, label="overlap",
            ))

    result = sorted(turns + extra, key=lambda x: x.start)
    log.info(f"[nn] Added {len(extra)} overlap turn(s)")
    return result


def strategy_resegment(
    encoder:         EncoderClassifier,
    waveform:        torch.Tensor,
    sr:              int,
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
    centroids:       Dict[str, np.ndarray],
    condition:       str = "noisy",
    top_k:           int = 2,
) -> List[Turn]:
    """
    Strategy B — ECAPA Resegmentation (balanced; recommended default).

    For each overlap region, extract a mixed-audio embedding and compute
    cosine similarity to ALL known speaker centroids.
    Assign the top-K most similar speakers above the ambiguous_low threshold.

    Good for 2-speaker overlaps without requiring a separation model.
    """
    log.info("[Strategy: resegment] Running ECAPA resegmentation …")
    t        = THRESHOLDS[condition]
    c_ids    = list(centroids.keys())
    c_matrix = normalize(
        np.vstack([centroids[k] for k in c_ids]), norm="l2"
    )

    extra: List[Turn] = []

    for (ov_s, ov_e) in overlap_regions:
        s_samp = int(ov_s * sr)
        e_samp = int(ov_e * sr)
        chunk  = waveform[:, s_samp:e_samp].to(DEVICE)

        if chunk.shape[1] < 1600:
            log.debug(f"  Skipping short overlap {ov_s:.2f}–{ov_e:.2f}s")
            continue

        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().cpu().numpy()

        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()

        # Take top-K speakers that score above the ambiguity gate
        top_k_actual = min(top_k, len(c_ids))
        top_idx      = np.argsort(sims)[::-1][:top_k_actual]

        assigned = 0
        for idx in top_idx:
            score = float(sims[idx])
            if score < t["ambiguous_low"]:
                break   # scores are sorted; nothing below will pass
            extra.append(Turn(
                start=ov_s, end=ov_e,
                speaker=c_ids[idx],
                score=score,
                label="overlap",
            ))
            assigned += 1

        if assigned == 0:
            # Fallback: assign the single best speaker as uncertain
            best = int(np.argmax(sims))
            extra.append(Turn(
                start=ov_s, end=ov_e,
                speaker=c_ids[best],
                score=float(sims[best]),
                label="uncertain",
            ))

    result = sorted(turns + extra, key=lambda x: x.start)
    log.info(f"[resegment] Added {len(extra)} overlap/uncertain turn(s)")
    return result


def strategy_separate(
    encoder:         EncoderClassifier,
    waveform:        torch.Tensor,
    sr:              int,
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
    centroids:       Dict[str, np.ndarray],
    condition:       str = "noisy",
) -> List[Turn]:
    """
    Strategy C — SepFormer Separation + ECAPA Re-ID (most accurate).

    For each overlap region:
      1. Run SepFormer to produce N separated source signals
      2. Extract ECAPA embeddings from each source
      3. Match each source to the nearest speaker centroid (cosine)

    Best for long overlaps (> 1s) and 3+ speaker scenarios.
    Requires: pip install speechbrain
    """
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:
        log.error("SepformerSeparation not available — "
                  "falling back to resegment strategy.")
        return strategy_resegment(
            encoder, waveform, sr, turns, overlap_regions,
            centroids, condition, top_k=2,
        )

    log.info("[Strategy: separate] Loading SepFormer …")
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",   # noise + reverb robust version
        run_opts={"device": str(DEVICE)},
    )

    t        = THRESHOLDS[condition]
    c_ids    = list(centroids.keys())
    c_matrix = normalize(
        np.vstack([centroids[k] for k in c_ids]), norm="l2"
    )

    extra: List[Turn] = []

    for (ov_s, ov_e) in overlap_regions:
        s_samp = int(ov_s * sr)
        e_samp = int(ov_e * sr)
        chunk  = waveform[:, s_samp:e_samp].to(DEVICE)

        if chunk.shape[1] < int(0.5 * sr):
            log.debug(f"  Skipping overlap < 0.5s at {ov_s:.2f}s")
            continue

        try:
            with torch.no_grad():
                separated = separator.separate_batch(chunk)
                # separated: (1, samples, n_sources)
        except Exception as exc:
            log.warning(f"  SepFormer failed at {ov_s:.2f}s: {exc} — skipping")
            continue

        n_sources = separated.shape[-1]
        seen_speakers = set()

        for src_i in range(n_sources):
            src_audio = separated[..., src_i]   # (1, samples)

            with torch.no_grad():
                emb = encoder.encode_batch(src_audio).squeeze().cpu().numpy()

            emb_n = normalize(emb.reshape(1, -1), norm="l2")
            sims  = (emb_n @ c_matrix.T).flatten()
            best  = int(np.argmax(sims))
            score = float(sims[best])

            decision = classify_similarity(score, condition)
            spk      = c_ids[best]

            # Avoid assigning same speaker twice for one overlap region
            if spk in seen_speakers:
                continue
            seen_speakers.add(spk)

            label = "overlap" if decision != "different_speaker" else "uncertain"
            extra.append(Turn(
                start=ov_s, end=ov_e,
                speaker=spk,
                score=score,
                label=label,
            ))

    result = sorted(turns + extra, key=lambda x: x.start)
    log.info(f"[separate] Added {len(extra)} overlap turn(s)")
    return result


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — OUTPUT
# ══════════════════════════════════════════════════════════════════════

def export_rttm(result: DiarizationResult, path: str):
    """
    Write turns to RTTM (Rich Transcription Time Marks) format.
    Standard format for diarization evaluation (pyannote, dscore, etc.).

    Format per line:
        SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    file_id = Path(result.audio_path).stem
    lines   = []
    for t in result.turns:
        dur = max(0.001, t.end - t.start)
        lines.append(
            f"SPEAKER {file_id} 1 {t.start:.3f} {dur:.3f} "
            f"<NA> <NA> {t.speaker} <NA> <NA>"
        )
    Path(path).write_text("\n".join(lines) + "\n")
    log.info(f"RTTM written → {path}")


def print_result(result: DiarizationResult):
    """Pretty-print diarization result to stdout."""
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"  File       : {result.audio_path}")
    print(f"  Speakers   : {result.n_speakers}")
    print(f"  Strategy   : {result.strategy}")
    print(f"  Condition  : {result.condition}")
    print(f"  Thresholds : same>{result.thresholds.get('same', '?')}  "
          f"ambiguous>{result.thresholds.get('ambiguous_low', '?')}  "
          f"osd_gate={result.thresholds.get('osd_gate', '?')}")
    print(bar)
    print(f"  {'START':>8}   {'END':>8}   {'DUR':>6}   {'SCORE':>6}   "
          f"{'LABEL':<12}  SPEAKER")
    print(bar)

    for t in result.turns:
        score_str = f"{t.score:.3f}" if t.score > 0 else "  —  "
        tag       = f"[{t.label}]" if t.label != "speech" else ""
        print(f"  {t.start:>7.2f}s  {t.end:>7.2f}s  "
              f"{t.duration:>5.2f}s  {score_str:>6}  "
              f"{tag:<12}  {t.speaker}")

    print(bar)
    total_overlap = sum(
        t.duration for t in result.turns if t.label == "overlap"
    )
    total_uncertain = sum(
        t.duration for t in result.turns if t.label == "uncertain"
    )
    print(f"  Turns       : {len(result.turns)} total  |  "
          f"{len(result.clean_turns())} clean  |  "
          f"{len(result.overlap_turns())} overlap  |  "
          f"{len(result.uncertain_turns())} uncertain")
    print(f"  Overlap dur : {total_overlap:.2f}s")
    print(f"  Uncertain   : {total_uncertain:.2f}s")
    print(f"{bar}\n")


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    audio_path:  str,
    strategy:    str            = "resegment",   # "nn" | "resegment" | "separate"
    condition:   str            = "noisy",       # "clean" | "noisy" | "phone" | "forensic"
    n_speakers:  Optional[int]  = None,
    min_spk:     int            = 2,
    max_spk:     int            = 8,
    hf_token:    Optional[str]  = None,          # needed for pyannote OSD
    rttm_path:   Optional[str]  = None,
    seg_dur:     float          = 1.5,
    seg_step:    float          = 0.75,
    min_turn_dur: float         = 0.3,
) -> DiarizationResult:
    """
    Full robust overlap-aware speaker diarization pipeline.

    Parameters
    ----------
    audio_path   : path to .wav / .flac / .mp3 file
    strategy     : overlap handling — "nn" | "resegment" | "separate"
    condition    : acoustic condition for threshold selection
                   "clean" | "noisy" | "phone" | "forensic"
    n_speakers   : fix speaker count, or None for auto-detection
    min_spk/max_spk : search bounds for auto speaker count
    hf_token     : HuggingFace token for pyannote OSD (optional)
    rttm_path    : if set, writes RTTM output to this path
    seg_dur      : sliding window length in seconds
    seg_step     : sliding window hop in seconds
    min_turn_dur : discard turns shorter than this (seconds)

    Returns
    -------
    DiarizationResult with all turns, scores, and metadata
    """
    log.info(f"{'='*60}")
    log.info(f"  ECAPA-TDNN Diarization  |  strategy={strategy}  |  condition={condition}")
    log.info(f"{'='*60}")

    thresholds = THRESHOLDS[condition]

    # ── 1. Load audio ─────────────────────────────────────────────────
    waveform, sr = load_audio(audio_path)
    validate_audio(waveform, sr)

    # ── 2. Load ECAPA-TDNN ────────────────────────────────────────────
    encoder = load_ecapa()

    # ── 3. Extract sliding-window embeddings ──────────────────────────
    segments, embeddings = extract_embeddings(
        encoder, waveform, sr,
        seg_dur=seg_dur,
        seg_step=seg_step,
    )

    # ── 4. Cluster → speaker labels ───────────────────────────────────
    labels, n_spk = cluster_speakers(
        embeddings,
        n_speakers=n_speakers,
        min_spk=min_spk,
        max_spk=max_spk,
        smooth_labels=True,
    )

    # ── 5. Build contiguous turns ─────────────────────────────────────
    turns = build_turns(segments, labels, min_dur=min_turn_dur)
    turns = merge_short_turns(turns, min_dur=min_turn_dur)

    # ── 6. Detect overlapping regions ─────────────────────────────────
    #    Try pyannote first (better accuracy), fall back to embedding-based
    if hf_token:
        overlap_regions = detect_overlaps_pyannote(
            audio_path, hf_token, min_overlap_dur=0.3
        )
    else:
        log.info("No HF token — using embedding-based OSD (no pyannote needed)")
        # Bootstrap centroids from the initial clean turns for OSD
        try:
            bootstrap_centroids = build_speaker_centroids(
                encoder, waveform, sr, turns, overlap_regions=[], min_chunk_sec=0.3
            )
            overlap_regions = detect_overlaps_embedding(
                encoder, waveform, sr, turns,
                bootstrap_centroids, condition=condition,
            )
        except RuntimeError:
            log.warning("Could not build bootstrap centroids — no OSD applied")
            overlap_regions = []

    # ── 7. Build speaker centroids from clean segments ────────────────
    try:
        centroids = build_speaker_centroids(
            encoder, waveform, sr, turns, overlap_regions
        )
    except RuntimeError as e:
        log.warning(f"{e} — using all turns for centroids")
        centroids = build_speaker_centroids(
            encoder, waveform, sr, turns, overlap_regions=[]
        )

    # ── 8. Score all turns against centroids ──────────────────────────
    turns = score_turns_against_centroids(
        encoder, waveform, sr, turns, centroids, condition
    )

    # ── 9. Apply overlap strategy ─────────────────────────────────────
    if not overlap_regions:
        log.info("No overlap regions found — returning standard diarization")
        final_turns = turns
    elif strategy == "nn":
        final_turns = strategy_nearest_neighbour(turns, overlap_regions)
    elif strategy == "resegment":
        final_turns = strategy_resegment(
            encoder, waveform, sr, turns, overlap_regions,
            centroids, condition=condition, top_k=2,
        )
    elif strategy == "separate":
        final_turns = strategy_separate(
            encoder, waveform, sr, turns, overlap_regions,
            centroids, condition=condition,
        )
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose: nn | resegment | separate"
        )

    # ── 10. Package result ────────────────────────────────────────────
    result = DiarizationResult(
        turns=final_turns,
        n_speakers=n_spk,
        audio_path=audio_path,
        strategy=strategy,
        condition=condition,
        thresholds=thresholds,
    )

    # ── 11. Optional RTTM export ──────────────────────────────────────
    if rttm_path:
        export_rttm(result, rttm_path)

    return result


if __name__ == "__main__":
  from main._main_overlap_aware_diarization import main

  main()
