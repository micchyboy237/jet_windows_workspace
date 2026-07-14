"""
Overlap-Aware Speaker Diarization with Selectable Embedding Models
==================================================================
Full robust pipeline integrating:
  • Selectable speaker embedding models (via embedding_model_factory)
  • Pyannote OSD — overlap speech detection
  • SepFormer (SpeechBrain) — speech separation (optional)
  • Adaptive cosine thresholds by acoustic condition
  • Three overlap strategies — nn | resegment | separate
  • RTTM export — standard diarization output format
  • Confidence scoring — per-turn similarity scores
  • Robust turn-building — median-filter + min-duration merge

Install:
    pip install speechbrain pyannote.audio librosa torch scikit-learn scipy

Usage:
    python overlap_aware_diarization.py meeting.wav --strategy resegment --condition noisy
    python overlap_aware_diarization.py call.wav --strategy separate --condition phone --speakers 2
    python overlap_aware_diarization.py studio.wav --strategy nn --condition clean --rttm out.rttm
    python overlap_aware_diarization.py audio.wav --embedding-model modelscope_eres2netv2
"""
import torch
import librosa
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, TypedDict
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.signal import medfilt

# Import the embedding model factory
try:
    from services.embedding_model_factory import (
        BaseEmbeddingModel,
        EmbeddingModelType,
        create_embedding_model,
    )
    from services.audio_utils import load_audio
except ImportError:
    from embedding_model_factory import (
        BaseEmbeddingModel,
        EmbeddingModelType,
        create_embedding_model,
    )
    from audio_utils import load_audio

log = logging.getLogger("diarization")
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

# ── Types ──

class SegmentInfo(TypedDict):
    """
    Metadata for a single extracted speaker-turn segment, as produced by
    `extract_speaker_segments` (pre-save, no file paths yet).
    """
    segment_num:  int
    segment_name: str
    speaker:      str
    start:        float
    end:          float
    duration:     float
    score:        float
    label:        str
    type:         str   # "clean" | "outlier"
    global_order: int


# ── Default pipeline parameters ──
# These defaults are shared between the module and any CLI wrappers.
# They reflect the same values used in split_speaker_segments() and _main_*.py get_args().

DEFAULT_STRATEGY        = "resegment"
DEFAULT_CONDITION       = "noisy"
DEFAULT_N_SPEAKERS      = None          # auto-detect
DEFAULT_MIN_SPK         = 2
DEFAULT_MAX_SPK         = 8
DEFAULT_HF_TOKEN        = None
DEFAULT_RTTM_PATH       = None
DEFAULT_SEG_DUR         = 2.0           # seconds
DEFAULT_SEG_STEP        = 1.0           # seconds
DEFAULT_MIN_TURN_DUR    = 0.3           # seconds
DEFAULT_EMBEDDING_MODEL = "nemo_titanet"
DEFAULT_DEVICE          = None          # auto-detect


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS: Dict[str, Dict[str, float]] = {
    "clean": {
        "same":          0.75,
        "ambiguous_low": 0.65,
        "osd_gate":      0.85,
        "hard_accept":   0.90,
        "hard_reject":   0.25,
    },
    "noisy": {
        "same":          0.80,
        "ambiguous_low": 0.70,
        "osd_gate":      0.85,
        "hard_accept":   0.90,
        "hard_reject":   0.25,
    },
    "phone": {
        "same":          0.82,
        "ambiguous_low": 0.72,
        "osd_gate":      0.85,
        "hard_accept":   0.92,
        "hard_reject":   0.30,
    },
    "forensic": {
        "same":          0.90,
        "ambiguous_low": 0.70,
        "osd_gate":      0.85,
        "hard_accept":   0.95,
        "hard_reject":   0.50,
    },
}


@dataclass
class Turn:
    """A single speaker turn, possibly overlapping with another."""
    start:    float
    end:      float
    speaker:  str
    score:    float = 0.0
    label:    str   = "speech"
    
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
    embedding_model: str = "nemo_titanet"
    
    def clean_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "speech"]
    
    def overlap_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "overlap"]
    
    def uncertain_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.label == "uncertain"]


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


def validate_audio(waveform: torch.Tensor, sr: int, min_duration: float = 1.0):
    """Raise if audio is too short to diarize meaningfully."""
    duration = waveform.shape[1] / sr
    if duration < min_duration:
        raise ValueError(
            f"Audio is only {duration:.2f}s — need at least {min_duration}s."
        )


def load_embedding_model(
    model_type: Union[str, BaseEmbeddingModel] = "nemo_titanet",
    device: Optional[torch.device] = None,
) -> BaseEmbeddingModel:
    """
    Load a speaker embedding model via the factory, or reuse one if
    an already-instantiated BaseEmbeddingModel is passed in.

    Parameters
    ----------
    model_type : str or BaseEmbeddingModel
        Either a model type string (e.g. "nemo_titanet") to load fresh,
        or an existing BaseEmbeddingModel instance to reuse as-is
        (skips loading entirely).
    device : torch.device, optional

    Returns
    -------
    BaseEmbeddingModel instance ready for encoding
    """
    if isinstance(model_type, BaseEmbeddingModel):
        log.info(
            f"Reusing already-loaded embedding model instance: {model_type} "
            f"(skipping load_embedding_model / create_embedding_model)"
        )
        return model_type

    log.info(f"Loading embedding model: {model_type}")
    model = create_embedding_model(
        model_type=model_type,
        device=device or DEVICE,
    )
    log.info(f"Embedding model ready: {model}")
    return model


def _encode_batch(
    model: BaseEmbeddingModel,
    waveform: torch.Tensor,
    sr: int,
) -> np.ndarray:
    """
    Encode a batch of audio chunks using the factory model.
    
    This adapter handles the different interfaces:
    - SpeechBrain models have native encode_batch()
    - Other models use encode() on individual chunks
    
    Parameters
    ----------
    model : BaseEmbeddingModel
        The embedding model from factory
    waveform : torch.Tensor
        Shape (batch, samples) or (1, samples)
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Embeddings with shape (batch, dim) or (dim,)
    """
    # SpeechBrain models have native encode_batch
    if hasattr(model, '_classifier') and model._classifier is not None:
        if hasattr(model._classifier, 'encode_batch'):
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.float().to(model._device)
            emb = model._classifier.encode_batch(waveform)
            return emb.squeeze().cpu().numpy()
    
    # NeMo models have get_embedding but we need batch
    if hasattr(model, '_speaker_model') and model._speaker_model is not None:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        embeddings = []
        for i in range(waveform.shape[0]):
            chunk = waveform[i:i+1]
            emb = model.encode(chunk, sr)
            embeddings.append(emb)
        result = np.vstack(embeddings)
        return result.squeeze() if result.shape[0] == 1 else result
    
    # Default: encode each chunk individually
    if waveform.dim() == 1:
        emb = model.encode(waveform, sr)
        return emb.flatten()
    else:
        embeddings = []
        for i in range(waveform.shape[0]):
            chunk = waveform[i]
            emb = model.encode(chunk, sr)
            embeddings.append(emb.flatten())
        result = np.vstack(embeddings)
        return result


def extract_embeddings(
    model:    BaseEmbeddingModel,
    waveform: torch.Tensor,
    sr:       int,
    seg_dur:  float = 1.5,
    seg_step: float = 0.75,
    min_chunk_samples: int = 1600,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Sliding-window embedding extraction using factory model.
    
    Returns:
        segments   : list of (start_sec, end_sec) for each window
        embeddings : float32 array of shape (N, dim)
    """
    seg_samples  = int(seg_dur  * sr)
    step_samples = int(seg_step * sr)
    total        = waveform.shape[1]
    segments:   List[Tuple[float, float]] = []
    embeddings: List[np.ndarray]          = []
    
    start = 0
    while start + seg_samples <= total:
        end   = start + seg_samples
        chunk = waveform[:, start:end]
        if chunk.shape[1] < min_chunk_samples:
            start += step_samples
            continue
        
        with torch.no_grad():
            emb = _encode_batch(model, chunk, sr)
            emb = emb.squeeze()
        
        segments.append((start / sr, end / sr))
        embeddings.append(emb)
        start += step_samples
    
    if not embeddings:
        raise RuntimeError("No embeddings extracted — audio may be too short.")
    
    log.info(f"Extracted {len(segments)} embeddings "
             f"(window={seg_dur}s, hop={seg_step}s, dim={embeddings[0].shape[-1]})")
    return segments, np.vstack(embeddings).astype(np.float32)


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
    n_eig = len(eigenvalues)

    effective_min_spk = max(1, min(min_spk, n_eig))
    effective_max_spk = min(max_spk, n_eig - 1)

    if n_eig <= effective_min_spk or effective_max_spk < effective_min_spk:
        n = effective_min_spk
        log.warning(
            f"Too few embeddings ({n_eig}) for eigengap search in range "
            f"[{min_spk}, {max_spk}] — falling back to {n} speaker(s)"
        )
        return n

    gaps = np.abs(np.diff(eigenvalues[effective_min_spk - 1 : effective_max_spk + 1]))
    if gaps.size == 0:
        n = effective_min_spk
        log.warning(f"Empty eigengap search window (n_eig={n_eig}) — falling back to {n} speaker(s)")
        return n

    n = int(np.argmax(gaps) + effective_min_spk)
    log.info(
        f"Eigengap auto-detected {n} speaker(s) "
        f"(search range {effective_min_spk}–{effective_max_spk})"
    )
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
    Spectral clustering on L2-normalised embeddings (cosine affinity).
    
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
        labels = medfilt(labels.astype(float), kernel_size=median_k).astype(int)
    
    log.info(f"Clustered into {n_speakers} speaker(s) "
             f"({'auto' if n_speakers is None else 'fixed'})")
    return labels, n_speakers


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


def detect_overlaps_embedding(
    model:       BaseEmbeddingModel,
    waveform:    torch.Tensor,
    sr:          int,
    turns:       List[Turn],
    centroids:   Dict[str, np.ndarray],
    condition:   str = "noisy",
    window:      float = 0.5,
    hop:         float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Fallback OSD using speaker embeddings — no pyannote token required.
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
    )
    
    win_samp = int(window * sr)
    hop_samp = int(hop    * sr)
    total    = waveform.shape[1]
    overlap_windows: List[Tuple[float, float]] = []
    
    start = 0
    while start + win_samp <= total:
        end   = start + win_samp
        chunk = waveform[:, start:end]
        
        with torch.no_grad():
            emb = _encode_batch(model, chunk, sr)
            emb = emb.squeeze()
        
        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()
        active = np.sum(sims >= gate)
        
        if active >= 2:
            overlap_windows.append((start / sr, end / sr))
        
        start += hop_samp
    
    if not overlap_windows:
        return []
    
    regions: List[Tuple[float, float]] = []
    rs, re = overlap_windows[0]
    for (ws, we) in overlap_windows[1:]:
        if ws <= re + hop:
            re = we
        else:
            regions.append((rs, re))
            rs, re = ws, we
    regions.append((rs, re))
    
    log.info(f"Embedding-based OSD found {len(regions)} overlap region(s)")
    return regions


def is_in_overlap(start: float, end: float,
                  overlap_regions: List[Tuple[float, float]]) -> bool:
    """True if [start, end] overlaps with any flagged region."""
    for (ov_s, ov_e) in overlap_regions:
        if start < ov_e and end > ov_s:
            return True
    return False


def build_speaker_centroids(
    model:           BaseEmbeddingModel,
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
        chunk  = waveform[:, s_samp:e_samp]
        
        if chunk.shape[1] < min_samples:
            continue
        
        with torch.no_grad():
            emb = _encode_batch(model, chunk, sr)
            emb = emb.squeeze()
        
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
    model:       BaseEmbeddingModel,
    waveform:    torch.Tensor,
    sr:          int,
    turns:       List[Turn],
    centroids:   Dict[str, np.ndarray],
    condition:   str = "noisy",
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
        chunk  = waveform[:, s_samp:e_samp]
        
        if chunk.shape[1] < 1600:
            scored.append(turn)
            continue
        
        with torch.no_grad():
            emb = _encode_batch(model, chunk, sr)
            emb = emb.squeeze()
        
        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()
        
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
    model:           BaseEmbeddingModel,
    waveform:        torch.Tensor,
    sr:              int,
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
    centroids:       Dict[str, np.ndarray],
    condition:       str = "noisy",
    top_k:           int = 2,
) -> List[Turn]:
    """
    Strategy B — Embedding Resegmentation (balanced; recommended default).
    For each overlap region, extract a mixed-audio embedding and compute
    cosine similarity to ALL known speaker centroids.
    Assign the top-K most similar speakers above the ambiguous_low threshold.
    Good for 2-speaker overlaps without requiring a separation model.
    """
    log.info("[Strategy: resegment] Running embedding resegmentation …")
    
    t        = THRESHOLDS[condition]
    c_ids    = list(centroids.keys())
    c_matrix = normalize(
        np.vstack([centroids[k] for k in c_ids]), norm="l2"
    )
    
    extra: List[Turn] = []
    for (ov_s, ov_e) in overlap_regions:
        s_samp = int(ov_s * sr)
        e_samp = int(ov_e * sr)
        chunk  = waveform[:, s_samp:e_samp]
        
        if chunk.shape[1] < 1600:
            log.debug(f"  Skipping short overlap {ov_s:.2f}–{ov_e:.2f}s")
            continue
        
        with torch.no_grad():
            emb = _encode_batch(model, chunk, sr)
            emb = emb.squeeze()
        
        emb_n = normalize(emb.reshape(1, -1), norm="l2")
        sims  = (emb_n @ c_matrix.T).flatten()
        
        top_k_actual = min(top_k, len(c_ids))
        top_idx      = np.argsort(sims)[::-1][:top_k_actual]
        
        assigned = 0
        for idx in top_idx:
            score = float(sims[idx])
            if score < t["ambiguous_low"]:
                break
            extra.append(Turn(
                start=ov_s, end=ov_e,
                speaker=c_ids[idx],
                score=score,
                label="overlap",
            ))
            assigned += 1
        
        if assigned == 0:
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
    model:           BaseEmbeddingModel,
    waveform:        torch.Tensor,
    sr:              int,
    turns:           List[Turn],
    overlap_regions: List[Tuple[float, float]],
    centroids:       Dict[str, np.ndarray],
    condition:       str = "noisy",
) -> List[Turn]:
    """
    Strategy C — SepFormer Separation + Embedding Re-ID (most accurate).
    For each overlap region:
      1. Run SepFormer to produce N separated source signals
      2. Extract embeddings from each source
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
            model, waveform, sr, turns, overlap_regions,
            centroids, condition, top_k=2,
        )
    
    log.info("[Strategy: separate] Loading SepFormer …")
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
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
        chunk  = waveform[:, s_samp:e_samp]
        
        if chunk.shape[1] < int(0.5 * sr):
            log.debug(f"  Skipping overlap < 0.5s at {ov_s:.2f}s")
            continue
        
        try:
            with torch.no_grad():
                separated = separator.separate_batch(chunk)
        except Exception as exc:
            log.warning(f"  SepFormer failed at {ov_s:.2f}s: {exc} — skipping")
            continue
        
        n_sources = separated.shape[-1]
        seen_speakers = set()
        
        for src_i in range(n_sources):
            src_audio = separated[..., src_i]
            
            with torch.no_grad():
                emb = _encode_batch(model, src_audio, sr)
                emb = emb.squeeze()
            
            emb_n = normalize(emb.reshape(1, -1), norm="l2")
            sims  = (emb_n @ c_matrix.T).flatten()
            best  = int(np.argmax(sims))
            score = float(sims[best])
            
            decision = classify_similarity(score, condition)
            spk      = c_ids[best]
            
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


def diarize_multi_speakers(
    audio_path:     Union[str, Path, np.ndarray],
    strategy:       str            = DEFAULT_STRATEGY,
    condition:      str            = DEFAULT_CONDITION,
    n_speakers:     Optional[int]  = DEFAULT_N_SPEAKERS,
    min_spk:        int            = DEFAULT_MIN_SPK,
    max_spk:        int            = DEFAULT_MAX_SPK,
    hf_token:       Optional[str]  = DEFAULT_HF_TOKEN,
    rttm_path:      Optional[str]  = DEFAULT_RTTM_PATH,
    seg_dur:        float          = DEFAULT_SEG_DUR,
    seg_step:       float          = DEFAULT_SEG_STEP,
    min_turn_dur:   float          = DEFAULT_MIN_TURN_DUR,
    embedding_model: Union[str, BaseEmbeddingModel] = DEFAULT_EMBEDDING_MODEL,
    device:         Optional[torch.device] = DEFAULT_DEVICE,
) -> DiarizationResult:
    """
    Full robust overlap-aware speaker diarization pipeline.

    Parameters
    ----------
    audio_path      : path to .wav / .flac / .mp3 file, or a pre-loaded
                      mono/multi-channel numpy waveform (assumed 16kHz)
    strategy        : overlap handling — "nn" | "resegment" | "separate"
    condition       : acoustic condition for threshold selection
                      "clean" | "noisy" | "phone" | "forensic"
    n_speakers      : fix speaker count, or None for auto-detection
    min_spk/max_spk : search bounds for auto speaker count
    hf_token        : HuggingFace token for pyannote OSD (optional)
    rttm_path       : if set, writes RTTM output to this path
    seg_dur         : sliding window length in seconds
    seg_step        : sliding window hop in seconds
    min_turn_dur    : discard turns shorter than this (seconds)
    embedding_model : string or embedding model type (default: nemo_titanet)
    device          : torch device (auto-detected if None)

    Returns
    -------
    DiarizationResult with all turns, scores, and metadata.
    (Use `split_speaker_segments` instead if you also need per-turn audio.)
    """
    embedding_model_name = (
        embedding_model.model_type.value
        if isinstance(embedding_model, BaseEmbeddingModel)
        else embedding_model
    )
    log.info(f"{'='*60}")
    log.info(f"  Speaker Diarization")
    log.info(f"  strategy={strategy}  |  condition={condition}")
    log.info(f"  embedding_model={embedding_model_name}"
             f"{' (reused instance)' if isinstance(embedding_model, BaseEmbeddingModel) else ''}")
    log.info(f"{'='*60}")

    thresholds = THRESHOLDS[condition]

    if isinstance(audio_path, np.ndarray):
        waveform = torch.from_numpy(audio_path).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = 16000
        audio_path_label = "<numpy_array>"
        log.info(f"Using pre-loaded numpy waveform, assuming sr={sr}Hz")
    else:
        waveform, sr = load_audio(audio_path, sr=16000, return_as_tensor=True)
        audio_path_label = str(audio_path)

    validate_audio(waveform, sr)

    model = load_embedding_model(
        model_type=embedding_model,
        device=device,
    )
    segments, embeddings = extract_embeddings(
        model, waveform, sr,
        seg_dur=seg_dur,
        seg_step=seg_step,
    )
    labels, n_spk = cluster_speakers(
        embeddings,
        n_speakers=n_speakers,
        min_spk=min_spk,
        max_spk=max_spk,
        smooth_labels=True,
    )
    turns = build_turns(segments, labels, min_dur=min_turn_dur)
    turns = merge_short_turns(turns, min_dur=min_turn_dur)

    try:
        bootstrap_centroids = build_speaker_centroids(
            model, waveform, sr, turns, overlap_regions=[], min_chunk_sec=0.3
        )
        overlap_regions = detect_overlaps_embedding(
            model, waveform, sr, turns,
            bootstrap_centroids, condition=condition,
        )
    except RuntimeError:
        log.warning("Could not build bootstrap centroids — no OSD applied")
        overlap_regions = []

    try:
        centroids = build_speaker_centroids(
            model, waveform, sr, turns, overlap_regions
        )
    except RuntimeError as e:
        log.warning(f"{e} — using all turns for centroids")
        centroids = build_speaker_centroids(
            model, waveform, sr, turns, overlap_regions=[]
        )

    turns = score_turns_against_centroids(
        model, waveform, sr, turns, centroids, condition
    )

    if not overlap_regions:
        log.info("No overlap regions found — returning standard diarization")
        final_turns = turns
    elif strategy == "nn":
        final_turns = strategy_nearest_neighbour(turns, overlap_regions)
    elif strategy == "resegment":
        final_turns = strategy_resegment(
            model, waveform, sr, turns, overlap_regions,
            centroids, condition=condition, top_k=2,
        )
    elif strategy == "separate":
        final_turns = strategy_separate(
            model, waveform, sr, turns, overlap_regions,
            centroids, condition=condition,
        )
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose: nn | resegment | separate"
        )

    result = DiarizationResult(
        turns=final_turns,
        n_speakers=n_spk,
        audio_path=audio_path_label,
        strategy=strategy,
        condition=condition,
        thresholds=thresholds,
        embedding_model=embedding_model,
    )

    if rttm_path:
        export_rttm(result, rttm_path)

    log.info(f"diarize_multi_speakers complete — {len(final_turns)} turn(s), "
             f"{n_spk} speaker(s)")
    return result


def extract_speaker_segments(
    result:   DiarizationResult,
    waveform: Union[torch.Tensor, np.ndarray],
    sr:       int,
) -> List[Tuple[SegmentInfo, np.ndarray]]:
    """
    Slice per-turn audio out of the full waveform and build metadata for
    each turn — pure in-memory logic, no disk writes.

    All turns (clean and uncertain) are numbered sequentially by start
    time, using 3-digit segment numbers (001-999). "speech" turns are
    tagged type="clean"; everything else (overlap/uncertain) is tagged
    type="outlier".

    Args:
        result:   Diarization result containing all speaker turns.
        waveform: Full audio waveform, shape (channels, samples),
                  (samples,), or a torch.Tensor of either shape.
        sr:       Sample rate of the waveform.

    Returns:
        List of (segment_info, segment_audio) tuples ordered by start time.
        segment_info keys: segment_num, segment_name, speaker, start, end,
        duration, score, label, type, global_order.
    """
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.detach().cpu().numpy()
    else:
        waveform_np = np.asarray(waveform)
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[None, :]

    all_turns_sorted = sorted(result.turns, key=lambda t: t.start)
    clean_count = sum(1 for t in all_turns_sorted if t.label == "speech")
    outlier_count = len(all_turns_sorted) - clean_count
    log.info(f"Extracting {len(all_turns_sorted)} segment(s) "
             f"({clean_count} clean, {outlier_count} outlier)")

    extracted: List[Tuple[dict, np.ndarray]] = []
    for idx, turn in enumerate(all_turns_sorted, 1):
        seg_type = "clean" if turn.label == "speech" else "outlier"
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_audio = waveform_np[:, start_sample:end_sample].squeeze()

        segment_info: SegmentInfo = {
            "segment_num": idx,
            "segment_name": f"segment_{idx:03d}",
            "speaker": turn.speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.duration,
            "score": turn.score,
            "label": turn.label,
            "type": seg_type,
            "global_order": idx,
        }
        extracted.append((segment_info, segment_audio))
        log.debug(f"  ✓ {segment_info['segment_name']} [{seg_type}]: "
                  f"{turn.speaker} {turn.start:.2f}s-{turn.end:.2f}s "
                  f"({turn.duration:.2f}s, {segment_audio.shape[-1]} samples)")

    return extracted


def split_speaker_segments(
    audio_path:     Union[str, Path, np.ndarray],
    strategy:       str            = DEFAULT_STRATEGY,
    condition:      str            = DEFAULT_CONDITION,
    n_speakers:     Optional[int]  = DEFAULT_N_SPEAKERS,
    min_spk:        int            = DEFAULT_MIN_SPK,
    max_spk:        int            = DEFAULT_MAX_SPK,
    hf_token:       Optional[str]  = DEFAULT_HF_TOKEN,
    rttm_path:      Optional[str]  = DEFAULT_RTTM_PATH,
    seg_dur:        float          = DEFAULT_SEG_DUR,
    seg_step:       float          = DEFAULT_SEG_STEP,
    min_turn_dur:   float          = DEFAULT_MIN_TURN_DUR,
    embedding_model: Union[str, BaseEmbeddingModel] = DEFAULT_EMBEDDING_MODEL,
    device:         Optional[torch.device] = DEFAULT_DEVICE,
) -> Tuple[DiarizationResult, List[Tuple[SegmentInfo, np.ndarray]]]:
    """
    Run full diarization and extract per-turn audio segments in one call.

    Loads the audio once (unless already a numpy array), runs
    `diarize_multi_speakers` on the in-memory waveform (avoiding a second
    file read), then slices out each turn's audio via
    `extract_speaker_segments`.

    Returns
    -------
    (DiarizationResult, list of (segment_info, segment_audio) tuples)
    """
    if isinstance(audio_path, np.ndarray):
        waveform_np = audio_path
        sr = 16000
        log.info("split_speaker_segments: using pre-loaded numpy waveform")
    else:
        waveform, sr = load_audio(audio_path, sr=16000, return_as_tensor=True)
        waveform_np = waveform.squeeze(0).numpy()

    result = diarize_multi_speakers(
        audio_path=waveform_np,
        strategy=strategy,
        condition=condition,
        n_speakers=n_speakers,
        min_spk=min_spk,
        max_spk=max_spk,
        hf_token=hf_token,
        rttm_path=rttm_path,
        seg_dur=seg_dur,
        seg_step=seg_step,
        min_turn_dur=min_turn_dur,
        embedding_model=embedding_model,
        device=device,
    )
    # Preserve the original path label for reporting/RTTM naming purposes.
    if not isinstance(audio_path, np.ndarray):
        result.audio_path = str(audio_path)

    segments = extract_speaker_segments(result, waveform_np, sr)
    log.info(f"split_speaker_segments complete — {len(segments)} segment(s) extracted")
    return result, segments


if __name__ == "__main__":
    from main._main_overlap_aware_diarization import main
    main()
