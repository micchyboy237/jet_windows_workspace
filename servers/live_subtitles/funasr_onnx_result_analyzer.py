# funasr_onnx_result_analyzer.py
"""
Comprehensive ONNX model transcription result analyzer.
Captures, logs, and visualizes all intermediate results for debugging and monitoring.
"""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AudioMeta:
    """Metadata about the input audio."""
    sample_rate: int
    duration_seconds: float
    num_samples: int
    peak_amplitude: float
    rms_energy: float
    has_speech: bool


@dataclass
class FeatureMeta:
    """Metadata about extracted features."""
    fbank_shape: Tuple[int, int]
    lfr_shape: Tuple[int, int]
    padded_shape: Tuple[int, int, int]
    num_frames: int
    lfr_factor: int = 7  # SenseVoice LFR stacks 7 frames


@dataclass
class InferenceMeta:
    """Metadata about ONNX inference."""
    input_shapes: Dict[str, Tuple[int, ...]]
    output_shapes: Dict[str, Tuple[int, ...]]
    inference_time_ms: float
    provider: str


@dataclass
class DecodingMeta:
    """Metadata about CTC decoding."""
    raw_logits_shape: Tuple[int, int]
    num_blank_tokens: int
    num_unique_tokens: int
    token_sequence: List[int]
    dedup_ratio: float  # frames after dedup / total frames


@dataclass
class TranscriptionResult:
    """Complete transcription result with all metadata."""
    audio: AudioMeta
    features: FeatureMeta
    inference: InferenceMeta
    decoding: DecodingMeta
    raw_text: str
    clean_text: str
    language_detected: str
    emotion_detected: str
    is_speech: bool
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON logging."""
        return {
            "timestamp": self.timestamp,
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "duration_s": round(self.audio.duration_seconds, 3),
                "num_samples": self.audio.num_samples,
                "peak_amplitude": round(self.audio.peak_amplitude, 4),
                "rms_energy": round(self.audio.rms_energy, 4),
                "has_speech": self.audio.has_speech,
            },
            "features": {
                "fbank_shape": list(self.features.fbank_shape),
                "lfr_shape": list(self.features.lfr_shape),
                "padded_shape": list(self.features.padded_shape),
                "num_frames": self.features.num_frames,
            },
            "inference": {
                "input_shapes": {k: list(v) for k, v in self.inference.input_shapes.items()},
                "output_shapes": {k: list(v) for k, v in self.inference.output_shapes.items()},
                "time_ms": round(self.inference.inference_time_ms, 2),
                "provider": self.inference.provider,
            },
            "decoding": {
                "raw_logits_shape": list(self.decoding.raw_logits_shape),
                "num_blank_tokens": self.decoding.num_blank_tokens,
                "num_unique_tokens": self.decoding.num_unique_tokens,
                "dedup_ratio": round(self.decoding.dedup_ratio, 3),
                "token_sequence": self.decoding.token_sequence,
            },
            "transcription": {
                "raw_text": self.raw_text,
                "clean_text": self.clean_text,
                "language": self.language_detected,
                "emotion": self.emotion_detected,
                "is_speech": self.is_speech,
            },
            "error": self.error,
        }


class ONNXResultAnalyzer:
    """
    Hook-based analyzer that wraps SenseVoiceSmall to capture intermediate results.
    
    Usage:
        analyzer = ONNXResultAnalyzer(model_dir="iic/SenseVoiceSmall")
        result = analyzer.transcribe_with_analysis(audio_bytes, language="auto")
        print(result.clean_text)
        analyzer.visualize_result(result, save_path="analysis.png")
    """

    # SenseVoice special token ranges
    LANGUAGE_TOKENS = {
        3: "zh", 4: "en", 7: "yue", 11: "ja", 12: "ko", 13: "nospeech",
    }
    EMOTION_TOKENS = {
        0: "NEUTRAL", 1: "HAPPY", 2: "SAD", 3: "ANGRY", 4: "SURPRISED",
    }
    EVENT_TOKENS = {
        14: "withitn", 15: "woitn",  # text normalization
    }

    def __init__(
        self,
        model_dir: str = "iic/SenseVoiceSmall",
        device_id: Union[str, int] = "0",
        quantize: bool = True,
        log_results: bool = True,
        results_log_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the analyzer with wrapped SenseVoiceSmall model.
        
        Args:
            model_dir: Path or ModelScope ID for the model.
            device_id: ONNX Runtime device (-1 for CPU, 0+ for GPU).
            quantize: Use INT8 quantized model.
            log_results: Automatically log results to JSON.
            results_log_path: Custom path for results log file.
        """
        from funasr_onnx import SenseVoiceSmall
        
        self.model_dir = model_dir
        self.device_id = device_id
        self.log_results = log_results
        self.results_log_path = results_log_path or "onnx_transcription_results.jsonl"
        
        print(f"[Analyzer] Loading model from: {model_dir}")
        self.model = SenseVoiceSmall(
            model_dir=model_dir,
            device_id=device_id,
            quantize=quantize,
        )
        
        # Store original methods for hooking
        self._original_extract_feat = self.model.extract_feat
        self._original_infer = self.model.infer
        self._original_call = self.model.__call__
        
        # Storage for captured intermediates
        self._captured: Dict[str, Any] = {}
        self.results_history: List[TranscriptionResult] = []

        # ── ONNX verification probe ──────────────────────────────────────────
        print("\n[Analyzer] === ONNX Runtime Verification ===")
        try:
            session = self.model.ort_infer.session
            print(f"  Model path : {session.get_session_options()}")
            print(f"  Providers  : {session.get_providers()}")
            inputs  = [i.name for i in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            print(f"  Inputs     : {inputs}")
            print(f"  Outputs    : {outputs}")
            # Confirm the model file on disk
            import onnxruntime as ort
            print(f"  ORT version: {ort.__version__}")
        except AttributeError:
            # ort_infer might wrap the session differently
            try:
                print(f"  ort_infer type : {type(self.model.ort_infer)}")
                print(f"  ort_infer attrs: {[a for a in dir(self.model.ort_infer) if not a.startswith('_')]}")
            except Exception as e:
                print(f"  [WARN] Could not inspect ort_infer: {e}")
        print("[Analyzer] ==========================================\n")
        # ────────────────────────────────────────────────────────────────────

    def transcribe_with_analysis(
        self,
        audio_bytes: bytes,
        language: str = "auto",
        use_itn: bool = True,
    ) -> TranscriptionResult:
        self._captured = {}

        try:
            # Step 1: Load audio
            waveform, sr = self._load_and_analyze_audio(audio_bytes)
            audio_meta = self._build_audio_meta(waveform, sr)

            # Step 2: Extract features
            feats, feats_len = self._extract_feat_with_capture([waveform])
            feature_meta = self._build_feature_meta(feats, feats_len)

            # Step 3: Prepare language/textnorm tags as numpy arrays
            textnorm_str = "withitn" if use_itn else "woitn"
            raw_tags = self.model.read_tags(language, textnorm_str)

            # read_tags may return ([lang_id], [norm_id]) or flat ints — normalise defensively
            if isinstance(raw_tags, (list, tuple)) and len(raw_tags) == 2:
                lang_val, norm_val = raw_tags
            else:
                raise ValueError(f"Unexpected read_tags output: {raw_tags!r}")

            language_arr = np.array(
                lang_val if isinstance(lang_val, (list, tuple)) else [lang_val],
                dtype=np.int32,
            )
            textnorm_arr = np.array(
                norm_val if isinstance(norm_val, (list, tuple)) else [norm_val],
                dtype=np.int32,
            )

            # Step 4: Inference
            t_start = time.perf_counter()
            ctc_logits, encoder_out_lens = self._infer_with_capture(
                feats, feats_len, language_arr, textnorm_arr,
            )
            inference_time_ms = (time.perf_counter() - t_start) * 1000

            inference_meta = self._build_inference_meta(
                feats, feats_len, language_arr, textnorm_arr,
                ctc_logits, encoder_out_lens, inference_time_ms,
            )

            # Step 5: CTC decode
            decoding_meta, token_int = self._decode_with_analysis(
                ctc_logits, encoder_out_lens
            )

            # Step 6: Tokens → text
            raw_text = self.model.tokenizer.decode(token_int)

            # Step 7: Post-process
            from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
            clean_text = rich_transcription_postprocess(raw_text)

            # Step 8: Parse special tokens
            lang_detected, emotion_detected, is_speech = self._parse_special_tokens(
                raw_text, token_int
            )

            result = TranscriptionResult(
                audio=audio_meta,
                features=feature_meta,
                inference=inference_meta,
                decoding=decoding_meta,
                raw_text=raw_text,
                clean_text=clean_text,
                language_detected=lang_detected,
                emotion_detected=emotion_detected,
                is_speech=is_speech,
            )

            if self.log_results:
                self._log_result(result)

            self.results_history.append(result)
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()   # print full trace so next error is immediately visible
            error_result = TranscriptionResult(
                audio=AudioMeta(0, 0.0, 0, 0.0, 0.0, False),
                features=FeatureMeta((0, 0), (0, 0), (0, 0, 0), 0),
                inference=InferenceMeta({}, {}, 0.0, "unknown"),
                decoding=DecodingMeta((0, 0), 0, 0, [], 0.0),
                raw_text="",
                clean_text="",
                language_detected="unknown",
                emotion_detected="unknown",
                is_speech=False,
                error=str(e),
            )
            if self.log_results:
                self._log_result(error_result)
            return error_result

    def _load_and_analyze_audio(
        self, audio_bytes: bytes, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """Load audio and return waveform with sample rate. Also stashes waveform for viz."""
        audio_io = io.BytesIO(audio_bytes)
        waveform, sr = librosa.load(audio_io, sr=target_sr, mono=True)
        self._last_waveform = waveform  # needed by visualize_result
        return waveform, sr

    def _build_audio_meta(self, waveform: np.ndarray, sr: int) -> AudioMeta:
        """Analyze audio and build metadata."""
        duration = len(waveform) / sr
        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        
        # Simple VAD: consider speech if RMS > threshold
        has_speech = rms > 0.005
        
        return AudioMeta(
            sample_rate=sr,
            duration_seconds=duration,
            num_samples=len(waveform),
            peak_amplitude=peak,
            rms_energy=rms,
            has_speech=has_speech,
        )

    def _extract_feat_with_capture(
        self, waveform_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using the model's own extract_feat, then capture intermediates."""
        # Use the model's own method to ensure compatibility
        padded_feats, feats_len_arr = self.model.extract_feat(waveform_list)

        # Best-effort capture of intermediate steps (not all funasr_onnx builds expose these)
        fbank_outputs = []
        lfr_outputs = []
        for waveform in waveform_list:
            try:
                fbank_out, _ = self.model.frontend.fbank(waveform)
                fbank_outputs.append(fbank_out.copy())
                lfr_out, _ = self.model.frontend.lfr_cmvn(fbank_out)
                lfr_outputs.append(lfr_out)
            except Exception:
                fbank_outputs.append(None)
                lfr_outputs.append(np.array([]))

        self._captured["fbank_outputs"] = fbank_outputs
        self._captured["lfr_outputs"] = lfr_outputs
        self._captured["padded_feats"] = padded_feats
        self._captured["feats_len"] = feats_len_arr

        return padded_feats, feats_len_arr

    def _build_feature_meta(
        self, feats: np.ndarray, feats_len: np.ndarray
    ) -> FeatureMeta:
        """Build feature metadata."""
        fbank_out = self._captured.get("fbank_outputs", [None])[0]
        fbank_shape = fbank_out.shape if fbank_out is not None else (0, 0)
        
        lfr_out = self._captured.get("lfr_outputs", [np.array([])])[0]
        lfr_shape = lfr_out.shape if lfr_out.size > 0 else (0, 0)
        
        return FeatureMeta(
            fbank_shape=fbank_shape,
            lfr_shape=lfr_shape,
            padded_shape=feats.shape,
            num_frames=int(feats_len[0]) if len(feats_len) > 0 else 0,
        )

    def _infer_with_capture(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ONNX inference and capture input/output shapes."""
        self._captured["model_inputs"] = {
            "feats": feats,
            "feats_len": feats_len,
            "language": language,
            "textnorm": textnorm,
        }

        # Shapes are safe now — all inputs are guaranteed np.ndarray
        try:
            input_names = self.model.ort_infer.get_input_names()
            output_names = self.model.ort_infer.get_output_names()
            self._captured["input_names"] = input_names
            self._captured["output_names"] = output_names
        except Exception:
            pass  # non-fatal; shape logging is best-effort

        outputs = self.model.ort_infer([feats, feats_len, language, textnorm])

        self._captured["raw_outputs"] = outputs
        return outputs[0], outputs[1]

    def _build_inference_meta(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
        ctc_logits: np.ndarray,
        encoder_out_lens: np.ndarray,
        inference_time_ms: float,
    ) -> InferenceMeta:
        """Build inference metadata — all inputs are np.ndarray so .shape is always valid."""
        input_shapes = {
            "feats": tuple(feats.shape),
            "feats_len": tuple(feats_len.shape),
            "language": tuple(language.shape),
            "textnorm": tuple(textnorm.shape),
        }
        output_shapes = {
            "ctc_logits": tuple(ctc_logits.shape),
            "encoder_out_lens": tuple(encoder_out_lens.shape),
        }

        try:
            provider = self.model.ort_infer.session.get_providers()[0]
        except Exception:
            provider = "unknown"

        return InferenceMeta(
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            inference_time_ms=inference_time_ms,
            provider=provider,
        )

    def _decode_with_analysis(
        self, ctc_logits: np.ndarray, encoder_out_lens: np.ndarray
    ) -> Tuple[DecodingMeta, List[int]]:
        """
        Perform CTC decoding with detailed analysis.
        
        Captures:
        - Logit statistics (max, mean, entropy)
        - Blank token ratio
        - Deduplication ratio
        - Token sequence
        """
        b = 0  # First batch item
        x = ctc_logits[b, : encoder_out_lens[b].item(), :]
        
        # Logit statistics
        self._captured["logit_stats"] = {
            "max_logit": float(np.max(x)),
            "mean_logit": float(np.mean(x)),
            "logit_std": float(np.std(x)),
            "top1_confidence": float(np.mean(np.max(x, axis=-1))),
        }
        
        # Top-k analysis per frame
        top5_indices = np.argsort(x, axis=-1)[:, -5:][:, ::-1]
        self._captured["top5_per_frame"] = top5_indices[:10].tolist()  # First 10 frames
        
        # CTC greedy decode
        yseq = np.argmax(x, axis=-1)
        
        # Count blanks
        num_blanks = int(np.sum(yseq == self.model.blank_id))
        total_frames = len(yseq)
        
        # Deduplicate
        mask = np.concatenate(([True], np.diff(yseq) != 0))
        yseq_dedup = yseq[mask]
        
        # Remove blanks
        non_blank_mask = yseq_dedup != self.model.blank_id
        token_int = yseq_dedup[non_blank_mask].tolist()
        
        # Statistics
        num_unique = len(token_int)
        dedup_ratio = len(yseq_dedup) / total_frames if total_frames > 0 else 0
        
        # Frame-level confidence heatmap (first 50 tokens)
        top_tokens = min(50, x.shape[-1])
        self._captured["confidence_map"] = x[:min(20, x.shape[0]), :top_tokens]
        
        return DecodingMeta(
            raw_logits_shape=x.shape,
            num_blank_tokens=num_blanks,
            num_unique_tokens=num_unique,
            token_sequence=token_int,
            dedup_ratio=dedup_ratio,
        ), token_int

    def _parse_special_tokens(
        self, raw_text: str, token_int: List[int]
    ) -> Tuple[str, str, bool]:
        """Extract language, emotion, and speech flag from raw_text tags."""

        LANGUAGE_TAG_MAP = {
            "zh": "zh", "en": "en", "yue": "yue",
            "ja": "ja", "ko": "ko", "nospeech": "nospeech",
        }
        EMOTION_TAG_MAP = {
            "NEUTRAL": "NEUTRAL", "HAPPY": "HAPPY", "SAD": "SAD",
            "ANGRY": "ANGRY", "SURPRISED": "SURPRISED",
        }

        lang = "unknown"
        emotion = "unknown"
        is_speech = True

        # Match <|tag|> tokens in order
        tags = re.findall(r"<\|([^|]+)\|>", raw_text)
        for tag in tags:
            if tag in LANGUAGE_TAG_MAP:
                lang = LANGUAGE_TAG_MAP[tag]
            if tag in EMOTION_TAG_MAP:
                emotion = EMOTION_TAG_MAP[tag]
            if tag == "nospeech":
                is_speech = False

        if lang == "nospeech":
            is_speech = False

        return lang, emotion, is_speech

    def _log_result(self, result: TranscriptionResult) -> None:
        """Append result to JSON lines log file."""
        log_entry = result.to_dict()
        with open(self.results_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def visualize_result(
        self,
        result: TranscriptionResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive visualization of the transcription analysis.
        
        Shows:
        1. Audio waveform with energy
        2. Feature (fbank) spectrogram
        3. CTC logit heatmap
        4. Frame confidence plot
        5. Summary statistics table
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
        
        # 1. Audio waveform
        ax1 = fig.add_subplot(gs[0, :])
        if hasattr(self, '_last_waveform'):
            waveform = self._last_waveform
            time_axis = np.linspace(0, result.audio.duration_seconds, len(waveform))
            ax1.plot(time_axis, waveform, color='steelblue', alpha=0.8, linewidth=0.5)
            ax1.set_title("Audio Waveform", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=result.audio.rms_energy, color='red', linestyle='--', 
                       label=f'RMS: {result.audio.rms_energy:.4f}')
            ax1.legend()
        
        # 2. FBank features
        ax2 = fig.add_subplot(gs[1, 0])
        fbank = self._captured.get("fbank_outputs", [None])[0]
        if fbank is not None:
            im = ax2.imshow(fbank.T, aspect='auto', origin='lower', cmap='magma')
            ax2.set_title(f"FBank Features\n{fbank.shape}", fontsize=11)
            ax2.set_xlabel("Frames")
            ax2.set_ylabel("Mel Bins")
            plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. LFR features
        ax3 = fig.add_subplot(gs[1, 1])
        lfr = self._captured.get("lfr_outputs", [None])[0]
        if lfr is not None:
            im3 = ax3.imshow(lfr.T, aspect='auto', origin='lower', cmap='viridis')
            ax3.set_title(f"LFR+CMVN Features\n{lfr.shape}", fontsize=11)
            ax3.set_xlabel("Frames")
            ax3.set_ylabel("Feature Dim")
            plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. CTC logit confidence
        ax4 = fig.add_subplot(gs[2, 0])
        conf_map = self._captured.get("confidence_map")
        if conf_map is not None:
            im4 = ax4.imshow(conf_map.T, aspect='auto', origin='lower', cmap='YlOrRd')
            ax4.set_title(f"CTC Logits (Top 50 tokens, first 20 frames)", fontsize=11)
            ax4.set_xlabel("Frames")
            ax4.set_ylabel("Token Index")
            plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        # 5. Frame confidence
        ax5 = fig.add_subplot(gs[2, 1])
        if conf_map is not None:
            max_per_frame = np.max(conf_map, axis=1)
            ax5.bar(range(len(max_per_frame)), max_per_frame, color='steelblue', alpha=0.7)
            ax5.axhline(y=0.5, color='red', linestyle='--', label='0.5 threshold')
            ax5.set_title(f"Frame Confidence\nMean: {np.mean(max_per_frame):.3f}", fontsize=11)
            ax5.set_xlabel("Frame")
            ax5.set_ylabel("Max Logit")
            ax5.set_ylim(0, 1)
            ax5.legend()
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        summary_text = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   TRANSCRIPTION ANALYSIS                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Audio                                                      ║
        ║    Duration: {result.audio.duration_seconds:.3f}s           ║
        ║    Sample Rate: {result.audio.sample_rate} Hz               ║
        ║    RMS Energy: {result.audio.rms_energy:.4f}                ║
        ║    Has Speech: {result.audio.has_speech}                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Features                                                   ║
        ║    FBank: {result.features.fbank_shape}                     ║
        ║    LFR: {result.features.lfr_shape}                         ║
        ║    Padded: {result.features.padded_shape}                   ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Inference                                                  ║
        ║    Provider: {result.inference.provider}                    ║
        ║    Time: {result.inference.inference_time_ms:.1f}ms         ║
        ║    Output Shape: {result.decoding.raw_logits_shape}         ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Decoding                                                   ║
        ║    Blank Tokens: {result.decoding.num_blank_tokens}         ║
        ║    Unique Tokens: {result.decoding.num_unique_tokens}       ║
        ║    Dedup Ratio: {result.decoding.dedup_ratio:.3f}           ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Transcription                                              ║
        ║    Raw: {result.raw_text[:80]}...                           ║
        ║    Clean: "{result.clean_text}"                             ║
        ║    Language: {result.language_detected}                     ║
        ║    Emotion: {result.emotion_detected}                       ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        
        ax6.text(0, 0.5, summary_text, fontfamily='monospace', fontsize=9,
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f"ONNX Transcription Analysis - {time.strftime('%Y-%m-%d %H:%M:%S')}", 
                    fontsize=14, fontweight="bold")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyzer] Visualization saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all transcriptions."""
        if not self.results_history:
            return {"error": "No results available"}
        
        successful = [r for r in self.results_history if r.error is None]
        failed = [r for r in self.results_history if r.error is not None]
        
        inference_times = [r.inference.inference_time_ms for r in successful]
        audio_durations = [r.audio.duration_seconds for r in successful]
        rtfs = [t / (d * 1000) for t, d in zip(inference_times, audio_durations)] if audio_durations else []
        
        return {
            "total_requests": len(self.results_history),
            "successful": len(successful),
            "failed": len(failed),
            "avg_inference_time_ms": np.mean(inference_times) if inference_times else 0,
            "avg_audio_duration_s": np.mean(audio_durations) if audio_durations else 0,
            "avg_rtf": np.mean(rtfs) if rtfs else 0,
            "p95_inference_time_ms": np.percentile(inference_times, 95) if inference_times else 0,
            "languages_detected": list(set(r.language_detected for r in successful)),
        }

    def reset_history(self) -> None:
        """Clear result history."""
        self.results_history.clear()


# Convenience function for quick analysis
def quick_analyze(
    audio_path: str,
    model_dir: str = "iic/SenseVoiceSmall",
    language: str = "auto",
    save_plot: Optional[str] = None,
) -> TranscriptionResult:
    """
    Quick one-shot analysis of an audio file.

    Args:
        audio_path: Path to audio file.
        model_dir: Model directory.
        language: Language hint.
        save_plot: Path to save visualization PNG.

    Returns:
        TranscriptionResult with full analysis.
    """
    analyzer = ONNXResultAnalyzer(model_dir=model_dir, log_results=False)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = analyzer.transcribe_with_analysis(audio_bytes, language=language)

    duration_s = result.audio.duration_seconds
    inf_ms = result.inference.inference_time_ms
    rtf_str = f"{inf_ms / (duration_s * 1000):.4f}" if duration_s > 0 else "N/A"

    logits_frames = result.decoding.raw_logits_shape[0]
    blank_ratio_str = (
        f"{result.decoding.num_blank_tokens / logits_frames:.3f}"
        if logits_frames > 0
        else "N/A"
    )

    print(f"\n{'='*60}")
    print(f"TRANSCRIPTION RESULT")
    print(f"{'='*60}")
    print(f"Clean Text:  {result.clean_text}")
    print(f"Raw Text:    {result.raw_text}")
    print(f"Language:    {result.language_detected}")
    print(f"Emotion:     {result.emotion_detected}")
    print(f"Inference:   {inf_ms:.1f}ms")
    print(f"Audio:       {duration_s:.3f}s")
    print(f"RTF:         {rtf_str}")
    print(f"Blank Ratio: {blank_ratio_str}")
    print(f"{'='*60}")

    if result.error:
        print(f"[ERROR] {result.error}")

    if save_plot:
        analyzer.visualize_result(result, save_path=save_plot)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\example\en.mp3"

    result = quick_analyze(
        audio_file,
        language="auto",
        save_plot="transcription_analysis.png",
    )
    # .to_dict() for JSON-serialisable output
    print(f"Quick Analysis Result:\n{json.dumps(result.to_dict(), indent=2, ensure_ascii=False)}")
