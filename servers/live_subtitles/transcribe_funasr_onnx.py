# servers/live_subtitles/transcribe_funasr_onnx.py
"""
ONNX-based SenseVoice transcription module for live subtitles.
Uses funasr_onnx for lightweight, fast inference without PyTorch dependency.

Enhanced with comprehensive result analysis and visualization capabilities.
"""
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import librosa
import numpy as np
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SupportedLanguage = Literal["auto", "zh", "en", "ja", "ko", "yue"]


@dataclass
class AudioMetadata:
    """Detailed audio analysis metadata."""
    sample_rate: int
    duration_seconds: float
    num_samples: int
    peak_amplitude: float
    rms_energy: float
    zero_crossing_rate: float
    has_speech: bool
    silence_ratio: float = 0.0


@dataclass
class FeatureMetadata:
    """Feature extraction pipeline metadata."""
    fbank_shape: Tuple[int, int]
    lfr_shape: Tuple[int, int]
    padded_shape: Tuple[int, int, int]
    num_valid_frames: int
    lfr_stack_factor: int = 7


@dataclass
class InferenceMetadata:
    """ONNX inference metadata."""
    input_names: List[str]
    output_names: List[str]
    input_shapes: Dict[str, Tuple[int, ...]]
    output_shapes: Dict[str, Tuple[int, ...]]
    inference_time_ms: float
    execution_provider: str
    num_threads: int
    model_input_dtype: str = "float32"


@dataclass
class DecodingMetadata:
    """CTC decoding analysis metadata."""
    raw_logits_shape: Tuple[int, int]
    vocab_size: int
    num_encoder_frames: int
    num_blank_tokens: int
    num_unique_tokens: int
    token_sequence: List[int]
    top1_confidence_mean: float
    top5_confidence_mean: float
    blank_ratio: float
    dedup_ratio: float
    entropy_mean: float


@dataclass
class TranscriptionResult:
    """
    Complete transcription result with full analysis pipeline data.
    
    This captures every intermediate result from audio loading through
    final text output, enabling comprehensive debugging and monitoring.
    """
    # Core output
    raw_text: str
    clean_text: str
    
    # Analysis metadata
    audio: AudioMetadata
    features: FeatureMetadata
    inference: InferenceMetadata
    decoding: DecodingMetadata
    
    # Special tokens
    language_detected: str
    emotion_detected: str
    is_speech: bool
    text_normalization: str  # "withitn" or "woitn"
    
    # Timing
    total_time_ms: float
    timestamp: float = field(default_factory=time.time)
    
    # Error handling
    error: Optional[str] = None
    warning: Optional[str] = None
    
    # Raw captured data (for visualization, not serialized by default)
    _captured_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self, include_raw_data: bool = False) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON logging.
        
        Args:
            include_raw_data: If True, include raw arrays (large, use sparingly).
        
        Returns:
            Dictionary suitable for JSON serialization.
        """
        result = {
            "timestamp": self.timestamp,
            "total_time_ms": round(self.total_time_ms, 2),
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "duration_s": round(self.audio.duration_seconds, 3),
                "num_samples": self.audio.num_samples,
                "peak_amplitude": round(self.audio.peak_amplitude, 4),
                "rms_energy": round(self.audio.rms_energy, 4),
                "zero_crossing_rate": round(self.audio.zero_crossing_rate, 2),
                "has_speech": self.audio.has_speech,
                "silence_ratio": round(self.audio.silence_ratio, 3),
            },
            "features": {
                "fbank_shape": list(self.features.fbank_shape),
                "lfr_shape": list(self.features.lfr_shape),
                "padded_shape": list(self.features.padded_shape),
                "num_valid_frames": self.features.num_valid_frames,
                "lfr_stack_factor": self.features.lfr_stack_factor,
            },
            "inference": {
                "input_shapes": {k: list(v) for k, v in self.inference.input_shapes.items()},
                "output_shapes": {k: list(v) for k, v in self.inference.output_shapes.items()},
                "time_ms": round(self.inference.inference_time_ms, 2),
                "provider": self.inference.execution_provider,
                "num_threads": self.inference.num_threads,
            },
            "decoding": {
                "raw_logits_shape": list(self.decoding.raw_logits_shape),
                "vocab_size": self.decoding.vocab_size,
                "num_encoder_frames": self.decoding.num_encoder_frames,
                "num_blank_tokens": self.decoding.num_blank_tokens,
                "num_unique_tokens": self.decoding.num_unique_tokens,
                "top1_confidence": round(self.decoding.top1_confidence_mean, 4),
                "top5_confidence": round(self.decoding.top5_confidence_mean, 4),
                "blank_ratio": round(self.decoding.blank_ratio, 3),
                "dedup_ratio": round(self.decoding.dedup_ratio, 3),
                "entropy_mean": round(self.decoding.entropy_mean, 3),
            },
            "transcription": {
                "raw_text": self.raw_text,
                "clean_text": self.clean_text,
                "language": self.language_detected,
                "emotion": self.emotion_detected,
                "is_speech": self.is_speech,
                "text_normalization": self.text_normalization,
            },
            "error": self.error,
            "warning": self.warning,
        }
        
        if include_raw_data and self._captured_data:
            # Convert numpy arrays to lists for JSON serialization
            raw_data = {}
            for key, value in self._captured_data.items():
                if isinstance(value, np.ndarray):
                    raw_data[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "data": value.flatten()[:1000].tolist(),  # Limit to first 1000 elements
                    }
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    raw_data[key] = [
                        {"shape": list(v.shape), "data": v.flatten()[:500].tolist()}
                        for v in value[:3]  # Limit to first 3 items
                    ]
            result["raw_data"] = raw_data
        
        return result
    
    def to_json(self, include_raw_data: bool = False, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(include_raw_data), indent=indent, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the result."""
        separator = "=" * 65
        print(f"\n{separator}")
        print(f"  ONNX TRANSCRIPTION RESULT SUMMARY")
        print(f"{separator}")
        
        if self.error:
            print(f"  ❌ ERROR: {self.error}")
            print(f"{separator}\n")
            return
        
        if self.warning:
            print(f"  ⚠️  WARNING: {self.warning}")
        
        print(f"  📝 Clean Text:   \"{self.clean_text}\"")
        print(f"  📄 Raw Text:     \"{self.raw_text[:100]}{'...' if len(self.raw_text) > 100 else ''}\"")
        print(f"  🌍 Language:     {self.language_detected}")
        print(f"  😊 Emotion:      {self.emotion_detected}")
        print(f"  🗣️  Is Speech:    {self.is_speech}")
        print(f"  📋 Text Norm:    {self.text_normalization}")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  ⏱️  Total Time:   {self.total_time_ms:.1f} ms")
        print(f"  🚀 Inference:    {self.inference.inference_time_ms:.1f} ms")
        print(f"  🎵 Audio:        {self.audio.duration_seconds:.3f}s @ {self.audio.sample_rate}Hz")
        print(f"  ⚡ RTF:          {self.inference.inference_time_ms / (self.audio.duration_seconds * 1000):.4f}")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  🔢 Encoder Frames:  {self.decoding.num_encoder_frames}")
        print(f"  🎯 Top-1 Conf:      {self.decoding.top1_confidence_mean:.3f}")
        print(f"  🎯 Top-5 Conf:      {self.decoding.top5_confidence_mean:.3f}")
        print(f"  📊 Entropy:         {self.decoding.entropy_mean:.3f}")
        print(f"  ⬜ Blank Ratio:     {self.decoding.blank_ratio:.3f}")
        print(f"  🔄 Dedup Ratio:     {self.decoding.dedup_ratio:.3f}")
        print(f"  🏷️  Unique Tokens:   {self.decoding.num_unique_tokens}")
        print(f"{separator}\n")


class SenseVoiceTranscriber:
    """
    Reusable ONNX-based transcriber with comprehensive result analysis.
    
    Features:
    - Direct numpy array input (no temp files needed)
    - Quantized INT8 model support for speed
    - Multi-language: zh, en, ja, ko, yue, auto
    - Inverse Text Normalization (ITN) via textnorm parameter
    - Full pipeline result capture for debugging and monitoring
    - Optional visualization of all intermediate results
    
    Usage:
        transcriber = SenseVoiceTranscriber(model_dir="iic/SenseVoiceSmall")
        result = transcriber.transcribe_with_analysis(audio_bytes, language="zh")
        result.print_summary()
        transcriber.visualize_result(result, "analysis.png")
    """
    
    # SenseVoice special token mappings
    LANGUAGE_TOKEN_MAP = {
        3: "zh", 4: "en", 7: "yue", 11: "ja", 12: "ko", 13: "nospeech",
        24884: "zh", 24885: "en", 24888: "yue", 24892: "ja", 24896: "ko", 24992: "nospeech",
    }
    EMOTION_TOKEN_MAP = {
        0: "NEUTRAL", 1: "HAPPY", 2: "SAD", 3: "ANGRY",
    }
    TEXTNORM_TOKEN_MAP = {
        14: "withitn", 15: "woitn",
        25016: "withitn", 25017: "woitn",
    }
    
    def __init__(
        self,
        model_dir: str = "iic/SenseVoiceSmall",
        device_id: Union[str, int] = "0",
        quantize: bool = True,
        batch_size: int = 1,
        intra_op_num_threads: int = 4,
        enable_analysis: bool = True,
        results_log_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the ONNX transcriber with analysis capabilities.
        
        Args:
            model_dir: ModelScope model ID or local path to model directory.
            device_id: Device ID for ONNX Runtime (-1 for CPU, 0+ for GPU).
            quantize: Use INT8 quantized model if available (faster, smaller).
            batch_size: Maximum batch size for inference.
            intra_op_num_threads: Threads for ONNX Runtime intra-op parallelism.
            enable_analysis: Capture full intermediate results for analysis.
            results_log_path: Path to JSONL file for logging results.
        """
        self.model_dir = model_dir
        self.device_id = str(device_id)
        self.quantize = quantize
        self.batch_size = batch_size
        self.intra_op_num_threads = intra_op_num_threads
        self.enable_analysis = enable_analysis
        self.results_log_path = results_log_path
        
        print(f"[SenseVoiceTranscriber] Loading ONNX model from: {model_dir}")
        print(f"  Device: {'GPU:' + self.device_id if self.device_id != '-1' else 'CPU'}")
        print(f"  Quantized: {quantize}")
        print(f"  Threads: {intra_op_num_threads}")
        print(f"  Analysis: {enable_analysis}")
        
        self.model = SenseVoiceSmall(
            model_dir=model_dir,
            batch_size=batch_size,
            device_id=device_id,
            quantize=quantize,
            intra_op_num_threads=intra_op_num_threads,
        )
        
        # Verify model loaded correctly
        self._verify_model_loaded()
        
        # Initialize result history
        self.results_history: List[TranscriptionResult] = []
        
        print("[SenseVoiceTranscriber] Model loaded successfully.")
    
    def _verify_model_loaded(self) -> None:
        """Verify the ONNX model is properly loaded and accessible."""
        try:
            input_names = self.model.ort_infer.get_input_names()
            output_names = self.model.ort_infer.get_output_names()
            provider = self.model.ort_infer.session.get_providers()[0]
            print(f"  Inputs: {input_names}")
            print(f"  Outputs: {output_names}")
            print(f"  Provider: {provider}")
        except Exception as e:
            print(f"  Warning: Could not verify model details: {e}")
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: SupportedLanguage = "auto",
        use_itn: bool = True,
    ) -> str:
        """
        Transcribe audio from raw bytes (simple interface).
        
        Args:
            audio_bytes: Raw audio data as bytes (WAV, MP3, FLAC, etc.).
            language: Target language hint ("auto", "zh", "en", "ja", "ko", "yue").
            use_itn: Apply Inverse Text Normalization if True.
            
        Returns:
            The transcribed clean text string.
        """
        result = self.transcribe_with_analysis(audio_bytes, language, use_itn)
        return result.clean_text
    
    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: SupportedLanguage = "auto",
        use_itn: bool = True,
    ) -> str:
        """
        Transcribe audio from a file path (simple interface).
        
        Args:
            file_path: Path to audio file.
            language: Target language hint.
            use_itn: Apply Inverse Text Normalization if True.
            
        Returns:
            The transcribed clean text string.
        """
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe_bytes(audio_bytes, language, use_itn)
    
    def transcribe_with_analysis(
        self,
        audio_bytes: bytes,
        language: SupportedLanguage = "auto",
        use_itn: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio with full analysis pipeline.
        
        This method captures every intermediate result from the processing
        pipeline, providing detailed metadata for debugging, monitoring,
        and quality analysis.
        
        Args:
            audio_bytes: Raw audio file bytes.
            language: Language hint ("auto", "zh", "en", "ja", "ko", "yue").
            use_itn: Apply Inverse Text Normalization if True.
            
        Returns:
            TranscriptionResult with full analysis data.
        """
        t_total_start = time.perf_counter()
        captured: Dict[str, Any] = {}
        warning: Optional[str] = None
        
        try:
            # ============================================================
            # Step 1: Load and analyze audio
            # ============================================================
            waveform, sr = self._load_audio_from_bytes(audio_bytes)
            audio_meta = self._analyze_audio(waveform, sr)
            captured["waveform"] = waveform
            
            if not audio_meta.has_speech:
                warning = "Low audio energy detected, possible silence or noise"
            
            # ============================================================
            # Step 2: Extract features (FBank -> LFR -> CMVN)
            # ============================================================
            feats, feats_len, feature_captured = self._extract_features_with_capture(
                [waveform]
            )
            captured.update(feature_captured)
            feature_meta = self._build_feature_metadata(feats, feats_len)
            
            # ============================================================
            # Step 3: Prepare language and textnorm tags
            # ============================================================
            textnorm_str = "withitn" if use_itn else "woitn"
            language_list, textnorm_list = self.model.read_tags(language, textnorm_str)
            
            language_array = np.array(language_list, dtype=np.int32)
            textnorm_array = np.array(textnorm_list, dtype=np.int32)
            
            # ============================================================
            # Step 4: ONNX inference with timing
            # ============================================================
            t_infer_start = time.perf_counter()
            ctc_logits, encoder_out_lens = self.model.infer(
                feats, feats_len, language_array, textnorm_array
            )
            inference_time_ms = (time.perf_counter() - t_infer_start) * 1000
            
            captured["ctc_logits"] = ctc_logits
            captured["encoder_out_lens"] = encoder_out_lens
            
            inference_meta = self._build_inference_metadata(
                feats, feats_len, language_array, textnorm_array,
                ctc_logits, encoder_out_lens, inference_time_ms,
            )
            
            # ============================================================
            # Step 5: CTC decoding with detailed analysis
            # ============================================================
            decoding_meta, token_int, decoding_captured = self._decode_with_analysis(
                ctc_logits, encoder_out_lens
            )
            captured.update(decoding_captured)
            
            # ============================================================
            # Step 6: Token-to-text decoding
            # ============================================================
            raw_text = self.model.tokenizer.decode(token_int)
            
            # ============================================================
            # Step 7: Post-processing
            # ============================================================
            clean_text = rich_transcription_postprocess(raw_text)
            
            # ============================================================
            # Step 8: Parse special tokens (language, emotion, etc.)
            # ============================================================
            lang_detected, emotion_detected, is_speech = self._parse_special_tokens(
                raw_text, token_int
            )
            
            # ============================================================
            # Build result
            # ============================================================
            total_time_ms = (time.perf_counter() - t_total_start) * 1000
            
            result = TranscriptionResult(
                raw_text=raw_text,
                clean_text=clean_text,
                audio=audio_meta,
                features=feature_meta,
                inference=inference_meta,
                decoding=decoding_meta,
                language_detected=lang_detected,
                emotion_detected=emotion_detected,
                is_speech=is_speech,
                text_normalization=textnorm_str,
                total_time_ms=total_time_ms,
                warning=warning,
                _captured_data=captured,
            )
            
        except Exception as e:
            total_time_ms = (time.perf_counter() - t_total_start) * 1000
            
            # Build error result with minimal metadata
            result = TranscriptionResult(
                raw_text="",
                clean_text="",
                audio=AudioMetadata(0, 0.0, 0, 0.0, 0.0, 0.0, False),
                features=FeatureMetadata((0, 0), (0, 0), (0, 0, 0), 0),
                inference=InferenceMetadata([], [], {}, {}, 0.0, "unknown", 0),
                decoding=DecodingMetadata((0, 0), 0, 0, 0, 0, [], 0.0, 0.0, 0.0, 0.0, 0.0),
                language_detected="unknown",
                emotion_detected="unknown",
                is_speech=False,
                text_normalization="woitn",
                total_time_ms=total_time_ms,
                error=str(e),
            )
        
        # Log and store
        self._log_result(result)
        self.results_history.append(result)
        
        return result
    
    def _load_audio_from_bytes(
        self,
        audio_bytes: bytes,
        target_sr: int = 16000,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes into a numpy array.
        
        Uses librosa which supports MP3, WAV, FLAC, and more via soundfile/audioread.
        
        Args:
            audio_bytes: Raw audio file bytes.
            target_sr: Target sample rate (SenseVoice expects 16kHz).
            
        Returns:
            Tuple of (waveform_array, sample_rate).
        """
        audio_io = io.BytesIO(audio_bytes)
        waveform, sr = librosa.load(audio_io, sr=target_sr, mono=True)
        return waveform, sr
    
    @staticmethod
    def _analyze_audio(waveform: np.ndarray, sr: int) -> AudioMetadata:
        """
        Analyze audio waveform and extract metadata.
        
        Computes:
        - Duration, peak amplitude, RMS energy
        - Zero-crossing rate (voice/unvoiced indicator)
        - Simple energy-based speech detection
        - Silence ratio (frames below threshold)
        
        Args:
            waveform: 1D numpy array of audio samples.
            sr: Sample rate in Hz.
            
        Returns:
            AudioMetadata with analysis results.
        """
        duration = len(waveform) / sr
        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        
        # Zero-crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(waveform)))) / (2 * len(waveform)))
        
        # Simple energy-based VAD
        energy_threshold = 0.005
        has_speech = rms > energy_threshold
        
        # Silence ratio using frame-based analysis
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        if len(waveform) >= frame_length:
            num_frames = (len(waveform) - frame_length) // hop_length + 1
            silent_frames = 0
            for i in range(num_frames):
                start = i * hop_length
                frame = waveform[start:start + frame_length]
                frame_rms = np.sqrt(np.mean(frame ** 2))
                if frame_rms < energy_threshold:
                    silent_frames += 1
            silence_ratio = silent_frames / num_frames if num_frames > 0 else 0.0
        else:
            silence_ratio = 1.0 if rms < energy_threshold else 0.0
        
        return AudioMetadata(
            sample_rate=sr,
            duration_seconds=duration,
            num_samples=len(waveform),
            peak_amplitude=peak,
            rms_energy=rms,
            zero_crossing_rate=zcr,
            has_speech=has_speech,
            silence_ratio=silence_ratio,
        )
    
    def _extract_features_with_capture(
        self, waveform_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extract features (FBank, LFR, CMVN) and capture intermediate outputs.
        
        Pipeline:
        1. FBank: Mel-scale filterbank energies
        2. LFR: Low Frame Rate stacking (7 frames -> 1)
        3. CMVN: Cepstral Mean and Variance Normalization
        4. Padding: Zero-pad to max length in batch
        
        Args:
            waveform_list: List of waveform arrays.
            
        Returns:
            Tuple of (padded_feats, feats_len, captured_data).
        """
        captured = {}
        feats, feats_len = [], []
        fbank_outputs = []
        
        for i, waveform in enumerate(waveform_list):
            # FBank extraction
            speech, _ = self.model.frontend.fbank(waveform)
            fbank_outputs.append(speech.copy() if speech is not None else None)
            
            if speech is None or speech.size == 0:
                raise ValueError(f"Empty speech detected for waveform {i}")
            
            # LFR + CMVN
            feat, feat_len = self.model.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)
        
        # Pad to max length
        max_len = np.max(feats_len)
        padded_feats = self.model.pad_feats(feats, max_len)
        feats_len_arr = np.array(feats_len).astype(np.int32)
        
        captured["fbank_outputs"] = fbank_outputs
        captured["lfr_outputs"] = [f.copy() for f in feats]
        captured["padded_feats"] = padded_feats.copy()
        
        return padded_feats, feats_len_arr, captured
    
    @staticmethod
    def _build_feature_metadata(
        feats: np.ndarray, feats_len: np.ndarray
    ) -> FeatureMetadata:
        """Build FeatureMetadata from feature extraction results."""
        return FeatureMetadata(
            fbank_shape=(0, 0),  # Will be updated if fbank data is available
            lfr_shape=(int(feats_len[0]), feats.shape[2]) if len(feats_len) > 0 else (0, 0),
            padded_shape=feats.shape,
            num_valid_frames=int(feats_len[0]) if len(feats_len) > 0 else 0,
        )
    
    def _build_inference_metadata(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
        ctc_logits: np.ndarray,
        encoder_out_lens: np.ndarray,
        inference_time_ms: float,
    ) -> InferenceMetadata:
        """Build InferenceMetadata from model inference results."""
        input_names = self.model.ort_infer.get_input_names()
        output_names = self.model.ort_infer.get_output_names()
        provider = self.model.ort_infer.session.get_providers()[0]
        
        return InferenceMetadata(
            input_names=input_names,
            output_names=output_names,
            input_shapes={
                input_names[0]: feats.shape,
                input_names[1]: feats_len.shape,
                input_names[2]: language.shape,
                input_names[3]: textnorm.shape,
            },
            output_shapes={
                output_names[0]: ctc_logits.shape,
                output_names[1]: encoder_out_lens.shape,
            },
            inference_time_ms=inference_time_ms,
            execution_provider=provider,
            num_threads=self.intra_op_num_threads,
        )
    
    def _decode_with_analysis(
        self, ctc_logits: np.ndarray, encoder_out_lens: np.ndarray
    ) -> Tuple[DecodingMetadata, List[int], Dict[str, Any]]:
        """
        Perform CTC greedy decoding with detailed statistical analysis.
        
        CTC Decoding Steps:
        1. Slice to valid encoder frames
        2. Argmax to get best token per frame
        3. Remove consecutive duplicates (CTC collapse)
        4. Remove blank tokens
        
        Analysis includes:
        - Confidence statistics (top-1, top-5)
        - Entropy per frame
        - Blank token ratio
        - Deduplication efficiency
        
        Args:
            ctc_logits: Raw logits from encoder, shape (B, T_enc, V).
            encoder_out_lens: Valid frame counts, shape (B,).
            
        Returns:
            Tuple of (DecodingMetadata, token_list, captured_data).
        """
        captured = {}
        b = 0  # First batch item
        
        # Extract valid frames
        num_encoder_frames = int(encoder_out_lens[b].item())
        x = ctc_logits[b, :num_encoder_frames, :]
        vocab_size = x.shape[1]
        
        # Apply softmax for probability analysis
        x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerically stable
        x_probs = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        
        # Confidence statistics
        sorted_indices = np.argsort(x, axis=-1)[:, ::-1]
        top1_confs = x_probs[np.arange(x.shape[0]), sorted_indices[:, 0]]
        top5_indices = sorted_indices[:, :5]
        top5_confs = np.array([
            np.sum(x_probs[i, top5_indices[i]]) for i in range(x.shape[0])
        ])
        
        top1_confidence_mean = float(np.mean(top1_confs))
        top5_confidence_mean = float(np.mean(top5_confs))
        
        # Entropy per frame
        entropy_per_frame = -np.sum(x_probs * np.log(x_probs + 1e-10), axis=-1)
        entropy_mean = float(np.mean(entropy_per_frame))
        
        captured["frame_confidence"] = top1_confs[:100].tolist()  # First 100 frames
        captured["frame_entropy"] = entropy_per_frame[:100].tolist()
        captured["top5_indices"] = top5_indices[:20].tolist()  # First 20 frames
        
        # CTC greedy decoding
        yseq = np.argmax(x, axis=-1)
        
        # Blank token analysis
        num_blanks = int(np.sum(yseq == self.model.blank_id))
        blank_ratio = num_blanks / num_encoder_frames if num_encoder_frames > 0 else 0
        
        # Deduplicate consecutive repeats
        keep_mask = np.concatenate(([True], np.diff(yseq) != 0))
        yseq_dedup = yseq[keep_mask]
        
        # Remove blank tokens
        non_blank_mask = yseq_dedup != self.model.blank_id
        token_int = yseq_dedup[non_blank_mask].tolist()
        
        # Deduplication statistics
        dedup_ratio = len(yseq_dedup) / num_encoder_frames if num_encoder_frames > 0 else 0
        
        captured["raw_token_sequence"] = yseq[:min(200, len(yseq))].tolist()
        captured["decoded_tokens"] = token_int
        
        decoding_meta = DecodingMetadata(
            raw_logits_shape=x.shape,
            vocab_size=vocab_size,
            num_encoder_frames=num_encoder_frames,
            num_blank_tokens=num_blanks,
            num_unique_tokens=len(token_int),
            token_sequence=token_int,
            top1_confidence_mean=top1_confidence_mean,
            top5_confidence_mean=top5_confidence_mean,
            blank_ratio=blank_ratio,
            dedup_ratio=dedup_ratio,
            entropy_mean=entropy_mean,
        )
        
        return decoding_meta, token_int, captured
    
    def _parse_special_tokens(
        self, raw_text: str, token_int: List[int]
    ) -> Tuple[str, str, bool]:
        """
        Parse SenseVoice special tokens to extract metadata.
        
        SenseVoice outputs special tokens for:
        - Language identification: <|zh|>, <|en|>, <|ja|>, etc.
        - Emotion recognition: <|NEUTRAL|>, <|HAPPY|>, etc.
        - Speech detection: <|nospeech|> indicates non-speech audio
        - Text normalization: <|withitn|>, <|woitn|>
        
        Args:
            raw_text: Raw decoded text with special tokens.
            token_int: Decoded token IDs.
            
        Returns:
            Tuple of (language, emotion, is_speech).
        """
        lang = "unknown"
        emotion = "unknown"
        is_speech = True
        
        # Check token IDs against known mappings
        for tid in token_int:
            if tid in self.LANGUAGE_TOKEN_MAP:
                mapped_lang = self.LANGUAGE_TOKEN_MAP[tid]
                if mapped_lang == "nospeech":
                    is_speech = False
                else:
                    lang = mapped_lang
            if tid in self.EMOTION_TOKEN_MAP:
                emotion = self.EMOTION_TOKEN_MAP[tid]
        
        # Also check text for special token patterns
        if "<|nospeech|>" in raw_text:
            is_speech = False
            lang = "nospeech"
        
        # Extract language from text pattern
        import re
        lang_match = re.search(r'<\|(\w+)\|>', raw_text)
        if lang_match and lang == "unknown":
            extracted = lang_match.group(1)
            if extracted.lower() in ["zh", "en", "ja", "ko", "yue"]:
                lang = extracted.lower()
        
        return lang, emotion, is_speech
    
    def _log_result(self, result: TranscriptionResult) -> None:
        """Log transcription result to JSON lines file if configured."""
        if not self.results_log_path:
            return
        
        try:
            log_entry = result.to_dict(include_raw_data=False)
            with open(self.results_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SenseVoiceTranscriber] Warning: Failed to log result: {e}")
    
    def visualize_result(
        self,
        result: TranscriptionResult,
        save_path: Optional[str] = None,
        show_plot: bool = False,
    ) -> Optional[str]:
        """
        Create comprehensive visualization of the transcription pipeline.
        
        Generates a multi-panel figure showing:
        1. Audio waveform with energy markers
        2. FBank spectrogram
        3. LFR+CMVN features
        4. CTC frame confidence over time
        5. Per-frame entropy
        6. Summary statistics
        
        Args:
            result: TranscriptionResult from transcribe_with_analysis.
            save_path: Path to save the visualization (PNG).
            show_plot: Display the plot interactively.
            
        Returns:
            Path to saved image if save_path provided, else None.
            
        Raises:
            ImportError: If matplotlib is not available.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )
        
        if result.error:
            print(f"[Visualize] Cannot visualize result with error: {result.error}")
            return None
        
        captured = result._captured_data
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)
        
        fig.suptitle(
            f"SenseVoice ONNX Transcription Analysis\n"
            f"Text: \"{result.clean_text[:80]}{'...' if len(result.clean_text) > 80 else ''}\"",
            fontsize=14, fontweight="bold"
        )
        
        # Panel 1: Audio waveform
        ax1 = fig.add_subplot(gs[0, :])
        waveform = captured.get("waveform")
        if waveform is not None:
            time_axis = np.linspace(0, result.audio.duration_seconds, len(waveform))
            ax1.plot(time_axis, waveform, color='#2E86AB', alpha=0.85, linewidth=0.5)
            ax1.axhline(y=result.audio.rms_energy, color='#D64045', linestyle='--', 
                       linewidth=1.5, label=f'RMS Energy: {result.audio.rms_energy:.4f}')
            ax1.axhline(y=-result.audio.rms_energy, color='#D64045', linestyle='--', linewidth=1.5)
            ax1.fill_between(time_axis, -result.audio.rms_energy, result.audio.rms_energy, 
                            alpha=0.15, color='#D64045')
            ax1.set_title(f"Audio Waveform ({result.audio.duration_seconds:.2f}s, "
                         f"Peak: {result.audio.peak_amplitude:.3f}, "
                         f"ZCR: {result.audio.zero_crossing_rate:.3f})", 
                         fontsize=11, fontweight="bold")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Amplitude")
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # Panel 2: FBank features
        ax2 = fig.add_subplot(gs[1, 0])
        fbank_outputs = captured.get("fbank_outputs", [])
        if fbank_outputs and fbank_outputs[0] is not None:
            fbank = fbank_outputs[0]
            im2 = ax2.imshow(fbank.T, aspect='auto', origin='lower', 
                           cmap='magma', interpolation='bilinear')
            ax2.set_title(f"Mel Filterbank Features\nShape: {fbank.shape}", 
                         fontsize=10, fontweight="bold")
            ax2.set_xlabel("Frames")
            ax2.set_ylabel("Mel Bins (80)")
            plt.colorbar(im2, ax=ax2, shrink=0.85, label='Energy (dB)')
        
        # Panel 3: LFR+CMVN features
        ax3 = fig.add_subplot(gs[1, 1])
        lfr_outputs = captured.get("lfr_outputs", [])
        if lfr_outputs and len(lfr_outputs[0]) > 0:
            lfr = lfr_outputs[0]
            im3 = ax3.imshow(lfr.T, aspect='auto', origin='lower', 
                           cmap='RdYlBu_r', interpolation='bilinear',
                           vmin=-3, vmax=3)
            ax3.set_title(f"LFR + CMVN Features\nShape: {lfr.shape} (7-frame stack)", 
                         fontsize=10, fontweight="bold")
            ax3.set_xlabel("Frames (subsampled)")
            ax3.set_ylabel("Feature Dimension (560)")
            plt.colorbar(im3, ax=ax3, shrink=0.85, label='Normalized Value')
        
        # Panel 4: Padded features overview
        ax4 = fig.add_subplot(gs[1, 2])
        padded_feats = captured.get("padded_feats")
        if padded_feats is not None:
            # Show first feature dimension as 1D projection
            mean_projection = np.mean(padded_feats[0], axis=-1)
            valid_len = result.features.num_valid_frames
            ax4.plot(mean_projection[:valid_len], color='#2E86AB', linewidth=1.5, 
                    label='Valid frames')
            if valid_len < len(mean_projection):
                ax4.plot(range(valid_len, len(mean_projection)), 
                        mean_projection[valid_len:], color='#D64045', 
                        linewidth=1.5, linestyle=':', label='Padding')
            ax4.axvline(x=valid_len, color='gray', linestyle='--', alpha=0.7)
            ax4.set_title(f"Feature Mean Projection\nValid: {valid_len}/{len(mean_projection)} frames", 
                         fontsize=10, fontweight="bold")
            ax4.set_xlabel("Frame Index")
            ax4.set_ylabel("Mean Feature Value")
            ax4.legend(fontsize=7)
            ax4.grid(True, alpha=0.3)
        
        # Panel 5: Frame-level confidence
        ax5 = fig.add_subplot(gs[2, 0])
        frame_conf = captured.get("frame_confidence", [])
        if frame_conf:
            frames = range(len(frame_conf))
            colors = ['#2E86AB' if c > 0.5 else '#D64045' for c in frame_conf]
            ax5.bar(frames, frame_conf, color=colors, alpha=0.8, width=1.0)
            ax5.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, 
                       label='50% confidence')
            ax5.axhline(y=result.decoding.top1_confidence_mean, color='green', 
                       linestyle='-', linewidth=2, 
                       label=f'Mean: {result.decoding.top1_confidence_mean:.3f}')
            ax5.set_title(f"Frame-Level Top-1 Confidence\n"
                         f"High conf frames: {sum(1 for c in frame_conf if c > 0.5)}/{len(frame_conf)}", 
                         fontsize=10, fontweight="bold")
            ax5.set_xlabel("Encoder Frame")
            ax5.set_ylabel("Confidence")
            ax5.set_ylim(0, 1.05)
            ax5.legend(fontsize=7)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Panel 6: Per-frame entropy
        ax6 = fig.add_subplot(gs[2, 1])
        frame_entropy = captured.get("frame_entropy", [])
        if frame_entropy:
            frames = range(len(frame_entropy))
            # Normalize entropy for color mapping
            max_entropy = np.log(result.decoding.vocab_size) if result.decoding.vocab_size > 0 else 10
            norm_entropy = [min(e / max_entropy, 1.0) for e in frame_entropy]
            colors = plt.cm.RdYlGn_r([1.0 - ne for ne in norm_entropy])
            ax6.bar(frames, frame_entropy, color=colors, alpha=0.8, width=1.0)
            ax6.axhline(y=result.decoding.entropy_mean, color='blue', 
                       linestyle='-', linewidth=2, 
                       label=f'Mean entropy: {result.decoding.entropy_mean:.3f}')
            ax6.set_title(f"Frame-Level Entropy\n"
                         f"(lower = more confident)", 
                         fontsize=10, fontweight="bold")
            ax6.set_xlabel("Encoder Frame")
            ax6.set_ylabel("Entropy (nats)")
            ax6.legend(fontsize=7)
            ax6.grid(True, alpha=0.3, axis='y')
        
        # Panel 7: Decoding statistics pie chart
        ax7 = fig.add_subplot(gs[2, 2])
        blank_count = result.decoding.num_blank_tokens
        unique_count = result.decoding.num_unique_tokens
        other_count = result.decoding.num_encoder_frames - blank_count - unique_count
        other_count = max(0, other_count)  # Avoid negative from dedup
        
        sizes = [blank_count, unique_count, other_count]
        labels = [f'Blank\n({blank_count})', 
                 f'Output Tokens\n({unique_count})',
                 f'Other\n({other_count})']
        colors_pie = ['#D64045', '#2E86AB', '#A8C256']
        explode = (0.02, 0.05, 0.02)
        
        wedges, texts, autotexts = ax7.pie(
            sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, shadow=False,
            textprops={'fontsize': 8}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax7.set_title(f"CTC Frame Distribution\nTotal: {result.decoding.num_encoder_frames} frames", 
                     fontsize=10, fontweight="bold")
        
        # Panel 8: Summary statistics table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Build summary table
        table_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Clean Text", result.clean_text[:60], "Language", result.language_detected.upper()],
            ["Emotion", result.emotion_detected, "Is Speech", str(result.is_speech)],
            ["Total Time", f"{result.total_time_ms:.1f} ms", "Inference Time", f"{result.inference.inference_time_ms:.1f} ms"],
            ["Audio Duration", f"{result.audio.duration_seconds:.3f}s", "RTF", f"{result.inference.inference_time_ms / (result.audio.duration_seconds * 1000):.4f}"],
            ["Encoder Frames", str(result.decoding.num_encoder_frames), "Vocab Size", str(result.decoding.vocab_size)],
            ["Top-1 Conf", f"{result.decoding.top1_confidence_mean:.3f}", "Top-5 Conf", f"{result.decoding.top5_confidence_mean:.3f}"],
            ["Blank Ratio", f"{result.decoding.blank_ratio:.3f}", "Dedup Ratio", f"{result.decoding.dedup_ratio:.3f}"],
            ["RMS Energy", f"{result.audio.rms_energy:.4f}", "Silence Ratio", f"{result.audio.silence_ratio:.3f}"],
            ["Execution Provider", result.inference.execution_provider, "Threads", str(result.inference.num_threads)],
        ]
        
        table = ax8.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            colWidths=[0.18, 0.32, 0.18, 0.32],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        
        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f0f4f8')
            cell.set_edgecolor('#d0d5db')
            cell.set_linewidth(0.5)
        
        ax8.set_title("Transcription Pipeline Summary", fontsize=12, fontweight="bold", pad=20)
        
        # Add footer
        fig.text(0.5, 0.01, 
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Model: SenseVoiceSmall ONNX | "
                f"Error: {result.error or 'None'}",
                ha='center', fontsize=8, color='gray', style='italic')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[Visualize] Saved analysis to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return save_path
    
    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all recorded results.
        
        Returns:
            Dictionary with summary statistics for monitoring/dashboards.
        """
        if not self.results_history:
            return {"error": "No results available", "total_requests": 0}
        
        successful = [r for r in self.results_history if r.error is None]
        failed = [r for r in self.results_history if r.error is not None]
        
        if not successful:
            return {
                "total_requests": len(self.results_history),
                "successful": 0,
                "failed": len(failed),
                "error": "No successful transcriptions",
            }
        
        # Timing statistics
        inference_times = [r.inference.inference_time_ms for r in successful]
        total_times = [r.total_time_ms for r in successful]
        audio_durations = [r.audio.duration_seconds for r in successful]
        rtfs = [t / (d * 1000) if d > 0 else float('inf') 
                for t, d in zip(inference_times, audio_durations)]
        
        # Confidence statistics
        top1_confs = [r.decoding.top1_confidence_mean for r in successful]
        entropies = [r.decoding.entropy_mean for r in successful]
        
        # Language distribution
        lang_counts = {}
        for r in successful:
            lang = r.language_detected
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Emotion distribution
        emotion_counts = {}
        for r in successful:
            emotion = r.emotion_detected
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_requests": len(self.results_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results_history) if self.results_history else 0,
            "timing": {
                "avg_inference_ms": float(np.mean(inference_times)),
                "p50_inference_ms": float(np.percentile(inference_times, 50)),
                "p95_inference_ms": float(np.percentile(inference_times, 95)),
                "p99_inference_ms": float(np.percentile(inference_times, 99)),
                "avg_total_ms": float(np.mean(total_times)),
                "avg_audio_duration_s": float(np.mean(audio_durations)),
                "avg_rtf": float(np.mean([r for r in rtfs if r != float('inf')])) if rtfs else 0,
                "min_rtf": float(np.min([r for r in rtfs if r != float('inf')])) if rtfs else 0,
                "max_rtf": float(np.max([r for r in rtfs if r != float('inf')])) if rtfs else 0,
            },
            "confidence": {
                "avg_top1": float(np.mean(top1_confs)),
                "min_top1": float(np.min(top1_confs)),
                "max_top1": float(np.max(top1_confs)),
                "avg_entropy": float(np.mean(entropies)),
                "low_confidence_ratio": float(np.mean([1 for c in top1_confs if c < 0.3]) / len(top1_confs)),
            },
            "languages": lang_counts,
            "emotions": emotion_counts,
            "speech_ratio": float(np.mean([1 for r in successful if r.is_speech])),
            "providers_used": list(set(r.inference.execution_provider for r in successful)),
        }
    
    def reset_history(self) -> None:
        """Clear all stored results history."""
        count = len(self.results_history)
        self.results_history.clear()
        print(f"[SenseVoiceTranscriber] Cleared {count} result(s) from history.")
    
    def export_results(
        self, 
        output_path: str, 
        format: Literal["jsonl", "json", "csv"] = "jsonl",
        include_raw_data: bool = False,
    ) -> str:
        """
        Export all stored results to a file.
        
        Args:
            output_path: Path for the output file.
            format: Export format ("jsonl", "json", "csv").
            include_raw_data: Include captured raw arrays (large files!).
            
        Returns:
            Path to the exported file.
        """
        if not self.results_history:
            print("[Export] No results to export.")
            return ""
        
        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in self.results_history:
                    f.write(result.to_json(include_raw_data=include_raw_data) + "\n")
        
        elif format == "json":
            data = [r.to_dict(include_raw_data=include_raw_data) for r in self.results_history]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            # Flatten the nested dict for CSV
            fieldnames = [
                "timestamp", "clean_text", "language_detected", "emotion_detected",
                "is_speech", "total_time_ms", "inference_time_ms", "audio_duration_s",
                "top1_confidence", "blank_ratio", "dedup_ratio", "entropy_mean",
                "num_encoder_frames", "num_unique_tokens", "error",
            ]
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results_history:
                    writer.writerow({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", 
                                                   time.localtime(r.timestamp)),
                        "clean_text": r.clean_text,
                        "language_detected": r.language_detected,
                        "emotion_detected": r.emotion_detected,
                        "is_speech": r.is_speech,
                        "total_time_ms": round(r.total_time_ms, 2),
                        "inference_time_ms": round(r.inference.inference_time_ms, 2),
                        "audio_duration_s": round(r.audio.duration_seconds, 3),
                        "top1_confidence": round(r.decoding.top1_confidence_mean, 4),
                        "blank_ratio": round(r.decoding.blank_ratio, 3),
                        "dedup_ratio": round(r.decoding.dedup_ratio, 3),
                        "entropy_mean": round(r.decoding.entropy_mean, 3),
                        "num_encoder_frames": r.decoding.num_encoder_frames,
                        "num_unique_tokens": r.decoding.num_unique_tokens,
                        "error": r.error or "",
                    })
        
        print(f"[Export] Exported {len(self.results_history)} results to: {output_path}")
        return output_path


# ============================================================================
# Global Singleton and Convenience Functions
# ============================================================================

_transcriber: Optional[SenseVoiceTranscriber] = None


def get_transcriber(
    model_dir: str = "iic/SenseVoiceSmall",
    device_id: Union[str, int] = "0",
    quantize: bool = True,
    enable_analysis: bool = False,
    results_log_path: Optional[str] = None,
) -> SenseVoiceTranscriber:
    """
    Get or create the global ONNX transcriber singleton.
    
    Args:
        model_dir: ModelScope model ID or local path.
        device_id: ONNX Runtime device ID (-1 for CPU).
        quantize: Use quantized model if available.
        enable_analysis: Enable full analysis pipeline.
        results_log_path: Path for results logging.
        
    Returns:
        The singleton SenseVoiceTranscriber instance.
    """
    global _transcriber
    if _transcriber is None:
        _transcriber = SenseVoiceTranscriber(
            model_dir=model_dir,
            device_id=device_id,
            quantize=quantize,
            enable_analysis=enable_analysis,
            results_log_path=results_log_path,
        )
    return _transcriber


def transcribe_audio(
    audio_bytes: bytes,
    language: SupportedLanguage = "auto",
    use_itn: bool = True,
    *,
    hotwords: Optional[Union[str, list[str]]] = None,
    context_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Transcribe raw audio bytes using ONNX SenseVoice.
    
    Simple interface designed for live server usage.
    
    Note: hotwords and context_prompt are accepted for API compatibility
    but are not supported by the ONNX SenseVoiceSmall model.
    
    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, FLAC, etc.).
        language: Language hint ("auto", "zh", "en", "ja", "ko", "yue").
        use_itn: Apply Inverse Text Normalization if True.
        hotwords: (Unused, kept for API compatibility).
        context_prompt: (Unused, kept for API compatibility).
        **kwargs: Additional arguments.
        
    Returns:
        Transcribed text string.
    """
    transcriber = get_transcriber()
    return transcriber.transcribe_bytes(
        audio_bytes,
        language=language,
        use_itn=use_itn,
    )


def transcribe_with_analysis(
    audio_bytes: bytes,
    language: SupportedLanguage = "auto",
    use_itn: bool = True,
) -> TranscriptionResult:
    """
    Transcribe audio with full analysis (convenience function).
    
    Args:
        audio_bytes: Raw audio file bytes.
        language: Language hint.
        use_itn: Apply Inverse Text Normalization.
        
    Returns:
        TranscriptionResult with full analysis.
    """
    transcriber = get_transcriber(enable_analysis=True)
    return transcriber.transcribe_with_analysis(
        audio_bytes,
        language=language,
        use_itn=use_itn,
    )


# ============================================================================
# Module Initialization (backward compatibility)
# ============================================================================

# For backward compatibility with code that accesses module-level attributes
transcriber = None
model = None


def _init_module():
    """Initialize module-level objects for backward compatibility."""
    global transcriber, model
    _t = get_transcriber()
    transcriber = _t
    model = _t.model


_init_module()


# ============================================================================
# Quick Test / Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("  SenseVoice ONNX Transcription - Result Analysis Demo")
    print("=" * 70)
    
    # Test files
    test_files = {
        "en": r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\example\en.mp3",
        "ja": r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FunAudioLLM_SenseVoice\example\ja.mp3",
    }
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    else:
        test_file = test_files.get("en", list(test_files.values())[0])
        test_lang = "auto"
    
    # Create transcriber with analysis enabled
    analyzer = SenseVoiceTranscriber(
        model_dir="iic/SenseVoiceSmall",
        device_id="0",
        quantize=True,
        enable_analysis=True,
        results_log_path="transcription_results.jsonl",
    )
    
    # Transcribe with analysis
    print(f"\n[Demo] Transcribing: {test_file}")
    print(f"[Demo] Language hint: {test_lang}")
    
    with open(test_file, "rb") as f:
        audio_bytes = f.read()
    
    result = analyzer.transcribe_with_analysis(
        audio_bytes, 
        language=test_lang,
        use_itn=True,
    )
    
    # Print summary
    result.print_summary()
    
    # Generate visualization
    try:
        viz_path = analyzer.visualize_result(
            result, 
            save_path="transcription_analysis.png",
            show_plot=False,
        )
        print(f"[Demo] Visualization saved to: {viz_path}")
    except ImportError:
        print("[Demo] matplotlib not available, skipping visualization.")
    
    # Print aggregate stats (just this one result)
    stats = analyzer.get_aggregate_statistics()
    print(f"\n[Aggregate Stats]")
    print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
    if 'timing' in stats:
        t = stats['timing']
        print(f"  Avg Inference: {t['avg_inference_ms']:.1f}ms")
        print(f"  Avg RTF: {t['avg_rtf']:.4f}")
    if 'confidence' in stats:
        c = stats['confidence']
        print(f"  Avg Top-1 Conf: {c['avg_top1']:.3f}")
        print(f"  Avg Entropy: {c['avg_entropy']:.3f}")
    
    # Export results
    analyzer.export_results("transcription_results.json", format="json")
    analyzer.export_results("transcription_results.csv", format="csv")
    
    print(f"\n[Demo] Complete!")
