# servers\live_subtitles\live_subtitles_server3.py
import asyncio
import json
import numpy as np
import shutil
import sys
import wave
import websockets
from typing import Optional, Deque
from pathlib import Path
from collections import deque
from energy import has_sound

# ====================== CLONED-REPO PATH SETUP (must be first) ======================
PROJECT_ROOT = Path(__file__).parent.parent.parent
FIRE_RED_VAD_PATH = str((PROJECT_ROOT / "Cloned_Repos" / "FireRedVAD").resolve())
REAZON_ASR_PATH = str((PROJECT_ROOT / "Cloned_Repos" / "ReazonSpeech" / "pkg" / "espnet-asr" / "src").resolve())

sys.path.insert(0, FIRE_RED_VAD_PATH)
sys.path.insert(0, REAZON_ASR_PATH)

# ====================== LOCAL MODULES ======================
from fireredvad_utils import load_vad
from speech_segment_tracker import SpeechSegmentTracker, AccumulatedSpeechSegment

# ====================== REAZONSPEECH IMPORTS (now resolvable) ======================
from reazonspeech.k2.asr.interface import TranscribeConfig
from reazonspeech.k2.asr.transcribe import transcribe
from reazonspeech.k2.asr.audio import audio_from_numpy
from reazonspeech.k2.asr.huggingface import load_model

SAMPLE_RATE = 16000
BUFFER_DURATION_SECONDS = 5 * 60  # 5 minutes
MAX_BUFFER_SAMPLES = BUFFER_DURATION_SECONDS * SAMPLE_RATE  # 480,000 samples @ 16kHz

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOUND_5_MINS_FILE = OUTPUT_DIR / "sound_5_mins.wav"


class CircularAudioBuffer:
    """
    Reusable circular buffer for accumulating audio data up to a fixed duration.
    Maintains the last N seconds of audio at a given sample rate.
    
    Thread-safe for single-threaded async use. For multi-threaded scenarios,
    add appropriate locks.
    """
    
    def __init__(self, max_samples: int, sample_rate: int):
        """
        Initialize the circular audio buffer.
        
        Args:
            max_samples: Maximum number of samples to retain (e.g., 5 min * 16000 Hz = 480000)
            sample_rate: Audio sample rate in Hz
        """
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self._buffer: Deque[np.float32] = deque(maxlen=max_samples)
    
    def append(self, audio_chunk: np.ndarray):
        """
        Append audio samples to the buffer. Oldest samples are automatically
        dropped when the buffer reaches capacity.
        
        Args:
            audio_chunk: numpy array of float32 audio samples
        """
        if len(audio_chunk) == 0:
            return
        # Extend deque with new samples; oldest samples automatically dropped
        self._buffer.extend(audio_chunk.astype(np.float32))
    
    def get_audio(self) -> np.ndarray:
        """
        Get the current audio content as a numpy array.
        
        Returns:
            numpy array of float32 samples (up to max_samples)
        """
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.array(self._buffer, dtype=np.float32)
    
    def save_to_wav(self, filepath: Path):
        """
        Save the current buffer contents to a WAV file.
        
        Args:
            filepath: Path where the WAV file should be saved
        """
        audio_data = self.get_audio()
        if len(audio_data) == 0:
            print(f"⚠️  No audio data to save to {filepath}")
            return
        
        # Convert float32 [-1, 1] to int16 for WAV format
        audio_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 2 bytes = int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        print(f"💾 Saved {len(audio_data)/self.sample_rate:.1f}s audio to {filepath}")
    
    def clear(self):
        """Clear all audio data from the buffer."""
        self._buffer.clear()
    
    def __len__(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self._buffer)
    
    @property
    def duration_seconds(self) -> float:
        """Return the current duration of audio in the buffer in seconds."""
        return len(self._buffer) / self.sample_rate
    
    def __repr__(self) -> str:
        return f"CircularAudioBuffer({self.duration_seconds:.1f}s / {self.max_samples/self.sample_rate:.1f}s max)"


class SubtitleServer:
    def __init__(self, vad_model_dir: str, server_config: dict):
        self.host = server_config["host"]
        self.port = server_config["port"]
        self.vad_model_dir = vad_model_dir

        self.vad = None
        self.asr_model = None
        self.tracker = None
        self.audio_buffer: Optional[CircularAudioBuffer] = None

    def _load_models(self):
        print("Loading FireRedVAD...")
        self.vad = load_vad(self.vad_model_dir)

        print("Loading ReazonSpeech ASR...")
        self.asr_model = load_model()  # auto-selects CUDA if available

        self.tracker = SpeechSegmentTracker(sample_rate=SAMPLE_RATE)
        
        # Initialize the 5-minute circular audio buffer
        self.audio_buffer = CircularAudioBuffer(
            max_samples=MAX_BUFFER_SAMPLES,
            sample_rate=SAMPLE_RATE
        )
        print(f"✅ All models ready – audio buffer: {self.audio_buffer} – waiting for clients")

    async def handler(self, websocket):
        print("🎤 Client connected – starting live subtitle stream")
        try:
            async for message in websocket:
                # raw bytes → float32 chunk
                audio_chunk = np.frombuffer(message, dtype=np.float32)

                if len(audio_chunk) == 0:
                    continue

                if not has_sound(audio_chunk):
                    continue

                # 🔁 Always accumulate audio in circular buffer (regardless of VAD)
                self.audio_buffer.append(audio_chunk)

                # VAD (stateful)
                vad_results = self.vad.detect_chunk(audio_chunk)

                # Tracker decides if a speech segment just finished
                completed: Optional[AccumulatedSpeechSegment] = self.tracker.process_vad_results(
                    audio_chunk, vad_results
                )

                if completed is not None:
                    # 💾 Save the last 5 minutes of audio on each detected speech event
                    self.audio_buffer.save_to_wav(SOUND_5_MINS_FILE)
                    
                    # Transcribe the exact speech waveform
                    audio_data = audio_from_numpy(completed.audio, SAMPLE_RATE)
                    cfg = TranscribeConfig(verbose=False)
                    asr_result = transcribe(self.asr_model, audio_data, cfg)

                    jp_text = asr_result.text.strip()
                    if jp_text:
                        segment_dict = {
                            "type": "segment",
                            "start": completed.start_seconds,
                            "end": completed.end_seconds,
                            "jp": jp_text,
                            "en": "",
                        }
                        await websocket.send(json.dumps(segment_dict))

                        completed.jp = jp_text
                        self.tracker.context_buffer.append(completed)

                        print(f"📤 Sent: [{segment_dict['start']:.2f} → {segment_dict['end']:.2f}] "
                                f"({len(jp_text)} chars, ASR dur ~{len(completed.audio)/16000:.1f}s) "
                                f"{jp_text[:110]}{'...' if len(jp_text) > 110 else ''}")

        except Exception as e:
            print(f"Handler error: {e}")
        finally:
            print("Client disconnected")

    async def start(self):
        self._load_models()
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"🚀 SubtitleServer listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8765
    vad_model_dir = str(
        Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD")
        .expanduser()
        .resolve()
    )
    server = SubtitleServer(vad_model_dir, {"host": host, "port": port})
    asyncio.run(server.start())
