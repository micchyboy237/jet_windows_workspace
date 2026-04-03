# servers\live_subtitles\live_subtitles_server3.py
import asyncio
import json
import numpy as np
import sys
import websockets
from typing import Optional
from pathlib import Path
import concurrent.futures

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

# ====================== SPEAKER EMBEDDING IMPORTS ======================
from speaker_manager import PyannoteEmbeddingModel, SpeakerManager

SAMPLE_RATE = 16000
MIN_RMS_THRESHOLD = 0.005  # Tune between ~0.003–0.02 depending on environment

class SubtitleServer:
    def __init__(self, vad_model_dir: str, server_config: dict):
        self.host = server_config["host"]
        self.port = server_config["port"]
        self.vad_model_dir = vad_model_dir

        self.vad = None
        self.asr_model = None
        self.tracker = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.speaker_model = None
        self.speaker_manager = None

    def _load_models(self):
        print("Loading FireRedVAD...")
        self.vad = load_vad(self.vad_model_dir)

        print("Loading ReazonSpeech ASR...")
        self.asr_model = load_model()  # auto-selects CUDA if available

        print("Loading speaker embedding model (pyannote)...")
        self.speaker_model = PyannoteEmbeddingModel(device="cpu")
        self.speaker_manager = SpeakerManager()

        self.tracker = SpeechSegmentTracker(sample_rate=SAMPLE_RATE)
        print("✅ All models ready – waiting for clients")

    async def handler(self, websocket):
        print("🎤 Client connected – starting live subtitle stream")
        try:
            async for message in websocket:
                # raw bytes → float32 chunk
                audio_chunk = np.frombuffer(message, dtype=np.float32)

                if len(audio_chunk) == 0:
                    continue

                # -------------------- Noise Gate (RMS-based) --------------------
                # Skip very low-energy chunks (likely background noise)
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                if rms < MIN_RMS_THRESHOLD:
                    continue

                # VAD (stateful)
                vad_results = self.vad.detect_chunk(audio_chunk)

                # Tracker decides if a speech segment just finished
                completed: Optional[AccumulatedSpeechSegment] = self.tracker.process_vad_results(
                    audio_chunk, vad_results
                )

                if completed is not None:
                    # Transcribe the exact speech waveform
                    audio_data = audio_from_numpy(completed.audio, SAMPLE_RATE)
                    cfg = TranscribeConfig(verbose=False)

                    # -------------------- Non-blocking ASR --------------------
                    loop = asyncio.get_running_loop()
                    asr_result = await loop.run_in_executor(
                        self.executor,
                        transcribe,
                        self.asr_model,
                        audio_data,
                        cfg,
                    )

                    jp_text = asr_result.text.strip()

                    if jp_text:
                        # -------------------- Speaker Embedding --------------------
                        embedding = self.speaker_model.embed(
                            completed.audio, SAMPLE_RATE
                        )
                        speaker_id = self.speaker_manager.assign_speaker(embedding)

                        segment_dict = {
                            "type": "segment",
                            "speaker": speaker_id,
                            "start": completed.start_seconds,
                            "end": completed.end_seconds,
                            "jp": jp_text,
                            "en": "",
                        }
                        await websocket.send(json.dumps(segment_dict))

                        completed.jp = jp_text
                        self.tracker.context_buffer.append(completed)

                        print(f"📤 Sent: [{segment_dict['start']:.2f} → {segment_dict['end']:.2f}] "
                              f"[S{speaker_id}] "
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
