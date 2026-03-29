# servers\live_subtitles\live_subtitles_server3.py
import asyncio
import json
import numpy as np
import sys
import websockets
from typing import Optional
from pathlib import Path

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
from reazonspeech.espnet.asr import load_model, transcribe, audio_from_numpy, TranscribeConfig


SAMPLE_RATE = 16000


class SubtitleServer:
    def __init__(self, vad_model_dir: str, server_config: dict):
        self.host = server_config["host"]
        self.port = server_config["port"]
        self.vad_model_dir = vad_model_dir

        self.vad = None
        self.asr_model = None
        self.tracker = None

    def _load_models(self):
        print("Loading FireRedVAD...")
        self.vad = load_vad(self.vad_model_dir)

        print("Loading ReazonSpeech ASR...")
        self.asr_model = load_model()  # auto-selects CUDA if available

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
                    asr_result = transcribe(self.asr_model, audio_data, cfg)

                    text = asr_result.text.strip()
                    if text:
                        segment_dict = {
                            "type": "segment",
                            "start": completed.start_seconds,
                            "end": completed.end_seconds,
                            "text": text,
                        }
                        await websocket.send(json.dumps(segment_dict))

                        completed.text = text
                        self.tracker.context_buffer.append(completed)

                        print(f"📤 Sent: [{segment_dict['start']:.2f} → {segment_dict['end']:.2f}] "
                                f"({len(text)} chars, ASR dur ~{len(completed.audio)/16000:.1f}s) "
                                f"{text[:110]}{'...' if len(text) > 110 else ''}")

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
