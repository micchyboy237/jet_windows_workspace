import asyncio
import json
import sys
import numpy as np
from pathlib import Path
from typing import Optional
import websockets

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIRE_RED_VAD_PATH = str((PROJECT_ROOT / "Cloned_Repos" / "FireRedVAD").resolve())
REAZON_ASR_PATH = str(
    (
        PROJECT_ROOT / "Cloned_Repos" / "ReazonSpeech" / "pkg" / "espnet-asr" / "src"
    ).resolve()
)
sys.path.insert(0, FIRE_RED_VAD_PATH)
sys.path.insert(0, REAZON_ASR_PATH)

from fireredvad_utils import load_vad
from reazonspeech.k2.asr.audio import audio_from_numpy
from reazonspeech.k2.asr.huggingface import load_model
from reazonspeech.k2.asr.interface import TranscribeConfig
from reazonspeech.k2.asr.transcribe import transcribe
from speech_segment_tracker import AccumulatedSpeechSegment, SpeechSegmentTracker
from translate_jp_en_llm import translate_japanese_to_english

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
        self.asr_model = load_model()
        self.tracker = SpeechSegmentTracker(sample_rate=SAMPLE_RATE)
        print("✅ All models ready – waiting for clients")

    async def handler(self, websocket):
        # Enforce the exact WebSocket path
        path = websocket.request.path
        if path != "/ws/live-subtitles":
            print(f"❌ Rejected connection on invalid path: {path}")
            await websocket.close()
            return

        print("🎤 Client connected – starting live subtitle stream")
        try:
            async for message in websocket:
                audio_chunk = np.frombuffer(message, dtype=np.float32)
                if len(audio_chunk) == 0:
                    continue

                vad_results = self.vad.detect_chunk(audio_chunk)
                completed: Optional[AccumulatedSpeechSegment] = (
                    self.tracker.process_vad_results(audio_chunk, vad_results)
                )
                if completed is not None:
                    audio_data = audio_from_numpy(completed.audio, SAMPLE_RATE)
                    cfg = TranscribeConfig(verbose=False)
                    asr_result = transcribe(self.asr_model, audio_data, cfg)
                    jp_text = asr_result.text.strip()
                    en_text = ""
                    if jp_text:
                        translated_result = translate_japanese_to_english(jp_text)
                        en_text = translated_result["text"].strip()

                    if jp_text:
                        segment_dict = {
                            "type": "segment",
                            "start": completed.start_seconds,
                            "end": completed.end_seconds,
                            "jp": jp_text,
                            "en": en_text,
                        }
                        await websocket.send(json.dumps(segment_dict))
                        completed.jp = jp_text
                        completed.en = en_text
                        self.tracker.context_buffer.append(completed)
                        print(
                            f"📤 Sent: [{segment_dict['start']:.2f} → {segment_dict['end']:.2f}] "
                            f"JP:{jp_text}\n"
                            f"EN:{en_text}"
                        )
        except Exception as e:
            print(f"Handler error: {e}")
        finally:
            print("Client disconnected")

    async def start(self):
        self._load_models()
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"🚀 SubtitleServer listening on ws://{self.host}:{self.port}/ws/live-subtitles")
            await asyncio.Future()

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    vad_model_dir = str(
        Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
    )
    server = SubtitleServer(vad_model_dir, {"host": host, "port": port})
    asyncio.run(server.start())