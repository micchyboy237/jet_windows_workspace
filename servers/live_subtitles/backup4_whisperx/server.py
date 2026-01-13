# server.py
# pip install fastapi uvicorn websockets numpy torch torchaudio speechbrain faster-whisper whisperx silero-vad

import asyncio
import base64
import json
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from rich.console import Console
from silero_vad import load_silero_vad, VADIterator

console = Console()

app = FastAPI(title="Live JA→EN ASR + Diarization (best effort)")

# ────────────────────────────────────────────────
#  Models (load once)
# ────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
console.print(f"[bold green]Using device: {DEVICE}[/bold green]")

# Fast Japanese ASR
asr_model = whisperx.load_model(
    "large-v3", device=DEVICE, compute_type="float16", language="ja"
)

# Translation model (fast & good JA→EN) - example with seamless-m4t or NLLB-200-distilled-600M
# Here we simulate with dummy function – replace with real one
def translate_ja_to_en(text: str) -> str:
    # Placeholder – use seamless_communication, argos_translate, transformers pipeline(NLLB), etc.
    return f"[EN] {text} (translated)"

# Very lightweight speaker embedding model (speechbrain is fast)
from speechbrain.pretrained import SpeakerRecognition
spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="./pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# Silero VAD
vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=16000, threshold=0.5)

# ────────────────────────────────────────────────
# State per connection
# ────────────────────────────────────────────────

class ConnectionState:
    def __init__(self):
        self.audio_buffer: Deque[np.ndarray] = deque(maxlen=480000)  # ~30s @16kHz
        self.segments: List[Dict] = []
        self.srt_lines: List[str] = []
        self.speaker_history: Dict[int, np.ndarray] = {}  # cluster_id → embedding
        self.next_speaker_id = 0
        self.sample_rate = 16000

    def add_audio_chunk(self, pcm_16bit: np.ndarray):
        self.audio_buffer.extend(pcm_16bit)

    async def process_speech(self, websocket: WebSocket):
        # Called when VAD detects end of speech
        if len(self.audio_buffer) < 16000 * 3:  # min 3 seconds
            return

        audio_np = np.concatenate(list(self.audio_buffer))
        self.audio_buffer.clear()  # reset for next utterance

        start_time = time.time()

        # 1. ASR (Japanese)
        result = asr_model.transcribe(
            audio_np,
            batch_size=8,
            language="ja",
            chunk_size=15
        )

        # 2. Translate segment texts
        for seg in result["segments"]:
            seg["text_en"] = translate_ja_to_en(seg["text"])

        # 3. Lightweight speaker assignment
        # Extract embedding from whole utterance (fast approximation)
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
        embedding = spk_model.encode_batch(waveform).squeeze().cpu().numpy()

        # Find closest speaker or create new
        best_spk = None
        best_score = -1
        for spk_id, emb in self.speaker_history.items():
            score = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
            if score > 0.75 and score > best_score:  # threshold tunable
                best_score = score
                best_spk = spk_id

        if best_spk is None:
            best_spk = f"SPEAKER_{self.next_speaker_id:02d}"
            self.speaker_history[self.next_speaker_id] = embedding
            self.next_speaker_id += 1

        # Assign to all segments in this utterance
        for seg in result["segments"]:
            seg["speaker"] = best_spk

        # 4. Format as .srt block
        block_start_idx = len(self.srt_lines) + 1
        for i, seg in enumerate(result["segments"], block_start_idx):
            start = seg["start"]
            end = seg["end"]
            text = f"{seg['speaker']} | {seg['text']} → {seg['text_en']}"
            self.srt_lines.extend([
                str(i),
                f"{self.format_timestamp(start)} --> {self.format_timestamp(end)}",
                text,
                ""
            ])

        # 5. Send to client
        await websocket.send_json({
            "type": "subtitle_update",
            "segments": result["segments"],
            "srt_block": "\n".join(self.srt_lines[-8:])  # last few for display
        })

        console.print(f"[cyan]Processed utterance in {time.time()-start_time:.2f}s[/cyan]")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ────────────────────────────────────────────────
# WebSocket endpoint
# ────────────────────────────────────────────────

@app.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = ConnectionState()
    console.print("[bold green]New client connected[/bold green]")

    try:
        while True:
            message = await websocket.receive_bytes()

            # Assume client sends raw 16-bit PCM little-endian
            pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

            state.add_audio_chunk(pcm)

            # VAD check
            for sample in pcm:
                speech_dict = vad_iterator(sample, return_seconds=True)
                if speech_dict:
                    if "start" in speech_dict:
                        console.print("[yellow]Speech start detected[/yellow]")
                    if "end" in speech_dict:
                        console.print("[yellow]Speech end detected → transcribing...[/yellow]")
                        await state.process_speech(websocket)

            # Optional: send partial / heartbeat
            # await websocket.send_json({"type": "alive"})

    except WebSocketDisconnect:
        console.print("[bold red]Client disconnected[/bold red]")
        # Save final .srt if needed
        with open("live_output.srt", "w", encoding="utf-8") as f:
            f.write("\n".join(state.srt_lines))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")