import os
import asyncio
import threading
from typing import Optional

import websockets
from pyaudiowpatch as pyaudio
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from tqdm import tqdm

console = Console()
SERVER_URL = "ws://localhost:8000/ws/subtitles"  # Change to your server IP if remote

class AudioCapturer:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

    def get_loopback_device(self):
        """Find WASAPI loopback on Windows or default on Mac (assumes BlackHole selected)."""
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if (dev["maxOutputChannels"] > 0 or "loopback" in dev["name"].lower()) and dev["hostApi"] == 0:  # WASAPI
                return i
        raise RuntimeError("No loopback device found. Install BlackHole on Mac or enable Stereo Mix on Windows.")

    def start(self):
        device_index = self.get_loopback_device()
        console.print(f"[blue]Using loopback device: {self.p.get_device_info_by_index(device_index)['name']}[/blue]")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
        )
        self.stream.start_stream()

    def read_chunk(self) -> bytes:
        if self.stream is None:
            raise RuntimeError("Stream not started")
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

async def stream_audio():
    capturer = AudioCapturer()
    capturer.start()
    console.print("[green]Capturing system audio...[/green]")

    current_subtitle = ""

    with Live(Panel(current_subtitle, title="Live English Subtitles", border_style="cyan"), refresh_per_second=10) as live:
        async with websockets.connect(SERVER_URL) as ws:
            console.print("[green]Connected to server[/green]")

            async def send_audio():
                pbar = tqdm(total=0, unit="chunk", desc="Streaming audio")
                while True:
                    chunk = capturer.read_chunk()
                    await ws.send(chunk)
                    pbar.update(1)

            async def receive_subtitles():
                nonlocal current_subtitle
                while True:
                    message = await ws.recv()
                    if message.startswith("[FINAL]"):
                        current_subtitle = message[7:]
                    elif message.startswith("[PARTIAL]"):
                        current_subtitle = message[9:] + " ..."
                    live.update(Panel(current_subtitle, title="Live English Subtitles", border_style="cyan"))

            try:
                await asyncio.gather(send_audio(), receive_subtitles())
            except websockets.ConnectionClosed:
                console.print("[red]Connection closed[/red]")

    capturer.stop()

if __name__ == "__main__":
    asyncio.run(stream_audio())