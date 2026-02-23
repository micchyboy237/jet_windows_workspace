"""Live Japanese to English subtitles client. Captures system audio and displays subtitles."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal, TypedDict

import numpy as np
import sounddevice as sd
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class SubtitleMessage(TypedDict):
    type: Literal["subtitle"]
    text: str
    jp_text: str


class LiveJPSubtitlesClient:
    """Client for capturing audio and displaying live subtitles."""

    def __init__(
        self,
        ws_url: str = "ws://localhost:8765",
        sample_rate: int = 16000,
        chunk_duration: float = 0.25,
    ):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.current_subtitle: str = ""
        self.previous_subtitles: list[str] = []
        self.console = Console()

    def list_audio_devices(self) -> None:
        """List available input devices for user to select system audio loopback."""
        devices = sd.query_devices()
        table = Table(
            title="Available Audio Input Devices (Select loopback for system audio)"
        )
        table.add_column("Index", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Default", style="yellow")
        for i, dev in enumerate(devices):
            default = (
                " * "
                if dev.get("default_samplerate", 0) > 0
                and dev.get("max_input_channels", 0) > 0
                else ""
            )
            table.add_row(str(i), dev["name"], default)
        self.console.print(table)
        self.console.print(
            "Tip: Use BlackHole (Mac M1) or VB-Audio (Windows) for system audio capture."
        )

    async def display_loop(self, websocket: websockets.WebSocketClientProtocol) -> None:
        """Display updating subtitles using rich Live."""
        with Live(refresh_per_second=10, console=self.console) as live:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), 0.05)
                    if isinstance(message, str):
                        data: dict[str, Any] = json.loads(message)
                        if data.get("type") == "subtitle":
                            en_text = data.get("text", "")
                            jp_text = data.get("jp_text", "")
                            self.previous_subtitles.append(self.current_subtitle)
                            self.current_subtitle = en_text
                            # Build display
                            display = Text()
                            for prev in self.previous_subtitles[-3:]:
                                if prev:
                                    display.append(prev + "\n", style="dim")
                            display.append(self.current_subtitle, style="bold green")
                            if jp_text:
                                display.append(f"\n[JP] {jp_text}", style="blue")
                            live.update(
                                Panel(
                                    display,
                                    title="Live JP→EN Subtitles",
                                    border_style="blue",
                                )
                            )
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.console.log(f"Display error: {e}")

    async def audio_sender(
        self,
        websocket: websockets.WebSocketClientProtocol,
        queue: asyncio.Queue[np.ndarray],
    ) -> None:
        """Send audio chunks from queue to server."""
        while True:
            chunk = await queue.get()
            await websocket.send(chunk.tobytes())
            queue.task_done()

    async def run(self) -> None:
        """Main entry point."""
        self.list_audio_devices()
        self.console.print(
            "Starting client... (set default input device to your system audio loopback)"
        )
        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)

        def callback(indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
            if status:
                self.console.log(f"Audio status: {status}")
            try:
                audio_queue.put_nowait(indata.copy().flatten())
            except asyncio.QueueFull:
                pass

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=callback,
        ):
            async with websockets.connect(self.ws_url) as ws:
                sender_task = asyncio.create_task(self.audio_sender(ws, audio_queue))
                display_task = asyncio.create_task(self.display_loop(ws))
                await asyncio.gather(sender_task, display_task, return_exceptions=True)


if __name__ == "__main__":
    client = LiveJPSubtitlesClient()
    asyncio.run(client.run())
