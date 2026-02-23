"""Live Japanese to English subtitles client. Captures system audio and displays subtitles."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Literal, TypedDict

import numpy as np
import sounddevice as sd
import websockets
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("live-sub-client")


class SubtitleMessage(TypedDict):
    type: Literal["subtitle"]
    text: str
    jp_text: str


class LiveJPSubtitlesClient:
    """Client for capturing audio and displaying live subtitles."""

    def __init__(
        self,
        ws_url: str = os.getenv("LOCAL_WS_LIVE_SUBTITLES_URL", "ws://localhost:8765"),
        sample_rate: int = 16000,
        chunk_duration: float = 0.25,
        show_audio_activity: bool = True,
    ):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.show_audio_activity = show_audio_activity

        self.current_subtitle: str = ""
        self.previous_subtitles: list[str] = []
        self.console = Console()

        # Status tracking
        self.status_text = "Initializing..."
        self.last_audio_time = time.time()
        self.audio_active = False
        self.connected = False

    def list_audio_devices(self) -> None:
        """List available input devices for user to select system audio loopback."""
        devices = sd.query_devices()
        table = Table(title="Available Audio Input Devices", show_header=True)
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Name", style="green")
        table.add_column("Channels", justify="right")
        table.add_column("Default Sample Rate", justify="right")
        table.add_column("Default", style="yellow")

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] == 0:
                continue
            default = "★" if dev.get("is_default_input", False) else ""
            table.add_row(
                str(i),
                dev["name"],
                str(dev["max_input_channels"]),
                f"{int(dev.get('default_samplerate', 0))} Hz",
                default,
            )

        self.console.print(table)
        self.console.print(
            "\n[bold yellow]Tip:[/] Use BlackHole (Mac) or VB-Audio / VoiceMeeter (Windows) for system audio capture.",
            style="dim",
        )

    def update_status(self, msg: str, live: Live | None = None) -> None:
        self.status_text = msg
        if live:
            live.refresh()

    async def display_loop(self, websocket: websockets.WebSocketClientProtocol) -> None:
        """Display updating subtitles + status using rich Live."""
        last_update = time.time()

        def make_content():
            nonlocal last_update
            now = time.time()

            subtitle_lines = Text()
            for prev in self.previous_subtitles[-3:]:
                if prev:
                    subtitle_lines.append(prev + "\n", style="dim italic")

            if self.current_subtitle:
                subtitle_lines.append(self.current_subtitle, style="bold green")
            else:
                subtitle_lines.append("Waiting for speech...", style="grey37 italic")

            jp_line = ""
            if hasattr(self, "last_jp_text") and self.last_jp_text:
                jp_line = f"\n[JP] {self.last_jp_text}"

            main_panel = Panel(
                subtitle_lines.append(jp_line, style="blue"),
                title="Live JP → EN Subtitles",
                border_style="bright_blue",
                padding=(1, 2),
            )

            # Connection & audio status
            conn_status = (
                "[green]Connected[/]" if self.connected else "[red]Disconnected[/]"
            )
            audio_age = now - self.last_audio_time
            audio_status = (
                "[green]Audio active[/]"
                if audio_age < 1.5
                else "[yellow]No audio recently[/]"
                if audio_age < 5
                else "[red]No audio detected[/]"
            )

            status_bar = Text.assemble(
                ("Status: ", "bold"),
                (f"{self.status_text}  •  ", ""),
                (conn_status, "bold"),
                "  •  ",
                (audio_status, "bold"),
                f"  •  Last chunk {audio_age:.1f}s ago",
            )

            return Group(status_bar, main_panel)

        with Live(
            make_content(),
            refresh_per_second=8,
            console=self.console,
            vertical_overflow="visible",
        ) as live:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), 0.08)
                    if isinstance(message, str):
                        data: dict[str, Any] = json.loads(message)
                        if data.get("type") == "subtitle":
                            en_text = (data.get("text") or "").strip()
                            jp_text = (data.get("jp_text") or "").strip()
                            is_partial = data.get("partial", False)
                            if en_text or jp_text:
                                self.previous_subtitles.append(self.current_subtitle)
                                self.current_subtitle = en_text
                                self.last_jp_text = jp_text
                                live.update(make_content())
                                style = "italic grey" if is_partial else "bold green"
                                logger.info(
                                    f"[{style}][EN] {en_text} [dim](JP: {jp_text})[/]"
                                )

                except asyncio.TimeoutError:
                    live.update(make_content())  # refresh status
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.connected = False
                    self.update_status("Connection lost — reconnecting...", live)
                    break
                except Exception as e:
                    logger.exception("Display loop error")
                    self.update_status(f"Error: {str(e)}", live)
                    await asyncio.sleep(2)

    async def audio_sender(
        self,
        websocket: websockets.WebSocketClientProtocol,
        queue: asyncio.Queue[np.ndarray],
    ) -> None:
        """Send audio chunks from queue to server."""
        sent_chunks = 0
        while True:
            try:
                chunk = await queue.get()
                await websocket.send(chunk.tobytes())
                sent_chunks += 1
                self.last_audio_time = time.time()
                self.audio_active = True
                queue.task_done()

                if sent_chunks % 40 == 0:  # ~10 seconds
                    logger.debug(f"Sent {sent_chunks} audio chunks")
            except Exception as e:
                logger.error(f"Audio sender failed: {e}")
                break

    async def run(self) -> None:
        """Main entry point."""
        logger.info("[bold cyan]Live Japanese → English Subtitles Client[/]")
        self.list_audio_devices()

        self.update_status("Starting audio capture...")
        logger.info("Starting audio capture (use system loopback device)")

        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=60)

        def callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                logger.warning(f"Audio callback warning: {status}")
            try:
                audio_queue.put_nowait(indata.copy().flatten())
            except asyncio.QueueFull:
                logger.warning(
                    "Audio queue full — dropping chunk", extra={"markup": True}
                )

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=callback,
            ):
                self.update_status("Audio stream opened")
                logger.info(
                    f"[green]Audio input stream opened[/] ({self.chunk_duration}s chunks)"
                )

                async with websockets.connect(
                    self.ws_url,
                    ping_interval=15,
                    ping_timeout=35,
                    max_size=2**20,
                ) as ws:
                    self.connected = True
                    self.update_status("Connected to subtitle server")
                    logger.info(f"[green]Connected[/] to {self.ws_url}")

                    sender_task = asyncio.create_task(
                        self.audio_sender(ws, audio_queue)
                    )
                    display_task = asyncio.create_task(self.display_loop(ws))

                    await asyncio.gather(
                        sender_task, display_task, return_exceptions=True
                    )

        except KeyboardInterrupt:
            logger.info("[yellow]Shutdown requested by user[/]")
        except Exception as e:
            logger.exception("Fatal error in main loop")
            console.print(f"\n[red bold]Error:[/] {e}")
        finally:
            self.connected = False
            logger.info("[dim]Client shutdown[/]")


if __name__ == "__main__":
    client = LiveJPSubtitlesClient()
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye.[/]")
