"""Live Japanese to English subtitles server. Handles ASR and translation via WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Deque, Dict

import numpy as np
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from utils.asr import ASRTranscriber
from utils.audio_utils import AudioStreamProcessor
from utils.translation import JapaneseToEnglishTranslator
from websockets.asyncio.server import serve, ServerConnection

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_time=True)]
)
logger = logging.getLogger("jp-sub-server")


class ServerState:
    """Shared state for Live display"""
    def __init__(self):
        self.clients: Dict[int, dict] = {}           # client_id → status dict
        self.recent_subtitles: Deque[dict] = deque(maxlen=5)
        self.total_utterances = 0
        self.total_chunks = 0
        self.start_time = time.time()


class JPSubtitlesServer:
    """Server for processing live audio to subtitles."""

    def __init__(
        self,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=None,
    ):
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_size = max_size
        self.asr = ASRTranscriber()
        self.translator = JapaneseToEnglishTranslator()
        self.processors: dict[int, AudioStreamProcessor] = {}
        self.state = ServerState()
        self.console = console

    async def handler(self, websocket: ServerConnection):
        """Handle one client connection."""
        client_id = id(websocket)
        peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected [dim]({peer})[/] id={client_id}")

        self.state.clients[client_id] = {
            "peer": peer,
            "status": "connected",
            "last_active": time.time(),
            "chunks": 0,
            "utterances": 0,
        }

        processor = AudioStreamProcessor(self.asr, self.translator)
        self.processors[client_id] = processor

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self.state.clients[client_id]["chunks"] += 1
                    self.state.clients[client_id]["last_active"] = time.time()
                    chunk = np.frombuffer(message, dtype=np.float32)
                    result = processor.process_chunk(chunk)
                    if result:
                        en_text, jp_text, is_partial = result
                        if en_text.strip():
                            self.state.total_utterances += 1
                            self.state.clients[client_id]["utterances"] += 1
                            self.state.recent_subtitles.append({
                                "time": time.strftime("%H:%M:%S"),
                                "en": en_text[:80] + ("…" if len(en_text) > 80 else ""),
                                "jp": jp_text[:60] + ("…" if len(jp_text) > 60 else ""),
                                "client": client_id,
                            })

                            response = {
                                "type": "subtitle",
                                "is_partial": is_partial,
                                "text": en_text,
                                "jp_text": jp_text,
                            }
                            await websocket.send(json.dumps(response))
                            logger.info(f"[green]Sent[/] [dim]#{client_id}[/] {en_text[:60]}{'…' if len(en_text)>60 else ''}")

        except Exception as e:
            logger.exception(f"Client error [dim]#{client_id}[/]")
            self.state.clients[client_id]["status"] = f"error: {str(e)[:40]}"
        finally:
            self.processors.pop(client_id, None)
            self.state.clients.pop(client_id, None)
            logger.info(f"Client disconnected [dim]#{client_id} ({peer})[/]")

    def make_status_renderable(self):
        uptime = time.time() - self.state.start_time
        uptime_str = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"

        table = Table(expand=True, show_header=False, border_style="dim")
        table.add_column("Client", style="cyan", width=12)
        table.add_column("Peer", style="green")
        table.add_column("Status", width=20)
        table.add_column("Chunks / Utter.", justify="right")

        for cid, info in sorted(self.state.clients.items()):
            status_color = {
                "connected": "green",
                "processing": "yellow",
                "error": "red",
            }.get(info["status"], "white")

            table.add_row(
                f"#{cid}",
                info["peer"],
                f"[{status_color}]{info['status']}[/]",
                f"{info['chunks']:,} / {info['utterances']}",
            )

        recent = Text()
        for item in self.state.recent_subtitles:
            recent.append(f"[{item['time']}] ", style="dim")
            recent.append(f"#{item['client']} ", style="cyan")
            recent.append(f"{item['en']}", style="green")
            if item['jp']:
                recent.append(f"  [dim]({item['jp']})[/]", style="blue")
            recent.append("\n")

        stats = Text.assemble(
            ("Uptime: ", "bold"), (uptime_str, "green"), "  •  ",
            ("Clients: ", "bold"), (f"{len(self.state.clients)}", "magenta"), "  •  ",
            ("Utterances: ", "bold"), (f"{self.state.total_utterances:,}", "yellow")
        )

        return Group(
            Panel(stats, title="Server Status", border_style="bright_black"),
            table,
            Panel(recent or Text("No subtitles yet...", style="dim"), title="Recent Subtitles", border_style="blue")
        )

    async def start(self):
        async with serve(
            self.handler,
            host=self.host,
            port=self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=self.max_size,
        ) as server:
            self.console.print(
                f"[bold green]Server running on[/] ws://{self.host}:{self.port}",
                highlight=False
            )
            logger.info("[bold cyan]Live Japanese → English Subtitles Server started[/]")

            with Live(
                self.make_status_renderable(),
                refresh_per_second=4,
                console=self.console,
                screen=False,
            ) as live:
                try:
                    await server.serve_forever()
                except KeyboardInterrupt:
                    logger.info("[yellow]Shutdown requested — closing server...[/]")
                    await asyncio.sleep(0.5)


if __name__ == "__main__":
    server = JPSubtitlesServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/]")
