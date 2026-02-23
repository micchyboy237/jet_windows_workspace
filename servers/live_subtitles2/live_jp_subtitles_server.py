"""Live Japanese to English subtitles server. Handles ASR and translation via WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from utils.asr import ASRTranscriber
from utils.audio_utils import AudioStreamProcessor
from utils.translation import JapaneseToEnglishTranslator
from websockets.server import serve

console = Console()
logging.basicConfig(
    level="INFO", handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


class JPSubtitlesServer:
    """Server for processing live audio to subtitles."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.asr = ASRTranscriber(model_size="medium")  # "large-v3" if VRAM > 6GB
        self.translator = JapaneseToEnglishTranslator()
        self.processors: dict[int, AudioStreamProcessor] = {}
        self.console = console

    async def handler(self, websocket):
        """Handle one client connection."""
        client_id = id(websocket)
        logger.info(f"Client connected: {client_id}")
        processor = AudioStreamProcessor(self.asr, self.translator)
        self.processors[client_id] = processor
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    chunk = np.frombuffer(message, dtype=np.float32)
                    result = processor.process_chunk(chunk)
                    if result:
                        en_text, jp_text = result
                        response = {
                            "type": "subtitle",
                            "text": en_text,
                            "jp_text": jp_text,
                        }
                        await websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self.processors.pop(client_id, None)
            logger.info(f"Client disconnected: {client_id}")

    async def start(self):
        """Start the WebSocket server."""
        async with serve(self.handler, self.host, self.port):
            self.console.print(
                f"[green]Server running on ws://{self.host}:{self.port}[/green]"
            )
            self.console.print("Ready for client connection. Use Ctrl+C to stop.")
            await asyncio.Future()  # run forever


if __name__ == "__main__":
    server = JPSubtitlesServer()
    asyncio.run(server.start())
