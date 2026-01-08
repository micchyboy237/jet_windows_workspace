import os
import asyncio
import json
from typing import Dict

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from rich.console import Console
from rich.table import Table

console = Console()
app = FastAPI()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Set your key here or in .env
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?model=nova-2&language=ja&translate=en"

async def deepgram_transcribe(websocket: WebSocket):
    """Proxy audio from client WebSocket to Deepgram and relay transcripts back."""
    try:
        deepgram_ws = await websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        )
        console.print("[green]Connected to Deepgram[/green]")

        async def receive_from_client():
            while True:
                data = await websocket.receive_bytes()
                await deepgram_ws.send(data)

        async def receive_from_deepgram():
            while True:
                message = await deepgram_ws.recv()
                transcript_data = json.loads(message)
                if "channel" in transcript_data and "alternatives" in transcript_data["channel"]:
                    alternatives = transcript_data["channel"]["alternatives"][0]
                    text = alternatives.get("transcript", "").strip()
                    if text and transcript_data.get("is_final"):
                        await websocket.send_text(f"[FINAL] {text}")
                        console.print(f"[bold green]Subtitle:[/bold green] {text}")
                    elif text:
                        await websocket.send_text(f"[PARTIAL] {text}")

        await asyncio.gather(receive_from_client(), receive_from_deepgram())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await websocket.close()

@app.websocket("/ws/subtitles")
async def subtitles_endpoint(websocket: WebSocket):
    await websocket.accept()
    console.print("[blue]Client connected[/blue]")
    try:
        await deepgram_transcribe(websocket)
    except WebSocketDisconnect:
        console.print("[yellow]Client disconnected[/yellow]")