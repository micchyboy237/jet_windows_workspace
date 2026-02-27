import asyncio
import base64
import json
from typing import Any, Dict
import websockets
from rich.console import Console
from transcribe_jp_whisper import transcribe_japanese_whisper
from translate_jp_en_opus import translate_japanese_to_english

console = Console()
SAMPLE_RATE = 16000


async def handle_client(websocket):
    """Async handler for each WS client connection."""
    try:
        while True:
            message = await websocket.recv()
            data: Dict[str, Any] = json.loads(message)
            if data.get("type") != "audio":
                continue
            audio_bytes = base64.b64decode(data["audio_bytes"])
            client_id = data.get("client_id", "unknown")
            utterance_id = data.get("utterance_id", "unknown")
            segment_num = data.get("segment_num", 0)
            console.print(
                f"[bold green]Received audio from {client_id} (utt: {utterance_id}, seg: {segment_num})[/bold green]"
            )
            trans_result = transcribe_japanese_whisper(
                audio_bytes,
                SAMPLE_RATE,
                client_id=client_id,
                utterance_id=utterance_id,
                segment_num=segment_num,
            )
            jp_text = trans_result["text_ja"]
            trans_conf = trans_result["confidence"]
            trans_quality = trans_result["quality_label"]
            console.print(
                f"[dim]JP Text: {jp_text} (conf: {trans_conf:.2f}, quality: {trans_quality})[/dim]"
            )
            en_result = translate_japanese_to_english(jp_text, enable_scoring=True)
            en_text = en_result["text"]
            en_conf = en_result["confidence"]
            en_quality = en_result["quality"]
            response = {
                "type": "translation",
                "en_text": en_text,
                "jp_text": jp_text,
                "confidence": en_conf,
                "quality": en_quality,
            }
            await websocket.send(json.dumps(response))
            console.print(f"[bold green]Sent translation: {en_text}[/bold green]")
    except websockets.exceptions.ConnectionClosed:
        console.print("[yellow]Client disconnected[/yellow]")


async def main():
    server = await websockets.serve(handle_client, "0.0.0.0", 8765)
    console.print(
        "[bold cyan]WebSocket server started on ws://0.0.0.0:8765[/bold cyan]"
    )
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
