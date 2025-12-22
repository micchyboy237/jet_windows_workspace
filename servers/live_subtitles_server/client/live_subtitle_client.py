import asyncio
import websockets
import sounddevice as sd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from typing import Optional

console = Console()

# Audio settings (Whisper native)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.5          # seconds — good balance for latency/network
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

SERVER_URL = "ws://127.0.0.1:8002/ws/subtitles"  # Updated to port 8002 (localhost)
# If server is on another machine: ws://192.168.x.x:8002/ws/subtitles

def find_loopback_device() -> Optional[int]:
    """Find the best available loopback / system audio capture device.
    Prioritizes VB-Audio virtual cable output (input device) or WASAPI loopback."""
    devices = sd.query_devices()
    console.print("[dim]Available audio input devices:[/dim]")
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            hostapi = sd.query_hostapis(dev['hostapi'])['name']
            console.print(f"  {i:2d}: {dev['name']} ({hostapi}) - channels: {dev['max_input_channels']}")
            input_devices.append((i, dev['name'].lower(), hostapi.lower()))

    # Priority order for system audio capture
    priority_hints = [
        "cable output",      # VB-Audio Virtual Cable / Point output → correct input device
        "vb-audio",
        "loopback",
        "stereo mix",
        "what u hear",
    ]

    for hint in priority_hints:
        for i, name, hostapi in input_devices:
            if hint in name or hint in hostapi:
                console.print(f"[green]Selected device {i}: {sd.query_devices(i, 'input')['name']}[/green]")
                return i

    # Fallback: default input device
    default_input = sd.default.device[0]
    if default_input is not None:
        console.print(f"[yellow]No preferred loopback found — using default input device {default_input}[/yellow]")
        return default_input

    console.print("[bold red]No input device available[/bold red]")
    return None


async def capture_and_stream():
    device_index = find_loopback_device()
    if device_index is None:
        console.print("[bold red]No suitable input device found. Exiting.[/bold red]")
        return

    console.print(Panel(
        f"[bold green]Streaming system audio via device {device_index} "
        f"({sd.query_devices(device_index, 'input')['name']}) to port 8002[/bold green]\n"
        f"Press Ctrl+C to stop"
    ))

    async with websockets.connect(SERVER_URL) as ws:
        console.print("[bold cyan]Connected to server — live subtitles active[/bold cyan]")

        def audio_callback(indata: np.ndarray, frames: int, time, status):
            if status:
                console.print(f"[yellow]Audio status: {status}[/yellow]")
            # Send mono int16 bytes (Whisper expects int16 PCM)
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)  # take first channel
            asyncio.create_task(ws.send(audio_int16.tobytes()))

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK_SIZE,
                device=device_index,
                callback=audio_callback,
                latency='low'
            ):
                console.print("[dim]Streaming... Ctrl+C to stop[/dim]")
                await asyncio.Future()  # run forever until interrupted
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Stopped by user[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    asyncio.run(capture_and_stream())