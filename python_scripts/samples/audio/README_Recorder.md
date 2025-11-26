# FFmpeg Cross-Platform Microphone Recorder

A tiny, robust, **fully cross-platform** Python utility to record microphone audio directly to WAV using **FFmpeg**.

Works out-of-the-box on:

- macOS (Intel & Apple Silicon M1/M2/M3)
- Windows 10 / 11 (your Ryzen 5 3600 machine included)
- Linux (ALSA + PulseAudio)

No fragile device-name guessing, no PortAudio dependencies — just pure FFmpeg with correct syntax per OS.

## Features

- Automatically chooses the right FFmpeg input format:
  - macOS → `avfoundation`
  - Windows → `dshow`
  - Linux → `alsa` (fallback: `default`)
- Lists real, human-readable audio devices on **all** platforms
- Uses simple numeric indices (`"0"`, `"1"`, etc.) — same API everywhere
- Records 16-bit PCM WAV at 44.1 kHz (CD quality)
- Configurable duration, sample rate, channels
- Returns `subprocess.Popen` object for full control (monitor, kill early, pipe output)
- Clear console feedback with emojis
- Zero external Python audio libraries required

## Installation

### 1. Install FFmpeg

| OS      | Command / Link                                                                           |
| ------- | ---------------------------------------------------------------------------------------- |
| Windows | Download static build → https://www.gyan.dev/ffmpeg/builds/<br>or `choco install ffmpeg` |
| macOS   | `brew install ffmpeg`                                                                    |
| Linux   | `sudo apt install ffmpeg` (Ubuntu/Debian)<br>`sudo dnf install ffmpeg` (Fedora)          |

Make sure `ffmpeg` is in your `PATH`.

### 2. Install Python package (optional)

```bash
pip install sounddevice  # only needed if you want to compare with the sounddevice version
```

No extra packages required for this script — pure stdlib + subprocess.

## Usage

```python
from pathlib import Path
from ffmpeg_mic_recorder import record_mic_stream

# Record 10 seconds using default microphone (index "0")
output_file = Path("recordings/my_voice.wav")
output_file.parent.mkdir(exist_ok=True)

process = record_mic_stream(
    duration=10,
    output_file=output_file,
    audio_index="0"   # change to "1", "2", etc. to pick another mic
)

if process:
    print("Recording started...")
    process.wait()  # blocks until done
    print("Done!")
```

### Example Output on Windows

```text
Available Windows audio devices: ['Microphone (Realtek(R) Audio)', 'Microphone (USB Audio Device)', 'Headset Microphone (Realtek Audio)']
Recording 10s → my_voice.wav
   Using: dshow | audio="Microphone (Realtek(R) Audio)"
```

### Example Output on macOS

```text
Available macOS audio devices: ['Built-in Microphone', 'BlackHole 2ch', 'ZoomAudioDevice']
Recording 10s → my_voice.wav
   Using: avfoundation | :0
```

## API Reference

```python
record_mic_stream(
    duration: int,
    output_file: Path,
    audio_index: str = "0"
) -> Optional[subprocess.Popen]
```

- `duration` → seconds to record
- `output_file` → destination `.wav` file (will be overwritten)
- `audio_index` → `"0"` = default mic, `"1"` = second mic, etc.
- Returns `Popen` object or `None` on failure

## Project Structure

```
ffmpeg_mic_recorder.py    ← main module (copy this file)
recordings/               ← example output folder
README.md                 ← this file
```

## Why FFmpeg Instead of sounddevice?

| Feature                      | FFmpeg version (this one)     | sounddevice version               |
| ---------------------------- | ----------------------------- | --------------------------------- |
| No native binaries needed    | Yes (just ffmpeg.exe)         | Needs PortAudio DLLs              |
| Guaranteed 16-bit WAV        | Yes                           | Yes                               |
| Works behind corporate proxy | Yes (uses system FFmpeg)      | Sometimes issues                  |
| Stream / pipe in pipelines   | Yes (full subprocess control) | Limited                           |
| Zero Python audio deps       | Yes                           | Needs `sounddevice` + `soundfile` |

Perfect for automation, servers, Docker, CI/CD, or when you already have FFmpeg installed.

## License

MIT License — feel free to use in personal or commercial projects.

---

Made with love for developers who just want their mic to work on **both** their Mac M1 and Windows gaming rig.

Enjoy recording! 🎙️✨
