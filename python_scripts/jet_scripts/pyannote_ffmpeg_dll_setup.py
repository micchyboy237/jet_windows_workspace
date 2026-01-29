import os
from pathlib import Path

ffmpeg_bin = Path(
    r"C:\Users\druiv\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin"
)

if not ffmpeg_bin.exists():
    raise RuntimeError(f"FFmpeg bin not found: {ffmpeg_bin}")

os.add_dll_directory(str(ffmpeg_bin))

# now import/instantiate your model
from pyannote.audio import Inference
