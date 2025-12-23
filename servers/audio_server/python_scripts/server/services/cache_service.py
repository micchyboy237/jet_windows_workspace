from typing import Optional
from faster_whisper import WhisperModel

from python_scripts.server.utils.logger import get_logger

log = get_logger("batch_transcription")

transcriber_model: Optional[WhisperModel] = None

# Exposed startup function called from main.py
async def load_model(
    model_name: str = "small",
    device: str = "cuda",
    compute_type: str = "int8"
) -> WhisperModel:
    global transcriber_model
    if transcriber_model:
        log.info(f"Reusing transcriber model cache hit [bold magenta]{model_name}[/bold magenta] (CUDA float32)")
        return transcriber_model
    log.info(f"Loading model [bold magenta]{model_name}[/bold magenta] (CUDA float32)")
    # model = WhisperModel(
    #     model_name,
    #     device="cuda",
    #     compute_type="float32",
    # )
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    transcriber_model = model
    log.info("[bold green]Model loaded and cached successfully[/bold green]")
    return model