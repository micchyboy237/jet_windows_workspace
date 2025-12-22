from __future__ import annotations

from pathlib import Path

import torch
from rich import print as rprint
from tqdm import tqdm
from transformers import pipeline


def build_s2t_pipeline(
    target_lang: str = "eng_Latn",
    translation_model: str = "facebook/nllb-200-distilled-600M",
    chunk_length_s: int = 15,
    device: int | str = 0 if torch.cuda.is_available() else -1,
) -> pipeline:
    """
    Builds the correct cascaded Japanese Speech-to-Text Translation pipeline.
    
    Fixed parameters:
    - model: "japanese-asr/ja-cascaded-s2t-translation" (dedicated Japanese ASR)
    - tgt_lang: NLLB target language code
    """
    rprint("[bold yellow]Loading Japanese S2T Translation pipeline... (this may take a minute)[/bold yellow]")
    
    pipe = pipeline(
        model="japanese-asr/ja-cascaded-s2t-translation",  # Correct model ID
        model_kwargs={"attn_implementation": "sdpa"},
        model_translation=translation_model,
        tgt_lang=target_lang,
        chunk_length_s=chunk_length_s,
        trust_remote_code=True,
        device=device,
    )
    
    if torch.cuda.is_available():
        rprint(f"[bold green]Pipeline loaded on GPU: {torch.cuda.get_device_name(0)}[/bold green]")
    else:
        rprint("[bold magenta]Pipeline loaded on CPU[/bold magenta]")
    
    return pipe


def transcribe_and_translate(
    pipe: pipeline,
    audio_path: str | Path,
) -> str:
    """Processes a single Japanese audio file and returns English translation."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    rprint(f"[bold cyan]Processing audio:[/bold cyan] {audio_path.name}")
    
    # Returns list of dicts with 'text' key for the translated text
    result = pipe(str(audio_path))
    
    # Handle possible multi-chunk output
    translation = " ".join(item["text"].strip() for item in result if item["text"].strip())
    rprint(f"[bold green]English Translation:[/bold green] {translation or '[No output]'}")
    
    return translation


def main() -> None:
    pipe = build_s2t_pipeline()

    # Update with your actual audio paths
    example_audios = [
        Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"),
        # Add more paths as needed
    ]

    for audio_file in tqdm(example_audios, desc="Translating audio files"):
        try:
            transcribe_and_translate(pipe, audio_file)
        except Exception as e:
            rprint(f"[bold red]Error processing {audio_file.name}: {e}[/bold red]")


if __name__ == "__main__":
    main()