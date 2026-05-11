import argparse
from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel
from rich.syntax import Syntax
import json

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from translators.translate_jp_en_shisa_llama import translate_japanese_to_english

console = Console()

def main():
    DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_2_speakers.wav"

    parser = argparse.ArgumentParser(
        description="Transcribe Japanese audio using SenseVoiceSmall + translation"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",  # Makes it optional (positional but can be omitted)
        default=DEFAULT_AUDIO,
        help="Path to the audio file (default: recording_missav_20s.wav)"
    )
    
    args = parser.parse_args()

    # ====================== MODEL SETUP ======================
    model_dir = "FunAudioLLM/SenseVoiceSmall"

    model = AutoModel(
        model=model_dir,
        disable_update=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        hub="hf",
    )

    console.print(f"[bold cyan]Processing:[/] {args.audio_path}")

    # ====================== INFERENCE ======================
    res = model.generate(
        input=args.audio_path,
        cache={},
        language="ja",
        use_itn=False,
        batch_size=32,
        output_timestamp=True,
        merge_vad=True,
        merge_length_s=15,
    )

    # ====================== RICH INSPECTION ======================
    console.rule("[bold cyan]Full Result Structure")
    console.print(f"Type of res: [bold yellow]{type(res)}[/]")
    console.print(f"Length of res: [bold yellow]{len(res)}[/]")

    if res:
        for i, item in enumerate(res):
            console.rule(f"[bold magenta]Item {i}")
            console.print(f"Keys: {list(item.keys())}")
            
            console.print("\n[bold]Full Content:[/]")
            print(f"{item!r}")

    # ====================== TRANSCRIPTION & TRANSLATION ======================
    ja_text = rich_transcription_postprocess(res[0]["text"])
    en_text = translate_japanese_to_english(ja_text)["text"]

    console.rule("[bold green]Final Result")
    console.print(f"JA: [bold white]{ja_text}[/]")
    console.print(f"EN: [bold white]{en_text}[/]")

if __name__ == "__main__":
    main()