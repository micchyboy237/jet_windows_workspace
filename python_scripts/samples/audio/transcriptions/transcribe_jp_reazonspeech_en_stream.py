"""
Streaming Japanese transcription + English translation (Clean output)
"""

import time
import sys
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from reazonspeech.espnet.asr.transcribe import load_model, transcribe_stream
from reazonspeech.espnet.asr.audio import audio_from_path
from reazonspeech.espnet.asr.interface import TranscribeConfig

console = Console(highlight=False, soft_wrap=True)

DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

parser = argparse.ArgumentParser(description="Streaming JP transcription + EN translation")
parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH)
args = parser.parse_args()
audio_path = args.audio_path

console.print(
    Panel.fit(f"🎙️ Starting Streaming Transcription + Translation\n[dim]Audio: {audio_path}[/dim]",
              style="bold cyan", box=box.ROUNDED)
)

# Load ASR
model = load_model()
audio = audio_from_path(audio_path)
console.print("✅ ASR Model loaded.\n", style="dim")

# Load Translator
console.print("Loading Shisa translation model...", style="dim")
trans_model_name = "shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForCausalLM.from_pretrained(
    trans_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
console.print("✅ Translation model loaded.\n", style="dim")

start_time = time.perf_counter()
full_segments: list[dict] = []
previous_time = start_time


def translate_japanese_to_english(ja_text: str) -> str:
    """Improved cleaning for Shisa model"""
    if not ja_text or not ja_text.strip():
        return ""

    messages = [
        {"role": "system", "content": "You are a precise and accurate translator. Always translate Japanese text to natural, fluent English. Output ONLY the English translation. Do not add any explanations, Japanese text, or extra comments."},
        {"role": "user", "content": f"Translate the following text from Japanese to English:\nJapanese: {ja_text.strip()}"}
    ]

    prompt = trans_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = trans_tokenizer(prompt, return_tensors="pt").to(trans_model.device)

    outputs = trans_model.generate(
        **inputs,
        max_new_tokens=256,           # Reduced
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=trans_tokenizer.eos_token_id,
        eos_token_id=trans_tokenizer.eos_token_id,
    )

    generated = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # === Aggressive cleaning ===
    # 1. Take only after the last "assistant"
    if "assistant" in generated:
        generated = generated.split("assistant")[-1].strip()

    # 2. Remove any remaining system/user prompts
    generated = re.sub(r"(?i)system:.*?(?=user|assistant|$)", "", generated, flags=re.DOTALL)
    generated = re.sub(r"(?i)user:.*?(?=assistant|$)", "", generated, flags=re.DOTALL)

    # 3. Remove common leftover tags
    generated = generated.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    # 4. Final safety: if it still contains Japanese or prompt keywords, fallback
    if any(word in generated.lower() for word in ["system", "user", "translate the following", "japanese:"]):
        # Take the last line only
        lines = [line.strip() for line in generated.split("\n") if line.strip()]
        generated = lines[-1] if lines else generated

    return generated.strip()


# === Streaming Loop ===
import os
os.environ["TQDM_DISABLE"] = "1"

with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(),
              console=console, transient=True) as progress:
    
    task = progress.add_task("[cyan]Transcribing & Translating...", total=None)

    for segment in transcribe_stream(model, audio, config=TranscribeConfig(verbose=False)):
        now = time.perf_counter()
        process_delta = now - previous_time
        previous_time = now
        elapsed = now - start_time

        clean_ja = segment.text.strip()
        en_text = translate_japanese_to_english(clean_ja) if clean_ja else ""

        if clean_ja:
            full_segments.append({
                "start": segment.start_seconds,
                "end": segment.end_seconds,
                "ja": clean_ja,
                "en": en_text
            })

        speed_color = "green" if process_delta < 0.6 else "yellow" if process_delta < 1.5 else "red"

        # Clean output as requested
        console.print(
            Text.assemble(
                "[", (f"{segment.start_seconds:6.2f}", "cyan"),
                " → ", (f"{segment.end_seconds:6.2f}", "cyan"),
                f"]  (+{process_delta:.2f}s)", style=speed_color
            )
        )
        console.print(f"   JA: {clean_ja}", style="bold white")
        console.print(f"   EN: {en_text or '(no translation)'}", style="bold green")

        sys.stdout.flush()
        progress.update(task, advance=1)

# === Final Output ===
console.print("\n" + "═" * 80, style="bold magenta")
console.print("[bold green]📝 Final Bilingual Transcript[/bold green]\n")

for seg in full_segments:
    console.print(Text.assemble("[", (f"{seg['start']:6.2f}", "cyan"), " → ", (f"{seg['end']:6.2f}", "cyan"), "]"))
    console.print(f"   JA: {seg['ja']}", style="white")
    console.print(f"   EN: {seg['en']}", style="green")
    console.print()

console.print(f"\n✅ Done! Total time: [cyan]{time.perf_counter() - start_time:.2f}s[/cyan] | Segments: [cyan]{len(full_segments)}[/cyan]")
