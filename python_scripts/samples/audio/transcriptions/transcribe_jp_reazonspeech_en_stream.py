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
    """Improved translation with few-shot examples for different discourse types
    (casual greeting, question, affirmative response, instruction) using the
    exact Jinja chat template. Prevents role-playing and forces pure English output."""
    if not ja_text or not ja_text.strip():
        return ""

    # Improved system prompt with few-shot examples for different discourse types
    system_prompt = """You are a precise and accurate translator. Always translate Japanese text to natural, fluent English. Output ONLY the English translation. Do not add any explanations, Japanese text, extra comments, or introductory phrases like 'Here's the translation:'.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        # Improved user prompt: still raw JA (matches few-shot style) but now clearly
        # positioned after the "Now translate the following..." instruction in system.
        {"role": "user", "content": ja_text.strip()}
    ]

    prompt = trans_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nRaw input:\n--- START OF INPUT ---\n{prompt}\n--- END OF INPUT ---\n")

    inputs = trans_tokenizer(prompt, return_tensors="pt").to(trans_model.device)

    with torch.no_grad():
        outputs = trans_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=trans_tokenizer.eos_token_id,
            eos_token_id=trans_tokenizer.eos_token_id,
        )
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        generated = trans_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    generated = generated.strip('"\'')  # remove outer quotes
    generated = re.sub(r'\s+', ' ', generated).strip()
    # Extra safety: if any Japanese remains, keep only the first sentence
    if any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in generated):
        sentences = re.split(r'(?<=[.!?])\s+', generated)
        generated = sentences[0] if sentences else generated

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
