# single_audio_transcription.py
import torch
import re
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Optional
from pathlib import Path
from rich.console import Console

from translators.translate_jp_en_opus import translate_japanese_to_english

console = Console()

# ────────────────────────────────────────────────
#  Configuration (you can move to config file later)
# ────────────────────────────────────────────────
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16_000

CHARS_TO_IGNORE = [
    ",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
    "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
    "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
    "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
    "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"
]
chars_to_ignore_regex = re.compile(f"[{re.escape(''.join(CHARS_TO_IGNORE))}]")

# ────────────────────────────────────────────────
#  Global (load once – reuse many times)
# ────────────────────────────────────────────────
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, local_files_only=True)
model.to(DEVICE)
model.eval()


def normalize_text(text: str) -> str:
    """Remove punctuation & convert to uppercase (matching your evaluation style)"""
    return re.sub(chars_to_ignore_regex, "", text).upper()


def transcribe_audio(
    audio_path: str | Path,
    normalize: bool = True,
    return_confidence: bool = False
) -> dict:
    """
    Transcribe one audio file with the Japanese wav2vec2 model.

    Args:
        audio_path: path to .wav / .mp3 / .m4a / ...
        normalize: whether to apply the same cleaning as in your eval script
        return_confidence: not implemented yet (needs logit → probability conversion)

    Returns:
        dict with keys: "text", "text_normalized" (if normalize=True), ...
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    console.rule(f"Transcribing {path.name}")

    # 1. Load & resample
    speech_array, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)

    # 2. Processor → model input
    inputs = processor(
        speech_array,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # 3. Inference
    with torch.inference_mode():
        logits = model(input_values, attention_mask=attention_mask).logits

    # 4. Decode
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]   # single item

    result = {"text": transcription}

    if normalize:
        result["text_normalized"] = normalize_text(transcription)

    # Future extension possibility
    if return_confidence:
        pass  # probs = torch.softmax(logits, dim=-1) → confidence logic

    console.print(f"[bold green]→[/] {result['text']}")
    if normalize:
        console.print(f"[dim]normalized →[/] {result.get('text_normalized', '')}")

    return result


# ────────────────────────────────────────────────
#  Example usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your file
    audio_file = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    try:
        result = transcribe_audio(audio_file, normalize=True)
        ja_text = result["text_normalized"]
        en_text = translate_japanese_to_english(ja_text)

        print("\nFinal result:")
        print(f"JA: {ja_text}")
        print(f"EN: {en_text}")
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")