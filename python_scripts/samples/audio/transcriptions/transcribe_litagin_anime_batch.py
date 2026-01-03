from typing import List, Dict, Sequence, Literal
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table

# =========================
# Config
# =========================
MODEL_ID = "litagin/anime-whisper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

GENERATE_KWARGS = {
    "language": "ja",
    "task": "transcribe",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
    "return_dict_in_generate": True,
    "output_scores": True,
}

# =========================
# Utilities for Confidence & Quality
# =========================
def calculate_confidence_score(
    token_logprobs: Sequence[float],
    *,
    clip_min: float = -5.0,
    clip_max: float = 0.0,
) -> float:
    if not token_logprobs:
        return 0.0
    clipped = np.clip(token_logprobs, clip_min, clip_max)
    probs = np.exp(clipped)
    mean_prob = float(np.mean(probs))
    return round(mean_prob * 100.0, 2)


QualityLabel = Literal["very low", "low", "medium", "high", "very high"]


def categorize_quality_label(confidence_score: float) -> QualityLabel:
    if confidence_score < 20.0:
        return "very low"
    if confidence_score < 40.0:
        return "low"
    if confidence_score < 60.0:
        return "medium"
    if confidence_score < 80.0:
        return "high"
    return "very high"


# =========================
# Batch Transcriber Class (Manual for full token-level control)
# =========================
class AnimeWhisperBatchTranscriber:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        batch_size: int = 4,  # Conservative default for GTX 1660
    ):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.generate_kwargs = GENERATE_KWARGS.copy()

    def _load_and_preprocess(self, paths: List[str]) -> Dict[str, torch.Tensor]:
        audios = []
        for path in paths:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr != 16000:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
                audio_tensor = torch.nn.functional.interpolate(
                    audio_tensor,
                    scale_factor=16000 / sr,
                    mode="linear",
                    align_corners=False,
                ).squeeze().numpy()
                audio = audio_tensor
            audios.append(audio)

        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device),
        }

    def _process_batch_output(
        self,
        outputs,
        batch_paths: List[str],
    ) -> List[Dict]:
        sequences = outputs.sequences
        transition_scores = self.model.compute_transition_scores(
            sequences, outputs.scores, normalize_logits=True
        )
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        results = []
        for idx, (text, path) in enumerate(zip(texts, batch_paths)):
            token_ids = sequences[idx].tolist()
            token_logprobs = transition_scores[idx].tolist()
            token_data = []
            for token_id, logprob in zip(token_ids, token_logprobs):
                piece = self.processor.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if piece.strip():
                    token_data.append({"token": piece, "logprob": float(logprob)})
            token_lps = [t["logprob"] for t in token_data]
            avg_logprob = sum(token_lps) / len(token_lps) if token_lps else float("nan")
            confidence = calculate_confidence_score(token_lps)
            quality = categorize_quality_label(confidence)

            # === NEW: Calculate duration ===
            audio, _ = sf.read(path)  # Load to get length (fast metadata read for most formats)
            duration_seconds = len(audio) / sf.info(path).samplerate
            duration_formatted = f"{duration_seconds:.2f}s"

            results.append({
                "path": path,
                "text": text,
                "avg_logprob": avg_logprob,
                "confidence_score": confidence,
                "quality": quality,
                "tokens": token_data,
                "duration_seconds": round(duration_seconds, 3),      # Raw float for potential further use
                "duration": duration_formatted,                     # Human-readable string for display
            })
        return results

    def transcribe(self, audio_paths: List[str]) -> List[Dict]:
        all_results: List[Dict] = []
        for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="Transcribing batches"):
            batch_paths = audio_paths[i:i + self.batch_size]
            batch_inputs = self._load_and_preprocess(batch_paths)

            with torch.no_grad():
                outputs = self.model.generate(
                    batch_inputs["input_features"],
                    attention_mask=batch_inputs["attention_mask"],
                    **self.generate_kwargs,
                )

            batch_results = self._process_batch_output(outputs, batch_paths)
            all_results.extend(batch_results)

        return all_results


# =========================
# Pretty printing helpers
# =========================
def print_results(results: List[Dict]):
    table = Table(title="Anime Whisper Transcription Results")
    table.add_column("File", style="cyan")
    table.add_column("Duration", justify="right", style="magenta")  # NEW
    table.add_column("Text", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Quality", style="bold")
    for r in results:
        table.add_row(
            Path(r["path"]).name,
            r["duration"],                                      # NEW
            r["text"],
            f"{r['confidence_score']:.2f}%",
            r["quality"],
        )
    rprint(table)
    # Optional: detailed token view for first result
    if results:
        rprint("\n[bold]Detailed tokens (first file):[/bold]")
        rprint(json.dumps(results[0]["tokens"], indent=2, ensure_ascii=False))


# =========================
# Main block
# =========================
if __name__ == "__main__":
    from utils import resolve_audio_paths

    audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server_for_test"
    audio_paths = resolve_audio_paths(audio_dir, recursive=True)
    audio_paths = audio_paths[:5]

    if not audio_paths:
        rprint("[red]No audio files found in the directory![/red]")
    else:
        rprint(f"[blue]Found {len(audio_paths)} audio files. Starting batch transcription...[/blue]")

        transcriber = AnimeWhisperBatchTranscriber(batch_size=4)  # Adjust based on VRAM
        results = transcriber.transcribe([str(p) for p in audio_paths])

        print_results(results)

        # Optional: save full results to JSON
        output_json = "transcription_results.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        rprint(f"\n[green]Full results saved to {output_json}[/green]")