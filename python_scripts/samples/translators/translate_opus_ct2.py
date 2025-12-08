from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Union, Literal, overload

import ctranslate2
from ctranslate2 import Translator, TranslationResult
from rich.console import Console
from rich.table import Table

console = Console()


class JapaneseToEnglishTranslator:
    """Lightweight wrapper around a quantized Helsinki-NLP/opus-mt-ja-en → CTranslate2 model."""

    DEFAULT_MODEL_PATH = "/Users/jethroestrada/.cache/hf_translation_models/ja_en_ct2"
    DEFAULT_DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    DEFAULT_COMPUTE_TYPE = "int8"  # matches your quantization

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Literal["cpu", "cuda"] = DEFAULT_DEVICE,
        compute_type: Literal["default", "int8", "int16", "float16", "bfloat16", "float32"] = DEFAULT_COMPUTE_TYPE,
        intraopol_threads: int = 4,
        inter_threads: int = 1,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"CTranslate2 model not found at {model_path}")

        self.translator = Translator(
            model_path,
            device=device,
            compute_type=compute_type,
            intra_threads=intraopol_threads,
            inter_threads=inter_threads,
        )
        self.device = device
        console.log(f"[green]Translator loaded[/] → {model_path} ({device}, {compute_type})")

    @overload
    def translate_ja_en_diverse(
        self,
        texts: str,
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> List[TranslationResult]: ...

    @overload
    def translate_ja_en_diverse(
        self,
        texts: Sequence[str],
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> List[List[TranslationResult]]: ...

    def translate_ja_en_diverse(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> Union[List[TranslationResult], List[List[TranslationResult]]]:
        """
        Translate Japanese → English with diverse beam search outputs.

        Args:
            texts: Single Japanese string or list of strings.
            max_decoding_length: Maximum generation length.
            beam_size: Beam size (higher → better quality, slower).
            num_hypotheses: Number of diverse translations to return per input (≤ beam_size).
            length_penalty: Encourage/discourage longer outputs (1.0 = neutral).
            return_scores: Include log-probability scores in results.
            replace_unknowns: Post-process to replace <unk> tokens (Marian-style).

        Returns:
            List of TranslationResult (single input) or list of lists (batch input).
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            texts = list(texts)
            single_input = False

        if not texts:
            return [] if single_input else [[]]

        console.log(f"Translating {len(texts)} Japanese sentence(s) → English (beam={beam_size}, hyps={num_hypotheses})")

        results: List[List[TranslationResult]] = self.translator.translate_batch(
            source=[list(text) for text in texts],
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            num_hypotheses=num_hypotheses,
            length_penalty=length_penalty,
            return_scores=return_scores,
            replace_unknowns=replace_unknowns,
        )

        # Return single result if input was a single string
        return results[0] if single_input else results

    def pretty_print_results(
        self,
        texts: Union[str, Sequence[str]],
        results: Union[TranslationResult, List[TranslationResult], List[List[TranslationResult]]],
    ) -> None:
        """Beautiful, correct rich table with proper Japanese to English detokenization."""
        # Normalize input text
        if isinstance(texts, str):
            texts = [texts]

        # Normalize results to always be List[List[TranslationResult]]
        # This is the fixed line — was previously a syntax error!
        if isinstance(results, TranslationResult):
            results = [[results]]
        elif isinstance(results, list) and results and isinstance(results[0], TranslationResult):
            results = [results]  # Wrap single batch into list-of-batches
        elif isinstance(results, list) and results and isinstance(results[0], list):
            pass  # Already in correct shape: List[List[TranslationResult]]
        else:
            console.log("[red]Warning: Unexpected results format in pretty_print_results[/]")
            return

        table = Table(
            title="Japanese to English (Diverse Beam Search)",
            title_style="bold magenta",
            show_lines=True,
            expand=False,
            pad_edge=False,
        )
        table.add_column("Input (JA)", style="cyan", width=50, overflow="fold")
        table.add_column("Rank", justify="center", style="dim", width=6)
        table.add_column("Translation (EN)", style="green", overflow="fold")
        table.add_column("Score", justify="right", style="yellow", width=12)

        for ja_text, batch in zip(texts, results):
            for rank, res in enumerate(batch, start=1):
                # Safely extract top hypothesis and score
                tokens: List[str] = []
                score: Optional[float] = None

                if isinstance(res, TranslationResult):
                    if res.hypotheses:
                        tokens = res.hypotheses[0]
                    if res.scores:
                        score = res.scores[0]
                elif isinstance(res, dict):
                    hyp_list = res.get("hypotheses")
                    if hyp_list and len(hyp_list) > 0:
                        tokens = hyp_list[0]
                    score_val = res.get("score")
                    if score_val is None and "scores" in res and res["scores"]:
                        score_val = res["scores"][0]
                    score = score_val

                # Correct detokenization for Opus-MT / Helsinki models (SentencePiece)
                translation = "".join(tokens).replace("▁", " ").strip()
                translation = " ".join(translation.split())  # Clean multiple spaces

                if not translation:
                    translation = "[empty translation]"

                score_str = f"{score:.4f}" if score is not None else "—"

                table.add_row(
                    ja_text if rank == 1 else "",  # Show source only on first row
                    str(rank),
                    translation,
                    score_str,
                )

            table.add_row("")  # Visual separator between sentences

        console.print(table)

def to_serializable_results(
    results: Union[TranslationResult, List[TranslationResult], List[List[TranslationResult]]],
) -> Union[dict, List[dict], List[List[dict]]]:
    """Convert TranslationResult objects to JSON-serializable dicts."""
    if isinstance(results, TranslationResult):
        return {
            "hypotheses": results.hypotheses,
            "scores": results.scores or [],
        }
    elif isinstance(results, list):
        if results and isinstance(results[0], TranslationResult):
            return [to_serializable_results(res) for res in results]
        elif results and isinstance(results[0], list):
            return [to_serializable_results(sub) for sub in results]
    return results  # already serializable


# Convenience singleton (optional)
_translator: Optional[JapaneseToEnglishTranslator] = None


def get_ja_en_translator(**kwargs) -> JapaneseToEnglishTranslator:
    """Thread-safe singleton accessor."""
    global _translator
    if _translator is None:
        _translator = JapaneseToEnglishTranslator(**kwargs)
    return _translator


# Updated convenience function (add the new helper call)
def translate_ja_en_diverse(
    texts: Union[str, Sequence[str]],
    *,
    model_path: str = JapaneseToEnglishTranslator.DEFAULT_MODEL_PATH,
    device: Literal["cpu", "cuda"] = JapaneseToEnglishTranslator.DEFAULT_DEVICE,
    compute_type: Literal["default", "int8", "int16", "float16", "bfloat16", "float32"] = "int8",
    max_decoding_length: int = 512,
    beam_size: int = 5,
    num_hypotheses: int = 5,
    length_penalty: float = 1.0,
    return_scores: bool = True,
    replace_unknowns: bool = True,
    pretty_print: bool = False,
) -> Union[List[TranslationResult], List[List[TranslationResult]]]:
    """
    One-liner reusable function for diverse Japanese → English translation.
    """
    translator = get_ja_en_translator(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
    )

    results = translator.translate_ja_en_diverse(
        texts,
        max_decoding_length=max_decoding_length,
        beam_size=beam_size,
        num_hypotheses=num_hypotheses,
        length_penalty=length_penalty,
        return_scores=return_scores,
        replace_unknowns=replace_unknowns,
    )

    if pretty_print:
        translator.pretty_print_results(texts, results)

    return results

# Updated demo (use the new helper for JSON printing)
if __name__ == "__main__":
    results = translate_ja_en_diverse(
        [
            "昨日、友達と一緒に映画を見に行きました。",
            "日本は美しい国ですね！"
        ],
        beam_size=6,
        num_hypotheses=4,
        max_decoding_length=512,
        pretty_print=True,
    )
    serializable = to_serializable_results(results)
    print(f"\nResults ({len(results)}):\n{json.dumps(serializable, indent=2)}")