# translators/jp_en.py
from __future__ import annotations

import torch
from typing import Literal

from transformers import MarianMTModel, MarianTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
log = console.log

# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------
model_name = "Helsinki-NLP/opus-mt-ja-en"

with console.status("[bold magenta]Loading Helsinki-NLP/opus-mt-ja-en...", spinner="dots"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

log("[bold green]Model ready on[/]", f"[bold cyan]{device.upper()}[/]")

Strategy = Literal["sampling", "diverse_beam", "fast_sampling"]


def translate_ja_en_diverse(
    ja_text: str,
    *,
    n: int = 8,
    strategy: Strategy = "sampling",
    temperature: float = 0.9,
    top_p: float = 0.95,
    diversity_penalty: float = 1.2,
    seed: int | None = None,
) -> list[tuple[str, float]]:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(ja_text, return_tensors="pt", padding=True).to(device)

    # ==================================================================
    # 1. SAMPLING + accurate re-scoring (recommended)
    # ==================================================================
    if strategy == "sampling":
        log(f"[bold]Generating[/] {n} samples → [cyan]sampling + re-scoring[/]")

        candidates = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            num_beams=1,                    # fixes num_return_sequences > num_beams error
            max_length=512,
            no_repeat_ngram_size=2,
            early_stopping=False,
        )

        translations = tokenizer.batch_decode(candidates, skip_special_tokens=True)

        # Re-score each candidate
        encoded = tokenizer(
            translations,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**encoded, labels=encoded.input_ids)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = encoded.input_ids[..., 1:].contiguous()  # fixed incomplete line

            log_probs = torch.gather(
                torch.log_softmax(shift_logits, dim=-1),
                dim=2,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            # Mask padding
            pad_mask = shift_labels != tokenizer.pad_token_id
            log_probs = (log_probs * pad_mask).sum(dim=-1)

        results = list(zip(translations, log_probs.cpu().tolist()))
        results.sort(key=lambda x: x[1], reverse=True)  # highest prob first
        return results

    # ==================================================================
    # 2. DIVERSE BEAM SEARCH — FIXED VERSION
    # ==================================================================
    elif strategy == "diverse_beam":
        log(f"[bold]Running[/] diverse beam search → {n} diverse outputs")

        # Critical: total beams must be multiple of groups
        num_groups = n
        beams_per_group = 3  # good balance: quality + diversity
        num_beams = num_groups * beams_per_group

        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_beam_groups=num_groups,
            num_return_sequences=n,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        # sequences_scores already contains the true log-probability of each returned sequence
        log_probs = outputs.sequences_scores.cpu().tolist()

        translations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        results = sorted(zip(translations, log_probs), key=lambda x: x[1], reverse=True)
        return results

    # ==================================================================
    # 3. FAST SAMPLING – correct & robust (works with any n)
    # ==================================================================
    elif strategy == "fast_sampling":
        log(f"[bold]Fast sampling[/] → {n} quick outputs (sequential to avoid MPS limit)")

        from tqdm import tqdm

        translations = []
        seq_log_probs = []

        for i in tqdm(range(n), desc="Sampling sequences"):
            if seed is not None:
                torch.manual_seed(seed + i)  # Vary seed per sample for diversity

            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,  # Key: batch=1
                num_beams=1,
                max_length=512,
                no_repeat_ngram_size=2,
                return_dict_in_generate=True,
                output_scores=True,
            )

            trans = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            translations.append(trans)

            # Compute log prob from scores
            log_prob = 0.0
            if outputs.scores:
                log_probs_per_step = [torch.log_softmax(s, dim=-1) for s in outputs.scores]
                generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
                for step, token_id in enumerate(generated_ids):
                    if token_id.item() in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                        break
                    log_prob += log_probs_per_step[step][0, token_id].item()  # [0] for batch=1
            seq_log_probs.append(log_prob)

        results = sorted(
            zip(translations, seq_log_probs),
            key=lambda x: x[1],
            reverse=True,
        )

        return results

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def print_results(
    ja_text: str,
    results: list[tuple[str, float]],
    strategy: str,
    n_display: int = 6,
) -> None:
    table = Table(title=f"[bold magenta]{strategy.upper()} STRATEGY[/]", show_header=True)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Translation", style="white")
    table.add_column("Score", justify="right", style="green")

    for i, (text, score) in enumerate(results[:n_display], 1):
        if strategy == "fast_sampling":
            table.add_row(str(i), text, f"{score:.3f}")
        else:
            prob = torch.exp(torch.tensor(score)).item()
            table.add_row(str(i), text, f"{score:+.3f} → [yellow]{prob:.4f}[/]")

    console.print(table)
    if len(results) > n_display:
        console.print(f"   ... and {len(results) - n_display} more\n")


# ———————————————————————————— MAIN DEMO ————————————————————————————
if __name__ == "__main__":
    ja_text = "おい、そんな一気に冷たいものを食べると腹を壊す"

    rprint(Panel(f"[bold cyan]{ja_text}[/]", title="Original Japanese", border_style="bright_blue"))

    strategies: list[Strategy] = ["sampling", "diverse_beam", "fast_sampling"]

    for strat in strategies:
        console.rule(f"[bold red]{strat.upper()}[/]")

        results = translate_ja_en_diverse(
            ja_text,
            n=8,
            strategy=strat,
            temperature=0.95,
            top_p=0.92,
            diversity_penalty=1.5,
            seed=42,  # reproducible comparison
        )

        print_results(ja_text, results, strat, n_display=6)

        console.print()  # spacing