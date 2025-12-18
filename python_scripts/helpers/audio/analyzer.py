from __future__ import annotations
from typing import List, Dict, Any, Literal
import numpy as np
from rich.table import Table
from rich.console import Console
from rich import box
from rich.text import Text

from ctranslate2 import StorageView
from scipy.stats import entropy

from translator_types import TranslationResult

console = Console()

AlignmentCategory = Literal["perfect", "good", "split", "reordered", "diffuse", "suspicious"]

def analyze_translation_results(
    results: List[TranslationResult],  # List[TranslationResult] from ctranslate2
    source_tokens: List[List[str]] | None = None,
    min_peak_threshold: float = 0.4,
    min_split_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Analyze attention + scores from ctranslate2.translate_batch(...)
    Returns rich human-readable insights per translation.
    """
    analyses = []

    for idx, res in enumerate(results):
        hyp = res.hypotheses[0]  # best hypothesis
        tokens = hyp.tokens if hasattr(hyp, "tokens") else hyp
        score = res.scores[0]
        attn = res.attention[0]  # [hyp][step] → [src_len]

        num_tgt = len(tokens)
        num_src = len(attn[0]) if attn else 0
        avg_logprob = score / num_tgt
        confidence = np.exp(avg_logprob)

        # Compute alignment peaks
        alignments = []
        categories: List[AlignmentCategory] = []

        for t, attn_row in enumerate(attn):
            row = np.array(attn_row)
            if row.sum() < 0.9:
                row = row / row.sum()  # renormalize if needed

            peak_idx = int(np.argmax(row))
            peak_val = float(row[peak_idx])

            # Check for split attention (two strong peaks)
            sorted_idx = np.argsort(row)[-2:]
            if len(sorted_idx) >= 2 and abs(sorted_idx[0] - sorted_idx[1]) <= 2:
                split_val = row[sorted_idx[0]] + row[sorted_idx[1]]
                if split_val > peak_val + 0.1:
                    peak_idx = tuple(int(i) for i in sorted_idx)
                    peak_val = float(split_val)

            # Categorize
            if isinstance(peak_idx, tuple):
                cat: AlignmentCategory = "split"
            elif peak_val >= 0.7:
                cat = "perfect"
            elif peak_val >= min_peak_threshold:
                cat = "good"
            elif peak_val >= 0.25:
                cat = "reordered"
            elif peak_val < 0.15:
                cat = "suspicious"
            else:
                cat = "diffuse"

            alignments.append({
                "target_token": tokens[t],
                "source_pos": peak_idx,
                "peak_weight": round(peak_val, 3),
                "category": cat,
            })
            categories.append(cat)

        # Summary stats
        cat_counts = {c: categories.count(c) for c in set(categories)}
        alignment_score = (
            3 * cat_counts.get("perfect", 0) +
            2 * cat_counts.get("good", 0) +
            1 * cat_counts.get("split", 0) +
            0 * cat_counts.get("reordered", 0) -
            1 * cat_counts.get("diffuse", 0) -
            3 * cat_counts.get("suspicious", 0)
        ) / num_tgt

        analysis = {
            "idx": idx,
            "source_tokens": source_tokens[idx] if source_tokens and idx < len(source_tokens) else None,
            "target": " ".join(tokens).replace("▁", " ").strip(),
            "score": round(score, 4),
            "avg_logprob": round(avg_logprob, 4),
            "confidence_pct": round(confidence * 100, 2),
            "length": num_tgt,
            "alignments": alignments,
            "alignment_score": round(alignment_score, 3),
            "alignment_summary": cat_counts,
            "verdict": "EXCELLENT" if score > -1.0 and alignment_score > 1.8 else
                      "GOOD" if score > -3.0 else
                      "UNCERTAIN" if score > -8.0 else "POOR"
        }
        analyses.append(analysis)

    return analyses

# === Rating helpers (return styled Text with (LEVEL)) ===
def rated_score(score: float) -> Text:
    if score > -0.8:
        level = "HIGH"
        color = "green"
    elif score > -2.5:
        level = "MED"
        color = "yellow"
    else:
        level = "LOW"
        color = "red"
    return Text(f"{score:.3f} ({level})", style=f"bold {color}")

def rated_confidence(pct: float) -> Text:
    if pct >= 95:
        level = "HIGH"
        color = "green"
    elif pct >= 85:
        level = "MED"
        color = "yellow"
    else:
        level = "LOW"
        color = "red"
    return Text(f"{pct:.1f}% ({level})", style=f"bold {color}")

def rated_alignment(score: float) -> Text:
    if score >= 2.0:
        level = "HIGH"
        color = "green"
    elif score >= 1.2:
        level = "MED"
        color = "yellow"
    else:
        level = "LOW"
        color = "red"
    return Text(f"{score:.2f} ({level})", style=f"bold {color}")

def styled_verdict(verdict: str) -> Text:
    styles = {
        "EXCELLENT": "bold bright_green",
        "GOOD": "bold green",
        "UNCERTAIN": "bold yellow",
        "POOR": "bold red",
    }
    return Text(verdict, style=styles.get(verdict, "dim"))

def print_analysis_table(analyses: List[Dict[str, Any]]):
    table = Table(
        title="Translation Quality & Attention Analysis",
        box=box.ROUNDED,
        title_style="bold magenta",
        header_style="bold white"
    )
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Target", style="white", width=50)
    table.add_column("Score", justify="right", width=14)
    table.add_column("Conf%", justify="right", width=14)
    table.add_column("Verdict", justify="center", width=12)
    table.add_column("Align", justify="right", width=14)
    table.add_column("Summary", style="dim")

    for a in analyses:
        table.add_row(
            str(a["idx"]),
            a["target"][:47] + "..." if len(a["target"]) > 50 else a["target"],
            rated_score(a["score"]),
            rated_confidence(a["confidence_pct"]),
            styled_verdict(a["verdict"]),
            rated_alignment(a["alignment_score"]),
            ", ".join(f"{k}: {v}" for k, v in a["alignment_summary"].items() if v > 0)
        )

    console.print(table)


def analyze_logits(logits_per_token: List[List[np.ndarray]]) -> Dict[str, Any]:
    """
    Analyze token-level model confidence from logits_vocab.
    Handles CTranslate2 StorageView → NumPy conversion automatically.
    Input: List of [hypothesis] → List of np.ndarray (vocab_size,)
    """
    all_confidences = []
    all_entropies = []
    all_surprisals = []
    risky_count = 0
    high_entropy_count = 0

    for hyp_logits in logits_per_token:  # One hypothesis at a time
        hyp_conf = []
        hyp_entropy = []

        for logits in hyp_logits:
            # Auto-convert StorageView if needed
            if isinstance(logits, StorageView):
                logits = logits.to_numpy()
            elif not isinstance(logits, np.ndarray):
                continue  # Skip malformed

            # Numerical stability: subtract max
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / (probs.sum() + 1e-12)

            pred_idx = int(np.argmax(logits))
            pred_prob = float(probs[pred_idx])

            token_entropy = float(entropy(probs, base=np.e))
            token_surprisal = -np.log(max(pred_prob, 1e-12))

            hyp_conf.append(pred_prob)
            hyp_entropy.append(token_entropy)
            all_surprisals.append(token_surprisal)

            if pred_prob < 0.1:
                risky_count += 1
            if token_entropy > 3.0:
                high_entropy_count += 1

        all_confidences.extend(hyp_conf)
        all_entropies.extend(hyp_entropy)

    if not all_confidences:
        return {"error": "No valid logits found"}

    return {
        "avg_token_confidence": round(np.mean(all_confidences), 4),
        "min_token_confidence": round(np.min(all_confidences), 4),
        "avg_entropy": round(np.mean(all_entropies), 3),
        "risky_tokens_count": risky_count,
        "high_entropy_count": high_entropy_count,
        "avg_surprisal": round(np.mean(all_surprisals), 3),
        "total_tokens": len(all_confidences),
    }

def print_analysis_table_with_logits(analyses: List[Dict[str, Any]]):
    table = Table(
        title="Translation Quality & Attention + Logits Analysis",
        box=box.ROUNDED,
        title_style="bold magenta"
    )
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Target", style="white", width=45)
    table.add_column("Score", justify="right", width=15)
    table.add_column("Conf%", justify="right", width=15)
    table.add_column("Verdict", justify="center", width=12)
    table.add_column("Align", justify="right", width=14)
    table.add_column("Token Conf", justify="right", width=16)
    table.add_column("Entropy", justify="right", width=12)
    table.add_column("Risk", justify="center", width=10)
    table.add_column("Summary", style="dim")

    for a in analyses:
        # Extract logits insights if available
        logits_info = {}
        if hasattr(a["result"], "logits_vocab") and a["result"].logits_vocab:
            logits_info = analyze_logits(a["result"].logits_vocab[0])

        token_conf = logits_info.get("avg_token_confidence", "N/A")
        entropy_val = logits_info.get("avg_entropy", "N/A")
        risk = ""
        if logits_info:
            risks = []
            if logits_info["risky_tokens_count"] > 0:
                risks.append(f"{logits_info['risky_tokens_count']} low-conf")
            if logits_info["high_entropy_count"] > 0:
                risks.append("high-entropy")
            risk = " / ".join(risks) if risks else "clean"

        # Rating styles
        def rated_token_conf(val):
            if val == "N/A": return Text("N/A", style="dim")
            if val >= 0.9: return Text(f"{val:.3f} (HIGH)", style="bold green")
            if val >= 0.7: return Text(f"{val:.3f} (MED)", style="bold yellow")
            return Text(f"{val:.3f} (LOW)", style="bold red")

        def rated_entropy(val):
            if val == "N/A": return Text("N/A", style="dim")
            if val < 1.5: return Text(f"{val:.2f} (LOW)", style="bold green")
            if val < 3.0: return Text(f"{val:.2f} (MED)", style="bold yellow")
            return Text(f"{val:.2f} (HIGH)", style="bold red")

        table.add_row(
            str(a["idx"]),
            a["target"][:42] + "..." if len(a["target"]) > 42 else a["target"],
            rated_score(a["score"]),
            rated_confidence(a["confidence_pct"]),
            styled_verdict(a["verdict"]),
            rated_alignment(a["alignment_score"]),
            rated_token_conf(token_conf),
            rated_entropy(entropy_val),
            Text(risk, style="bold red" if risk and risk != "clean" else "dim"),
            ", ".join(f"{k}: {v}" for k, v in a["alignment_summary"].items() if v > 0)
        )

    console.print(table)

def print_logits_insights(logits_analysis: Dict[str, Any] | None):
    if not logits_analysis or "error" in logits_analysis:
        console.print("[dim]Logits analysis: Not available[/dim]")
        return

    table = Table(title="Token-Level Model Confidence (Logits)", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Rating", justify="center")

    def rate_conf(v): return "HIGH" if v >= 0.9 else "MED" if v >= 0.7 else "LOW"
    def color_conf(v): return "green" if v >= 0.9 else "yellow" if v >= 0.7 else "red"

    table.add_row(
        "Avg Token Conf",
        f"{logits_analysis['avg_token_confidence']:.4f}",
        Text(rate_conf(logits_analysis['avg_token_confidence']), style=f"bold {color_conf(logits_analysis['avg_token_confidence'])}")
    )
    table.add_row(
        "Min Token Conf",
        f"{logits_analysis['min_token_confidence']:.4f}",
        Text("LOW", style="bold red") if logits_analysis['min_token_confidence'] < 0.5 else Text("OK", style="yellow")
    )
    table.add_row("Risky Tokens (<10%)", str(logits_analysis['risky_tokens_count']), "red" if logits_analysis['risky_tokens_count'] > 0 else "green")
    table.add_row("High Entropy Tokens", str(logits_analysis['high_entropy_count']), "red" if logits_analysis['high_entropy_count'] > 0 else "green")

    console.print(table)