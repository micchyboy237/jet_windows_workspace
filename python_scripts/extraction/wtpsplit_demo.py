from wtpsplit import SaT
from rich.console import Console
from rich.panel import Panel

console = Console()

text = """\
Understanding Sentence Segmentation Modern NLP models can split text very accurately SaT is one of the best open-source options available in 2025 and 2026 However real-world content from websites PDFs emails and scanned documents often contains many challenges Headings and subheadings appear without clear separation Lists and bullets mix with normal prose Tables captions footnotes and references get concatenated Good sentence and paragraph extractors should try to filter or correctly group most of these noise elements That's why semantic-aware tools are becoming essential "Clean segmentation makes downstream tasks such as summarization entity recognition and question answering much easier" said an NLP researcher in late 2024 We love using wtpsplit combined with SaT models because they handle multilingual text poorly punctuated input and domain shifts surprisingly well Recent benchmarks show that larger SaT variants like sat-12l-sm often outperform rule-based tools and even some LLM-based segmenters on corrupted or informal text Nevertheless no single threshold works perfectly for every document type News articles benefit from moderate settings while forum posts chat logs recipes and poetry may need more aggressive splitting Legal contracts and scientific papers on the other hand usually require conservative boundaries to avoid breaking important clauses or definitions Last updated February 2026 Contact us for custom adaptations
• Bullet one explains the basic idea
• Bullet two adds more details without punctuation
• Bullet three continues the list and sometimes runs longer than expected
Another paragraph starts here without obvious separation It might contain important information mixed with minor details We should group related ideas together but split when topic or intent clearly changes For example this sentence is short. This one is longer and contains a quote: "Threshold tuning is more art than science sometimes" In practice you will need to experiment with your own data to find the sweet spot between too many fragments and overly merged blocks Good luck with your segmentation tasks
"""

sat = SaT("sat-3l-sm", style_or_domain="ud", language="en")

# List of combinations to compare
# Expanded combinations covering low → high extremes
combinations = [
    {"name": "Extreme aggressive (almost no merging)",          "sent_th": 0.00,  "para_th": 0.00},
    {"name": "Very aggressive sentences + para",                "sent_th": 0.05,  "para_th": 0.10},
    {"name": "Aggressive sentences + low para",                 "sent_th": 0.10,  "para_th": 0.30},
    {"name": "More sentences + aggressive para",                "sent_th": 0.20,  "para_th": 0.40},
    {"name": "Balanced aggressive (many breaks)",               "sent_th": None,  "para_th": 0.45},
    {"name": "Recommended start – balanced",                    "sent_th": None,  "para_th": 0.60},
    {"name": "Conservative paragraphs",                         "sent_th": None,  "para_th": 0.75},
    {"name": "Very conservative paragraphs",                    "sent_th": None,  "para_th": 0.90},
    {"name": "Stricter sentences + moderate para",              "sent_th": 0.35,  "para_th": 0.60},
    {"name": "High sentence thr + high para (few breaks)",      "sent_th": 0.50,  "para_th": 0.80},
    {"name": "Very high thr → almost no splits at all",         "sent_th": 0.80,  "para_th": 0.95},
    {"name": "Extreme conservative (almost one block)",         "sent_th": 1.00,  "para_th": 1.00},
]

for combo in combinations:
    sent_label = combo["sent_th"] if combo["sent_th"] is not None else "default (~0.01–0.025)"
    console.print(Panel(
        f"[bold cyan]{combo['name']}[/bold cyan]\n"
        f"sentence threshold = {sent_label}\n"
        f"paragraph_threshold = {combo['para_th']}",
        border_style="bright_blue"
    ))

    try:
        result = sat.split(
            text,
            do_paragraph_segmentation=True,
            threshold=combo["sent_th"],
            paragraph_threshold=combo["para_th"],
        )

        for i, para_sentences in enumerate(result, 1):
            para_text = " ".join(s.strip() for s in para_sentences if s.strip())
            preview_len = 160
            preview = para_text[:preview_len] + ("..." if len(para_text) > preview_len else "")
            console.print(f"  [dim]{i:2d}.[/dim] {preview}")
        console.print("")

    except Exception as e:
        console.print(f"[red]Error with these thresholds: {e}[/red]\n")
