from translate_jp_en_llm2 import translate_japanese_to_english
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.rule import Rule

console = Console()

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]llama-server KV Cache Demo[/bold cyan]\n[dim]Progressive / Growing Subtitles[/dim]",
        border_style="cyan"
    ))

    history = []
    progressive_subtitles = [
        "今日は",  #  "Today," or "It's today." (incomplete)
        "今日はとても",  #  "Today is very" (still incomplete)
        "今日はとても疲れた。",  #  "I'm so tired today."
        "あなたのことが好きだよ。",  #  "I like you."
        "早く行かないと電車に乗り遅れる！",  #  "Hurry up or we'll miss the train!"
    ]

    console.print(Rule("[bold]Progressive Growing Input Test[/bold]"))
    console.print()

    results_en = []
    table_rows = []

    for i, growing_text in enumerate(progressive_subtitles, start=1):
        console.print(f"[bold yellow]\[Step {i}][/bold yellow] [white]{growing_text}[/white]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]Translating...[/dim]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            result = translate_japanese_to_english(growing_text, history=history)

        en_text = result["text"]
        results_en.append(en_text)

        console.print(f"  [green]↳ EN:[/green] [italic]{en_text}[/italic]")
        console.print()

        table_rows.append((i, growing_text, en_text))

        history.append({"role": "user", "content": growing_text})
        history.append({"role": "assistant", "content": en_text})

    # Final results table
    console.print(Rule("[bold]Final Results[/bold]"))
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Japanese Input", style="cyan", ratio=2)
    table.add_column("English Translation", style="green", ratio=3)

    for step, jp, en in table_rows:
        table.add_row(str(step), jp, en)

    console.print(table)
    console.print()
    console.print(Panel(
        "[dim]Progressive test completed.\n"
        "Watch how [cyan]cached[/cyan] tokens increase as history grows.\n"
        "Only the latest growing user message needs re-evaluation.[/dim]",
        border_style="dim"
    ))