from pathlib import Path


def linkify(path: str | Path):
    path = Path(path)
    # Provide clickable file link with basename (for rich/terminal that support it)
    return f"[bold blue][link=file://{path}]{path.name}[/link][/bold blue]"
