"""
llama_log_utils.py
Helpers for saving llama_cpp_wrapper call logs to disk.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# One shared rich console for the process
console = Console(highlight=False)

# Thread-local counter so concurrent calls get unique folder names
_call_counter_lock = threading.Lock()
_call_counter = 0


def _next_call_index() -> int:
    global _call_counter
    with _call_counter_lock:
        _call_counter += 1
        return _call_counter


def make_call_dir(logs_dir: Path) -> Path:
    """Create and return a unique timestamped subfolder for one LLM call."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = _next_call_index()
    folder = logs_dir / f"{ts}_{idx:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_file_logger(call_dir: Path) -> logging.Logger:
    """Return a Logger that writes plain text to call_dir/call.log."""
    log_path = call_dir / "call.log"
    logger = logging.getLogger(str(call_dir))  # unique name per call
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


def save_json(path: Path, obj: Any) -> None:
    """Write obj as indented JSON to path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_markdown(path: Path, content: str, reasoning: Optional[str] = None) -> None:
    """Write assistant reply (and optional reasoning) as a Markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        if reasoning:
            f.write("## Reasoning\n\n")
            f.write(reasoning.strip())
            f.write("\n\n---\n\n")
        f.write("## Response\n\n")
        f.write(content.strip())
        f.write("\n")


def print_reasoning_chunk(chunk: str) -> None:
    """Print a reasoning/thinking chunk in a dim cyan style, no newline flush."""
    console.print(Text(chunk, style="dim cyan"), end="", soft_wrap=True)


def print_content_chunk(chunk: str) -> None:
    """Print a regular content chunk in bright white, no newline flush."""
    console.print(Text(chunk, style="bright_white"), end="", soft_wrap=True)


def print_request_panel(messages: list, params: Dict[str, Any]) -> None:
    """Print a summary panel before sending the request."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "(no user message)",
    )
    if isinstance(last_user, list):  # multipart content
        last_user = " ".join(
            p.get("text", "") for p in last_user if p.get("type") == "text"
        )
    param_lines = "\n".join(
        f"  [dim]{k}[/dim]: {v}"
        for k, v in params.items()
        if v is not None and k not in ("messages",)
    )
    body = f"[bold]Last user message:[/bold] {last_user[:200]}\n\n[bold]Params:[/bold]\n{param_lines}"
    console.print(Panel(body, title="[bold blue]⟶ LLM Request[/bold blue]", expand=False))


def print_response_panel(content: str, reasoning: Optional[str], usage: Optional[Dict]) -> None:
    """Print a summary panel after a complete (non-stream) response."""
    body = ""
    if reasoning:
        body += f"[dim cyan][Reasoning][/dim cyan]\n{reasoning[:400]}\n\n"
    body += f"[bright_white]{content[:600]}[/bright_white]"
    if usage:
        body += f"\n\n[dim]Tokens — prompt: {usage.get('prompt_tokens')}  completion: {usage.get('completion_tokens')}  total: {usage.get('total_tokens')}[/dim]"
    console.print(Panel(body, title="[bold green]⟵ LLM Response[/bold green]", expand=False))


def print_stream_end_panel(content: str, reasoning: Optional[str], elapsed: float) -> None:
    """Print a closing panel after streaming finishes."""
    body = f"[dim]elapsed: {elapsed:.2f}s[/dim]"
    if reasoning:
        body = f"[dim cyan]Reasoning chars: {len(reasoning)}[/dim cyan]  " + body
    body = f"[bright_white]Content chars: {len(content)}[/bright_white]  " + body
    console.print()  # newline after streamed chunks
    console.print(Panel(body, title="[bold green]⟵ Stream complete[/bold green]", expand=False))