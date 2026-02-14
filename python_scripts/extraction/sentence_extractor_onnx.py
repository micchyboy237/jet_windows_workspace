from pathlib import Path
from typing import Literal

from rich.console import Console
import onnxruntime as ort
from wtpsplit import SaT

console = Console(highlight=False)


def load_sat_model(
    model_dir_or_name: str | Path,
    *,
    use_gpu: bool = False,
    batch_size: int = 32,
    stride: int = 64,
    block_size: int = 512,
    verbose: bool = False,
) -> SaT:
    """Load a SaT (Segment Any Text) model — preferably the ONNX version.

    This function provides a clean interface with GPU control via a single boolean.

    Args:
        model_dir_or_name: Local folder path or HF model id (e.g. "sat-3l-sm")
        use_gpu: If True, tries to use CUDA if available (falls back to CPU)
        batch_size: Inference batch size (affects speed & memory)
        stride: Stride used during sliding window inference
        block_size: Maximum block size for processing
        verbose: Show detailed loading messages

    Returns:
        SaT instance ready to use (.split(), .predict_proba())

    Raises:
        ValueError: If ONNX model is requested but not found
        ImportError: If onnxruntime-gpu is needed but not installed
    """
    model_path = Path(model_dir_or_name)

    # ── Decide providers ────────────────────────────────────────────────────────
    if use_gpu:
        preferred_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        console.print("[bold green]GPU mode requested → trying CUDA...[/bold green]")
    else:
        preferred_providers = ["CPUExecutionProvider"]
        console.print("[dim]Using CPU only (use_gpu=False)[/dim]")

    # ── Check what is actually available ────────────────────────────────────────
    available = ort.get_available_providers()
    if verbose:
        console.print(f"Available ONNX providers: [cyan]{', '.join(available)}[/cyan]")

    # Filter only providers we actually have
    providers = [p for p in preferred_providers if p in available]

    if not providers:
        console.print("[yellow]Warning: No requested providers available. Falling back to CPU.[/yellow]")
        providers = ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in providers:
        console.print("[bold green]→ ONNX Runtime will use GPU (CUDA)[/bold green]")
    else:
        console.print("[dim]→ Running on CPU[/dim]")

    # ── Load model ──────────────────────────────────────────────────────────────
    try:
        model = SaT(
            model_name_or_model=str(model_path) if model_path.is_dir() else model_dir_or_name,
            ort_providers=providers,
            # You can add ort_kwargs={...} here if needed (e.g. graph optimization)
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to load model[/bold red]  {exc.__class__.__name__}: {exc}")
        raise

    # Quick summary
    if verbose:
        console.print(f"[dim]Model loaded • batch_size={batch_size} • stride={stride} • block={block_size}[/dim]")

    return model


if __name__ == "__main__":
    text = "Your long multi-sentence text here..."

    # 1. Use GPU if available
    model_gpu = load_sat_model(
        r"C:\Users\druiv\.cache\huggingface\hub\models--segment-any-text--sat-3l-sm\snapshots\137da054051ad9f1eac42025f758db4ac9f22535",
        use_gpu=True,
        verbose=True,
    )
    sentences = model_gpu.split(text, threshold=0.25)
    print(sentences)

    # 2. Use CPU (default behavior)
    model_cpu = load_sat_model("sat-3l-sm")   # downloads if needed
    sentences = model_cpu.split(text, threshold=0.25)
    print(sentences)
