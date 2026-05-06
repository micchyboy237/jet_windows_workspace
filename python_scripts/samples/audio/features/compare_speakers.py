import argparse

from pyannote.audio import Inference, Model
from rich.console import Console
from scipy.spatial.distance import cdist


def main():
    parser = argparse.ArgumentParser(
        description="Compare two speaker embeddings (wav files) using cosine distance."
    )
    parser.add_argument("speaker1", type=str, help="Path to first speaker WAV file")
    parser.add_argument("speaker2", type=str, help="Path to second speaker WAV file")
    # parser.add_argument(
    #     "--access_token",
    #     type=str,
    #     default=None,
    #     help="HuggingFace access token (if required)",
    # )
    args = parser.parse_args()

    console = Console()

    # access_token = args.access_token or "ACCESS_TOKEN_GOES_HERE"

    model = Model.from_pretrained("pyannote/embedding")

    inference = Inference(model, window="whole")

    with console.status("[bold green]Computing embeddings..."):
        embedding1 = inference(args.speaker1)
        embedding2 = inference(args.speaker2)

    # `embeddingX` is (1 x D) numpy array extracted from the file as a whole.
    distance = float(cdist(embedding1, embedding2, metric="cosine")[0, 0])
    similarity = 1.0 - distance

    console.print(f"\n[bold blue]Speaker 1:[/] {args.speaker1}")
    console.print(f"[bold yellow]Speaker 2:[/] {args.speaker2}")
    console.print(f"[bold magenta]Cosine distance:[/] [white]{distance:.4f}[/white]")
    console.print(
        f"[bold green]Cosine similarity:[/] [white]{similarity:.4f}[/white]\n"
    )


if __name__ == "__main__":
    main()
