from typing import List, Dict, Any
from transformers import pipeline
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
import logging
from tqdm import tqdm

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)

console = Console()


class JapaneseNERExtractor:
    """Reusable extractor for Japanese Named Entity Recognition using knosing/japanese_ner_model."""

    def __init__(self) -> None:
        """Initialize the NER pipeline with the specific Japanese model."""
        logger.info("Loading Japanese NER model (this may take a moment on first run)...")
        self.ner_pipeline = pipeline(
            "ner",
            model="knosing/japanese_ner_model",
            tokenizer="tohoku-nlp/bert-base-japanese-v3",
            aggregation_strategy="simple",  # Merges subwords and provides entity_group
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        )
        logger.info("Model loaded successfully.")

    def extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract entities from one or multiple Japanese texts.

        Args:
            texts: List of Japanese strings (supports single text as list of one).

        Returns:
            List of results, each containing list of entity dicts with keys:
            'word', 'entity_group', 'score', 'start', 'end'
        """
        if not texts:
            return []

        logger.info(f"Processing {len(texts)} text(s)...")
        results = []
        for text in tqdm(texts, desc="NER Processing", unit="text"):
            entities = self.ner_pipeline(text)
            results.append(entities)

        logger.info("Extraction completed.")
        return results

    def print_results(self, texts: List[str], results: List[List[Dict[str, Any]]]) -> None:
        """Pretty-print results using rich tables."""
        for text, entities in zip(texts, results):
            console.rule("Input Text")
            console.print(text)

            if not entities:
                console.print("[yellow]No entities found.[/yellow]")
                continue

            table = Table(title="Extracted Entities", show_header=True, header_style="bold magenta")
            table.add_column("Word", style="cyan")
            table.add_column("Entity Type", style="green")
            table.add_column("Score", justify="right", style="yellow")

            for entity in entities:
                table.add_row(
                    entity["word"],
                    entity["entity_group"],
                    f"{entity['score']:.4f}"
                )

            console.print(table)
            console.print()


# Usage examples
if __name__ == "__main__":
    import torch  # For device check

    extractor = JapaneseNERExtractor()

    example_texts = [
        "山田太郎は2025年1月に東京都渋谷区で株式会社xAIジャパンと面談を行い、イーロン・マスクと握手した。",
        "東京タワーは日本を代表する施設です。",
    ]

    results = extractor.extract_entities(example_texts)
    extractor.print_results(example_texts, results)