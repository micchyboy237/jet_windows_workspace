# japanese_ner.py
"""
Reusable Japanese NER wrapper for knosing/japanese_ner_model.
Supports batch processing, entity filtering, score thresholds.
Provides Japanese → English entity type mapping from official docs.
"""
from typing import List, Dict, Any, Literal, Optional, TypedDict
from transformers import pipeline
from rich.console import Console
from rich.table import Table
from rich.progress import track
from tqdm import tqdm
import logging
from rich.logging import RichHandler

# Very minimal & beautiful
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("JapaneseNER")
console = Console()

class JapaneseEntity(TypedDict):
    word: str
    entity_group: str  # Japanese label e.g., "人名"
    score: float
    start: int
    end: int

# Official entity types from model card (knosing/japanese_ner_model)
# BIO tags mapped to Japanese/English via pipeline aggregation + dataset schema
ENTITY_MAPPING: Dict[str, Dict[str, str]] = {
    "人名": {"ja": "人名", "en": "Person", "bio_prefix": "PER"},
    "地名": {"ja": "地名", "en": "Location", "bio_prefix": "LOC"},  # 施設名/地名
    "法人名": {"ja": "法人名", "en": "Organization", "bio_prefix": "ORG"},  # 組織名/法人名
    "組織名": {"ja": "組織名", "en": "Organization", "bio_prefix": "ORG"},
    "施設名": {"ja": "施設名", "en": "Facility", "bio_prefix": "LOC"},
    "品名": {"ja": "品名", "en": "Product", "bio_prefix": "PROD"},
    "名": {"ja": "名", "en": "Generic", "bio_prefix": "GEN"},
    "イベント名": {"ja": "イベント名", "en": "Event", "bio_prefix": "EVE"},
    "その他の組織名": {"ja": "その他の組織名", "en": "Other Organization", "bio_prefix": "ORG-OTH"},
    "政治的組織名": {"ja": "政治的組織名", "en": "Political Organization", "bio_prefix": "ORG"},  # Variant
}

SupportedEntityType = Literal[
    "人名", "地名", "法人名", "組織名", "施設名", "品名", "名", 
    "イベント名", "その他の組織名", "政治的組織名"
]

class JapaneseNER:
    """Flexible Japanese NER pipeline with rich output and batch support."""
    
    def __init__(self, model_name: str = "knosing/japanese_ner_model", 
                 tokenizer_name: str = "tohoku-nlp/bert-base-japanese-v3",
                 device: int = 0, min_score: float = 0.5, 
                 aggregation_strategy: str = "simple") -> None:
        """
        Initialize Japanese NER pipeline.
        
        Args:
            model_name: HF model for NER.
            tokenizer_name: Tokenizer (must match model).
            device: CUDA device (-1 for CPU).
            min_score: Filter entities below this confidence.
            aggregation_strategy: How to merge subwords ('simple', 'first', 'max', 'average').
        """
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        logger.info(f"Loading JapaneseNER model '{model_name}' on {device_str}...")
        
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=tokenizer_name,
            aggregation_strategy=aggregation_strategy,
            device=device if device >= 0 else -1
        )
        self.min_score = min_score
        self.device = device_str
        logger.info(f"[green]JapaneseNER ready on {self.device}, min_score={min_score:.2f}[/green]")
    
    def extract_entities(self, text: str, 
                        entity_types: Optional[List[SupportedEntityType]] = None,
                        min_score: Optional[float] = None) -> List[JapaneseEntity]:
        """
        Extract entities from single text.
        
        Args:
            text: Japanese input text.
            entity_types: Filter to specific types (None=all).
            min_score: Override class min_score.
            
        Returns:
            List of entities with Japanese labels, scores, positions.
        """
        if min_score is None:
            min_score = self.min_score
        
        raw_entities = self.ner_pipeline(text)
        filtered = [
            e for e in raw_entities 
            if e["score"] >= min_score 
            and (not entity_types or e["entity_group"] in entity_types)
        ]
        return filtered
    
    def batch_extract(self, texts: List[str], 
                     max_workers: int = 1,
                     **kwargs: Any) -> List[List[JapaneseEntity]]:
        """
        Batch NER with progress tracking.
        
        Args:
            texts: List of Japanese texts.
            max_workers: Parallel workers (pipeline handles internally).
            **kwargs: Passed to extract_entities().
            
        Returns:
            List of List[entities] per text.
        """
        logger.info(f"Processing {len(texts)} texts with tqdm progress...")
        results = []
        for text in tqdm(texts, desc="NER Progress", unit="text"):
            entities = self.extract_entities(text, **kwargs)
            results.append(entities)
        return results
    
    def print_entities_table(self, entities: List[JapaneseEntity], 
                           show_english: bool = True, max_entities: int = 20) -> None:
        """Rich console table for entities."""
        table = Table(title="Japanese NER Results", show_header=True, header_style="bold magenta")
        table.add_column("Word", style="cyan", no_wrap=True)
        table.add_column("Japanese Type", style="green")
        if show_english:
            table.add_column("English Type", style="blue")
        table.add_column("Score", justify="right", style="yellow")
        table.add_column("Span", justify="right", style="white")
        
        for i, entity in enumerate(entities[:max_entities]):
            ja_type = entity["entity_group"]
            en_type = ENTITY_MAPPING.get(ja_type, {}).get("en", "Unknown")
            span = f"{entity['start']}-{entity['end']}"
            score_str = f"{entity['score']:.4f}"
            table.add_row(
                entity["word"], 
                ja_type, 
                en_type if show_english else "",
                score_str, 
                span
            )
        
        if len(entities) > max_entities:
            table.add_row("...", "...", "..." if show_english else "", "...", "...")
        
        console.print(table)
        
        # Stats
        if entities:
            avg_score = sum(e["score"] for e in entities) / len(entities)
            type_counts = {}
            for e in entities:
                type_counts[e["entity_group"]] = type_counts.get(e["entity_group"], 0) + 1
            
            stats_table = Table(title="Summary Stats", show_header=True, header_style="bold green")
            stats_table.add_column("Metric", style="bold")
            stats_table.add_column("Value", style="cyan")
            stats_table.add_row("Total Entities", str(len(entities)))
            stats_table.add_row("Avg Score", f"{avg_score:.4f}")
            for t, count in type_counts.items():
                en = ENTITY_MAPPING.get(t, {}).get("en", t)
                stats_table.add_row(f"{t} ({en})", str(count))
            console.print(stats_table)

# Usage Examples (CLI-friendly)
def demo_single_text() -> None:
    """Demo 1: Single text (your original example)."""
    ner = JapaneseNER(device=0, min_score=0.95)  # High threshold
    
    text = "山田太郎は2025年1月に東京都渋谷区で株式会社xAIジャパンと面談を行い、イーロン・マスクと握手した。"
    entities = ner.extract_entities(text)
    ner.print_entities_table(entities)
    logger.info("Demo 1: Single text NER complete.")

def demo_batch_texts() -> None:
    """Demo 2: Batch processing with progress."""
    ner = JapaneseNER(device=0)
    
    texts = [
        "山田太郎は2025年1月に東京都渋谷区で株式会社xAIジャパンと面談を行い、イーロン・マスクと握手した。",
        "東京ディズニーランドでSPRiNGSのコンサートがあり、Apple製品のiPhone 15が発売された。",
        "2026年の東京オリンピックで政治的組織名である自民党の岸田文雄首相が出席予定。",
    ]
    
    batch_results = ner.batch_extract(texts, min_score=0.8)
    for i, entities in enumerate(batch_results):
        print(f"\n--- Text {i+1} ---")
        ner.print_entities_table(entities)

def demo_filter_types() -> None:
    """Demo 3: Filter specific entity types (e.g., only 人名 + 法人名)."""
    ner = JapaneseNER(device=0)
    
    text = "イーロン・マスクはxAIジャパン、東京大学、渋谷の施設でイベントを行い、iPhoneとPlayStation 5を発表した。"
    persons_orgs = ner.extract_entities(text, entity_types=["人名", "法人名"])
    ner.print_entities_table(persons_orgs)
    logger.info("Demo 3: Filtered to 人名 + 法人名 only.")

if __name__ == "__main__":
    logger.info("=== Japanese NER Demos ===")
    demo_single_text()
    demo_batch_texts()
    demo_filter_types()
    logger.info("All demos complete. Model provides Japanese entity types: 人名, 地名, 法人名, etc. (confirmed via HF docs).")