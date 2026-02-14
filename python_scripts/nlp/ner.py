import hashlib
import json
from typing import Any, TypedDict

import spacy
import os
from rich.console import Console
from rich.logging import RichHandler
import logging

# ────────────────────────────────────────────────
#          Rich + structured logging setup
# ────────────────────────────────────────────────

console = Console(highlight=False)  # we'll control markup manually when needed

# Nice rich handler — timestamp, level icon, no ugly square brackets
rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    tracebacks_suppress=[spacy],
    tracebacks_show_locals=True,
    markup=True,
    show_time=True,
    show_path=False,          # cleaner output
    omit_repeated_times=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",     # rich handler already adds nice prefix
    datefmt="[%X]",
    handlers=[rich_handler]
)

logger = logging.getLogger("gliner.entity-extractor")

# Optional: make spacy quieter (it loves to spam INFO messages)
logging.getLogger("spacy").setLevel(logging.WARNING)

# ────────────────────────────────────────────────

Entity = dict[str, Any]

# Global cache
nlp_cache: spacy.Language | None = None
nlp_cache_hash: str | None = None

DEFAULT_GLNER_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
DEFAULT_CHUNK_SIZE = 2048


class Entity(TypedDict):
    text: str
    label: str
    score: float


def compute_config_hash(config: dict) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def load_nlp_pipeline(
    labels: list[str],
    style: str = "ent",
    model: str = DEFAULT_GLNER_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> spacy.Language:
    global nlp_cache, nlp_cache_hash

    config = {
        "gliner_model": model,
        "chunk_size": chunk_size,
        "labels": labels,
        "style": style,
    }

    new_hash = compute_config_hash(config)

    if nlp_cache is None:
        logger.info("[bold cyan]Creating GLiNER + spaCy pipeline cache...[/]")
        logger.info(f"  model   : [blue]{model}[/]")
        logger.info(f"  labels  : [green]{len(labels)}[/] classes")
        logger.info(f"  chunk   : [magenta]{chunk_size:,}[/] tokens")
        logger.info(f"  style   : [yellow]{style}[/]")

    elif nlp_cache_hash != new_hash:
        logger.warning("[bold yellow]Configuration changed → recreating pipeline[/]")
        logger.info(f"  new hash: [dim]{new_hash[:12]}…[/]")
        logger.info(f"  labels  : [green]{len(labels)}[/] classes")

    else:
        logger.debug("[dim]Reusing existing GLiNER pipeline cache[/]")
        return nlp_cache  # type: ignore

    # ─── Actual creation ────────────────────────────────────────
    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=config)

    # Update cache
    nlp_cache = nlp
    nlp_cache_hash = new_hash

    logger.info("[bold green]Pipeline ready ✓[/]")

    return nlp


def merge_dot_prefixed_words(text: str) -> str:
    """Merge tokens like '. NET' → '.NET' and 'C .' → 'C.' etc."""
    tokens = text.split()
    merged = []
    for token in tokens:
        if token.startswith(".") and merged and not merged[-1].startswith("."):
            merged[-1] += token
        elif merged and merged[-1].endswith("."):
            merged[-1] += token
        else:
            merged.append(token)
    return " ".join(merged)


def get_unique_entities(entities: list[Entity]) -> list[Entity]:
    """Keep only the highest-confidence version of each entity (label + normalized text)"""
    best: dict[str, Entity] = {}

    for ent in entities:
        # Very aggressive normalization for deduplication
        words = [t.replace(" ", "") for t in ent["text"].split() if t]
        norm_text = " ".join(words)

        key = f"{ent['label']}-{norm_text}"
        score = float(ent["score"])

        if key not in best or score > float(best[key]["score"]):
            best[key] = {
                "text": norm_text,
                "label": ent["label"],
                "score": score,
            }

    return list(best.values())


def extract_entities_from_text(nlp: spacy.Language, text: str) -> Entity:
    """Main entry point — extract → clean → group by label"""
    doc = nlp(text)

    raw_entities = [
        {
            "text": merge_dot_prefixed_words(ent.text),
            "label": ent.label_,
            "score": float(ent._.score),
        }
        for ent in doc.ents
    ]

    unique_entities = get_unique_entities(raw_entities)

    result: Entity = {}
    for ent in unique_entities:
        label_key = ent["label"].lower().replace(" ", "_")
        result.setdefault(label_key, [])
        if ent["text"] not in result[label_key]:
            result[label_key].append(ent["text"])

    # Optional: nice debug output when in DEBUG mode
    if logger.isEnabledFor(logging.DEBUG) and unique_entities:
        logger.debug(f"Extracted [bold]{len(unique_entities)}[/] unique entities")

    return result