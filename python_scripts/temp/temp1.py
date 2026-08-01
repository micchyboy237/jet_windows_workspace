from rich.console import Console
from rich.panel import Panel
from pydantic import BaseModel
from spacy.tokens import SpanGroup
from dataclasses import dataclass
from spacy.tokens import Doc, Span
import spacy
from spacy import displacy
from span_marker import SpanMarkerModel
from typing import Dict, List, Optional
import os
import json
from pathlib import Path

console = Console()

# SpanMarkerWord with label


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float
    label: str

    def __str__(self) -> str:
        return self.text


@dataclass
class DocSentence:
    text: str
    start_char: int
    end_char: int
    token_count: int


@dataclass
class DocEntity:
    text: str
    lemma: str  # Added lemma field
    label: str
    start_char: int
    end_char: int
    score: float
    vector_norm: float | None


@dataclass
class DocNounChunk:
    text: str
    root_text: str
    root_dep: str
    root_head_text: str


@dataclass
class DocSettings:
    lang: str
    direction: str


def process_text(text: str, nlp: spacy.language.Language, model: SpanMarkerModel) -> tuple[Doc, List[SpanMarkerWord]]:
    """Process text with spaCy pipeline and SpanMarker model, returning SpanMarkerWord predictions."""
    doc = nlp(text)
    predictions = model.predict(text)
    processed_predictions = [
        SpanMarkerWord(
            text=pred["span"],
            lemma=nlp(pred["span"])[0].lemma_ if pred["span"] else "",
            start_idx=pred["char_start_index"],
            end_idx=pred["char_end_index"],
            score=pred["score"],
            label=pred["label"]
        )
        for pred in predictions
    ]
    return doc, processed_predictions


def log_entities(predictions: List[SpanMarkerWord]) -> None:
    """Log named entities with relevant details."""
    console.print()
    console.print(f"[bold blue]Extracted Entities ({len(predictions)}):[/bold blue]")
    for entity in predictions:
        console.print()
        console.print(f"Text: [cyan]{entity.text}[/cyan]")
        console.print(f"Lemma: [cyan]{entity.lemma}[/cyan]")
        console.print(f"Label: [cyan]{entity.label}[/cyan]")
        console.print(f"Start: [green]{entity.start_idx}[/green]")
        console.print(f"End: [green]{entity.end_idx}[/green]")
        console.print(f"Score: [green]{entity.score:.4f}[/green]")
        console.print("[dim]---[/dim]")


def log_noun_chunks(doc: Doc) -> None:
    """Log noun chunks with relevant details."""
    console.print()
    console.print(f"[bold blue]Extracted Noun Chunks ({len(list(doc.noun_chunks))}):[/bold blue]")
    for chunk in doc.noun_chunks:
        console.print()
        console.print(f"Text: [cyan]{chunk.text}[/cyan]")
        console.print(f"Root Text: [cyan]{chunk.root.text}[/cyan]")
        console.print(f"Root Dependency: [green]{chunk.root.dep_}[/green]")
        console.print(f"Root Head Text: [green]{chunk.root.head.text}[/green]")
        console.print("[dim]---[/dim]")


def log_sentences(doc: Doc) -> None:
    """Log sentences with relevant details."""
    console.print()
    console.print(f"[bold blue]Extracted Sentences ({len(list(doc.sents))}):[/bold blue]")
    for i, sent in enumerate(doc.sents, 1):
        console.print()
        console.print(f"Sentence {i}: [cyan]{sent.text}[/cyan]")
        console.print(f"Start Char: [green]{sent.start_char}[/green]")
        console.print(f"End Char: [green]{sent.end_char}[/green]")
        console.print(f"Token Count: [green]{len(sent)}[/green]")
        console.print("[dim]---[/dim]")


def parse_entities(doc: Doc, predictions: List[SpanMarkerWord]) -> List[DocEntity]:
    """Parse SpanMarkerWord predictions into a list of DocEntity objects."""
    return [
        DocEntity(
            text=entity.text,
            lemma=entity.lemma,  # Include lemma
            label=entity.label,
            start_char=entity.start_idx,
            end_char=entity.end_idx,
            score=entity.score,
            vector_norm=(
                float(doc[entity.start_idx:entity.end_idx].vector_norm)  # Convert to Python float
                if doc[entity.start_idx:entity.end_idx].has_vector
                else None
            )
        )
        for entity in predictions
    ]


def parse_dependencies(doc: Doc) -> List[DocNounChunk]:
    """Parse a spaCy Doc into a list of DocNounChunk objects containing noun chunk details."""
    return [
        DocNounChunk(
            text=chunk.text,
            root_text=chunk.root.text,
            root_dep=chunk.root.dep_,
            root_head_text=chunk.root.head.text
        )
        for chunk in doc.noun_chunks
    ]


def parse_sentences(doc: Doc) -> List[DocSentence]:
    """Parse a spaCy Doc into a list of DocSentence objects containing sentence details."""
    return [
        DocSentence(
            text=sent.text,
            start_char=sent.start_char,
            end_char=sent.end_char,
            token_count=len(sent)
        )
        for sent in doc.sents
    ]


def parse_settings(doc: Doc) -> DocSettings:
    """Parse a spaCy Doc's settings into a DocSettings object."""
    return DocSettings(
        lang=doc.lang_,
        direction=doc.vocab.writing_system.get("direction", "ltr")
    )


def char_to_token_index(doc: Doc, char_start: int, char_end: int) -> tuple[Optional[int], Optional[int]]:
    """Convert character indices to token indices in a spaCy Doc."""
    start_token = None
    end_token = None
    for token in doc:
        if token.idx <= char_start < token.idx + len(token.text):
            start_token = token.i
        if token.idx < char_end <= token.idx + len(token.text):
            end_token = token.i + 1
            break
    return start_token, end_token


def create_span_group(doc: Doc, predictions: List[SpanMarkerWord]) -> SpanGroup:
    """Create a SpanGroup from SpanMarkerWord predictions for visualization."""
    spans = []
    for entity in predictions:
        start_token, end_token = char_to_token_index(
            doc, entity.start_idx, entity.end_idx)
        if start_token is not None and end_token is not None and start_token < len(doc) and end_token <= len(doc):
            try:
                span = Span(
                    doc,
                    start_token,
                    end_token,
                    label=entity.label,
                    kb_id=f"score:{entity.score:.4f}"
                )
                spans.append(span)
            except IndexError as e:
                console.print(f"[red]Error creating span for entity '[bold]{entity.text}[/bold]' "
                             f"(char {entity.start_idx}:{entity.end_idx}): {e}[/red]")
        else:
            console.print(f"[yellow]Skipping entity '[bold]{entity.text}[/bold]' due to invalid token indices "
                           f"(char {entity.start_idx}:{entity.end_idx})[/yellow]")
    return SpanGroup(doc, name="entities", spans=spans)


def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
    model = SpanMarkerModel.from_pretrained(
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super").to("cpu")

    # Input text
    text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the 
    Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her 
    death in 30 BCE."""

    # Process text
    doc, predictions = process_text(text, nlp, model)

    # Log entities, noun chunks, and sentences
    log_entities(predictions)
    log_noun_chunks(doc)
    log_sentences(doc)

    # Create span group for visualization
    doc.spans["entities"] = create_span_group(doc, predictions)

    # Parse and save data
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save entities
    entities_path = os.path.join(output_dir, "entities.json")
    with open(entities_path, "w") as f:
        json.dump([e.__dict__ for e in parse_entities(doc, predictions)], f, indent=2)
    console.print(f"[green]✓[/green] Saved entities to [link=file://{entities_path}]{Path(entities_path).name}[/link]")
    
    # Save dependencies
    deps_path = os.path.join(output_dir, "dependencies.json")
    with open(deps_path, "w") as f:
        json.dump([d.__dict__ for d in parse_dependencies(doc)], f, indent=2)
    console.print(f"[green]✓[/green] Saved dependencies to [link=file://{deps_path}]{Path(deps_path).name}[/link]")
    
    # Save sentences
    sentences_path = os.path.join(output_dir, "sentences.json")
    with open(sentences_path, "w") as f:
        json.dump([s.__dict__ for s in parse_sentences(doc)], f, indent=2)
    console.print(f"[green]✓[/green] Saved sentences to [link=file://{sentences_path}]{Path(sentences_path).name}[/link]")
    
    # Save settings
    settings_path = os.path.join(output_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(parse_settings(doc).__dict__, f, indent=2)
    console.print(f"[green]✓[/green] Saved settings to [link=file://{settings_path}]{Path(settings_path).name}[/link]")
    
    # Save spans
    spans_path = os.path.join(output_dir, "spans.json")
    with open(spans_path, "w") as f:
        json.dump(displacy.parse_spans(doc, options={"spans_key": "entities"}), f, indent=2)
    console.print(f"[green]✓[/green] Saved spans to [link=file://{spans_path}]{Path(spans_path).name}[/link]")

    # Visualize spans
    options = {
        "spans_key": "entities",
        "colors": {
            "person-other": "#ff9999",
            "location-GPE": "#99ff99",
            "date": "#9999ff",
            "product-airplane": "#ffcc99",
            "location-bodiesofwater": "#99ccff",
            "event-attack/battle/war/militaryconflict": "#cc99ff"
        }
    }
    displacy.render(doc, style="span", options=options, jupyter=False)


if __name__ == "__main__":
    main()
