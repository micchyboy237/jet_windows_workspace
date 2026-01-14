# coding: utf-8
"""
Japanese Semantic Search (CUDA, Local Cache Only)
================================================

Real-world use case:
- Semantic search over Japanese text
- FAQ matching, subtitle search, transcript lookup, RAG pre-filter

Constraints:
- CUDA only
- local_files_only=True (NO downloads)
- Reusable, modular, testable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# ============================
# Configuration
# ============================

@dataclass(frozen=True)
class BertConfig:
    model_name: str
    device: str = "cuda"
    max_length: int = 512
    local_files_only: bool = True


# ============================
# Embedding Model
# ============================

class JapaneseBertEmbedder:
    """
    Encapsulates tokenizer + model logic.
    CUDA-only, local cache only.
    """

    def __init__(self, config: BertConfig) -> None:
        if config.device != "cuda":
            raise ValueError("This embedder is CUDA-only")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )

        self.model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )

        self.model.to(config.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> Tensor:
        """
        Convert texts to embeddings using masked mean pooling.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        outputs = self.model(**inputs).last_hidden_state
        return self._mean_pooling(outputs, inputs["attention_mask"])

    @staticmethod
    def _mean_pooling(token_embeddings: Tensor, mask: Tensor) -> Tensor:
        """
        Mean pooling with attention mask.
        """
        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


# ============================
# Similarity Utilities
# ============================

def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute cosine similarity matrix.
    """
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return a_norm @ b_norm.T


# ============================
# Semantic Index
# ============================

class SemanticSearchIndex:
    """
    Simple in-memory semantic index.
    """

    def __init__(self, embedder: JapaneseBertEmbedder) -> None:
        self.embedder = embedder
        self._texts: List[str] = []
        self._embeddings: Tensor | None = None

    def add(self, texts: Iterable[str]) -> None:
        """
        Add documents to the index.
        """
        texts = list(texts)
        if not texts:
            return

        embeddings = self.embedder.encode(texts)

        self._texts.extend(texts)
        self._embeddings = (
            embeddings
            if self._embeddings is None
            else torch.cat((self._embeddings, embeddings), dim=0)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k most similar documents.
        """
        if self._embeddings is None:
            raise RuntimeError("Index is empty")

        query_embedding = self.embedder.encode([query])
        scores = cosine_similarity(query_embedding, self._embeddings)[0]

        k = min(top_k, scores.size(0))
        indices = torch.topk(scores, k=k).indices

        return [(self._texts[i], float(scores[i])) for i in indices]


# ============================
# Example Usage
# ============================

def example() -> None:
    """
    Example semantic search workflow.
    """

    config = BertConfig(
        model_name="tohoku-nlp/bert-base-japanese-v3",
        device="cuda",
        local_files_only=True,
    )

    embedder = JapaneseBertEmbedder(config)
    index = SemanticSearchIndex(embedder)

    documents = [
        "今日は良い天気ですね。",
        "明日の天気予報を教えてください。",
        "機械学習はとても面白い分野です。",
        "深層学習は機械学習の一分野です。",
        "ラーメンが好きです。",
    ]

    index.add(documents)

    query = "AIと機械学習について"
    results = index.search(query, top_k=3)

    for text, score in results:
        print(f"{score:.4f} | {text}")


if __name__ == "__main__":
    example()
