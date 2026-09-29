"""
Demo: OpenSearch Neural Sparse Encoding
FIXED for sentence-transformers==6.0.1 + transformers==5.17.0
"""
import torch
from sentence_transformers.sparse_encoder import SparseEncoder


def main():
    model = SparseEncoder(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        trust_remote_code=True,
        model_kwargs={
            "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"
        },
        # FIX for v6.0.1: Explicitly control tokenizer behavior to prevent
        # garbage position_ids from being passed to custom RoPE embeddings
        processor_kwargs={
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
        },
        device="cpu",  # Keep CPU until position_ids issue is resolved
    )

    doc_text = "Currently New York is rainy."
    # FIX: Pass processing_kwargs per-call to ensure position_ids are generated correctly
    doc_tensor = model.encode_document(
        doc_text,
        processing_kwargs={
            "text": {
                "max_length": 512,
                "truncation": True,
                "padding": True,
            }
        }
    )
    doc_embedding = model.decode(doc_tensor, top_k=10)

    print("=== DOCUMENT ENCODING ===")
    print(f"Input:  '{doc_text}'")
    print(f"Output: {dict(doc_embedding)}\n")

    query_text = "What's the weather in NY now?"
    query_tensor = model.encode_query(
        query_text,
        processing_kwargs={
            "text": {
                "max_length": 512,
                "truncation": True,
                "padding": True,
            }
        }
    )
    query_embedding = model.decode(query_tensor)

    print("=== QUERY ENCODING ===")
    print(f"Input:  '{query_text}'")
    print(f"Output: {dict(query_embedding)}\n")

    similarity = model.similarity(query_tensor, doc_tensor)
    print(f"Similarity score: {similarity}")

    doc_tokens = set(dict(doc_embedding).keys())
    query_tokens = set(dict(query_embedding).keys())
    shared = doc_tokens & query_tokens
    print(f"Shared expanded tokens: {shared}")


if __name__ == "__main__":
    main()