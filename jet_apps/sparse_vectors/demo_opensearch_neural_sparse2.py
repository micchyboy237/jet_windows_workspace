"""
Demo: OpenSearch Neural Sparse Encoding (FIXED)
Model: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
Fix: Force CPU-first diagnostic run + vocab sanity check + cache-safe load
"""
import logging
import os
import torch
from sentence_transformers.sparse_encoder import SparseEncoder

# --- Fix 1: force synchronous CUDA errors so any future crash points to the real line ---
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sparse_demo")

MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"

def load_model(device: str) -> SparseEncoder:
    log.info(f"Loading model '{MODEL_ID}' on device='{device}' ...")
    # --- Fix 2: pin revision so config/tokenizer/remote-code always match ---
    model = SparseEncoder(
        MODEL_ID,
        trust_remote_code=True,
        revision="main",          # replace with a specific commit hash if you want it locked
        device=device,
    )
    log.info("Model loaded.")

    # --- Fix 3: vocab-size sanity check (this is what actually catches the bug) ---
    tok_vocab = len(model.tokenizer)
    embed_module = model.transformers_model.get_input_embeddings()
    embed_rows = embed_module.weight.shape[0]
    log.info(f"Tokenizer vocab size: {tok_vocab} | Embedding table rows: {embed_rows}")
    if tok_vocab > embed_rows:
        raise RuntimeError(
            f"MISMATCH: tokenizer can produce ids up to {tok_vocab - 1}, but the "
            f"embedding table only has {embed_rows} rows. This WILL crash on GPU. "
            f"Clear ~/.cache/huggingface/modules/transformers_modules and "
            f"~/.cache/huggingface/hub/models--opensearch-project--opensearch-neural-sparse-encoding-doc-v3-gte, "
            f"then retry."
        )
    return model


def main():
    # --- Fix 4: always prove it on CPU first ---
    cpu_model = load_model("cpu")

    doc_text = "Currently New York is rainy."
    log.info(f"Encoding document on CPU: '{doc_text}'")
    doc_tensor = cpu_model.encode_document(doc_text)
    doc_embedding = cpu_model.decode(doc_tensor, top_k=10)
    log.info(f"CPU encode succeeded. Top tokens: {dict(doc_embedding)}")

    # --- Only now move to GPU, once CPU proved the model/tokenizer are consistent ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cpu_model if device == "cpu" else load_model(device)

    doc_tensor = model.encode_document(doc_text)
    doc_embedding = model.decode(doc_tensor, top_k=10)
    print("=== DOCUMENT ENCODING ===")
    print(f"Input:  '{doc_text}'")
    print(f"Output: {dict(doc_embedding)}\n")

    query_text = "What's the weather in NY now?"
    query_tensor = model.encode_query(query_text)
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