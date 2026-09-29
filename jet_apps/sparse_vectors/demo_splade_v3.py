"""
Demo: SPLADE-v3-DistilBERT (State-of-the-art pure sparse)
Asymmetric query/document encoding via Sentence Transformers v5+
Install: pip install -U sentence-transformers
"""
from sentence_transformers import SparseEncoder

def main():
    model = SparseEncoder("naver/splade-v3-distilbert")

    queries = ["what causes aging fast"]
    documents = [
        "UV-A light specifically is what mainly causes tanning, skin aging, and cataracts.",
        "Alzheimer's disease usually worsens slowly depending on genetic makeup.",
        "Bell's palsy and extreme tiredness and hepatitis have shared causes.",
    ]

    # Asymmetric encoding (critical for SPLADE-v3 performance)
    query_emb = model.encode_query(queries)
    doc_emb = model.encode_document(documents)

    print(f"Query shape: {query_emb.shape}")   # [1, 30522]
    print(f"Docs shape:  {doc_emb.shape}")      # [3, 30522]

    # Similarity via dot product (native sparse operation)
    similarities = model.similarity(query_emb, doc_emb)
    print(f"\nSimilarities: {similarities}")
    # Expected: tensor([[14.47, 9.44, 5.69]]) — doc[0] wins correctly

    # Interpretability: decode top tokens
    top_tokens = model.decode(doc_emb, top_k=8)
    for i, tokens in enumerate(top_tokens):
        pairs = ", ".join(f'("{t.strip()}", {v:.2f})' for t, v in tokens)
        print(f"  Doc[{i}] top tokens: {pairs}")

    # Control sparsity for production storage savings
    compact_emb = model.encode_document(documents, max_active_dims=32)
    stats = SparseEncoder.sparsity(compact_emb)
    print(f"\nWith max_active_dims=32:")
    print(f"  Sparsity: {stats['sparsity_ratio']:.2%}")
    print(f"  Avg active dims: {stats['active_dims']:.1f}")

if __name__ == "__main__":
    main()