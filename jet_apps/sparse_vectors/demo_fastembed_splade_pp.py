"""
Demo: FastEmbed SPLADE++ (ONNX-optimized for production)
Install: pip install -q fastembed
"""
import numpy as np
from fastembed import SparseTextEmbedding

def main():
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    documents = [
        "Apple releases new iPhone with titanium frame",
        "Chandrayaan-3 landed on the Moon in August 2023",
        "UV-A light causes skin aging and cataracts",
    ]

    # Batch encode → list of SparseEmbedding(indices, values)
    embeddings = list(model.embed(documents, batch_size=4))

    for i, emb in enumerate(embeddings):
        print(f"[{i}] {documents[i]}")
        print(f"    Active dims: {len(emb.indices)}")
        print(f"    Top 5 indices: {emb.indices[:5]}")
        print(f"    Top 5 weights: {np.round(emb.values[:5], 4)}")

        # Convert to vector DB format (Qdrant/Pinecone compatible)
        db_format = {int(idx): float(val) for idx, val in zip(emb.indices, emb.values)}
        print(f"    DB-ready entries: {len(db_format)}")
        print()

    # List all supported sparse models
    print("Available sparse models:")
    for m in SparseTextEmbedding.list_supported_models():
        print(f"  - {m['model']}")

if __name__ == "__main__":
    main()