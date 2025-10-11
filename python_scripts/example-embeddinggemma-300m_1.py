from typing import TypedDict
from sentence_transformers import SentenceTransformer

# Define the SearchResult typed dictionary for structured output
class SearchResult(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str

# Load the pre-trained sentence transformer model
model = SentenceTransformer("google/embeddinggemma-300m")

# Define query and documents for similarity ranking
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Encode query and documents into embeddings
query_embeddings = model.encode_query(query)  # Shape: (768,)
document_embeddings = model.encode_document(documents)  # Shape: (4, 768)

# Compute cosine similarities between query and documents
similarities = model.similarity(query_embeddings, document_embeddings)[0]

# Rank documents by similarity score (highest to lowest) and create SearchResult list
results: list[SearchResult] = [
    {"rank": i + 1, "doc_index": idx, "score": float(score), "text": documents[idx]}
    for i, (idx, score) in enumerate(
        sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    )
]

# Print results for verification
# Expected output:
# Rank 1: Mars (score: 0.7087134122848511)
# Rank 2: Jupiter (score: 0.5932487845420837)
# Rank 3: Saturn (score: 0.5909743905067444)
# Rank 4: Venus (score: 0.49890926480293274)
print(f"Query: \"{query}\"")
for result in results:
    print(result)

