from FlagEmbedding import FlagModel
from sklearn.cluster import KMeans

model = FlagModel("BAAI/bge-base-en-v1.5")

texts = [
    "Cats are lovely pets.",
    "Dogs make good companions.",
    "Quantum physics is a field of science.",
    "Entanglement and superposition are quantum concepts."
]

# get embeddings
embeddings = model.encode(texts)

# cluster
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(embeddings)

for text, label in zip(texts, labels):
    print(f"{label}: {text}")
