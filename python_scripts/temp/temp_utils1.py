from rich.console import Console
console = Console()

def identify_character(audio_paths: List[str], known_character_refs: dict[str, List[str]]) -> dict[str, str]:
    """
    Identify most likely character for each query audio by highest avg cosine similarity to reference audios.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Extract reference embeddings once
    ref_embeddings: dict[str, np.ndarray] = {}
    for char, paths in known_character_refs.items():
        embeds = [model.get_embedding(p) for p in paths]
        ref_embeddings[char] = np.mean(embeds, axis=0)

    results = {}
    for audio in audio_paths:
        query_emb = model.get_embedding(audio)
        similarities = {
            char: cosine_similarity([query_emb], [ref_emb])[0][0]
            for char, ref_emb in ref_embeddings.items()
        }
        best_char = max(similarities, key=similarities.get)
        results[audio] = best_char
        console.print(f"[bold]{Path(audio).name}[/] → likely [green]{best_char}[/] (score: {similarities[best_char]:.3f})")

    return results