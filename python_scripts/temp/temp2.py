import torch
from anime_speaker_embedding import AnimeSpeakerEmbedding
from pathlib import Path
from typing import List

from temp_utils1 import identify_character

device = "cuda" if torch.cuda.is_available() else "cpu"
# Choose variant based on need:
# variant="char" → distinguishes characters (even same VA with different styles)
# variant="va"   → groups by voice actor (ignores style differences)
model = AnimeSpeakerEmbedding(device=device, variant="char")  # or "va"


# Example usage
known_refs = {
    "Character_A": ["refs/char_a_sample1.wav", "refs/char_a_sample2.wav"],
    "Character_B": ["refs/char_b_sample1.wav"],
}
query_audios = ["unknown/line1.wav", "unknown/line2.wav"]
identify_character(query_audios, known_refs)