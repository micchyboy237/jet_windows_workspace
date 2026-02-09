from wtpsplit import SaT
import numpy as np
import json

sat = SaT("sat-12l-sm", style_or_domain="ud", language="en")

text = "This is a test. Another sentence without space herebutstillreadable?"

# Get newline / boundary probabilities
probs: np.ndarray = sat.predict_proba(text)

print(f"Length of probs: {len(probs)} (should ≈ len(text))")
print("First 10 probs:", probs[:10])

# Find probable sentence boundaries (example threshold)
boundaries = np.where(probs > 0.4)[0]
print("Probable boundary positions:", boundaries)
print(f"Probable boundary positions ({len(boundaries)}):", boundaries)

# Optional: with paragraph segmentation mode
probs_tuple = sat.predict_proba(
    text,
    return_paragraph_probabilities=True
)  # returns (sentence_probs, newline_probs) — but both are the same!
print(f"Paragraph probs ({len(probs_tuple)}):\n", probs_tuple)
