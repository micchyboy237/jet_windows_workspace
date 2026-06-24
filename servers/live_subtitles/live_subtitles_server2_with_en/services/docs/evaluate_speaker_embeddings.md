# Speaker Embedding Model Evaluator

Benchmark and compare speaker embedding models on your own audio dataset.
Produces EER, minDCF, intra/inter similarity, and latency per model — in one run.

---

## What You Need

### 1. Files

Place these two files in the same directory:

```
your_project/
├── embedding_model_factory.py      ← existing factory
└── evaluate_speaker_embeddings.py  ← this evaluator
```

### 2. Dataset

One subfolder per speaker, at least 2 audio files each:

```
my_speakers/
├── alice/
│   ├── clip_01.wav
│   └── clip_02.wav
├── bob/
│   ├── clip_01.wav
│   └── clip_02.wav
└── carol/
    └── ...
```

Supported formats: `.wav` `.flac` `.mp3` `.ogg` `.m4a`

### 3. Dependencies

Install what you need for the models you want to run:

```bash
# Always required
pip install numpy torch librosa rich

# pyannote (default model)
pip install pyannote.audio

# SpeechBrain models (ECAPA or x-vector)
pip install speechbrain

# NeMo TitaNet
pip install nemo_toolkit soundfile
```

> You only need to install the packages for the models you plan to evaluate.

---

## Run It

**Evaluate all models:**

```bash
python evaluate_speaker_embeddings.py --dataset ./my_speakers
```

**Evaluate specific models:**

```bash
python evaluate_speaker_embeddings.py \
    --dataset ./my_speakers \
    --models pyannote speechbrain_ecapa
```

**All options:**

```bash
python evaluate_speaker_embeddings.py \
    --dataset   ./my_speakers   \  # required: path to your speaker folders
    --models    pyannote speechbrain_ecapa speechbrain_xvect nemo_titanet \
    --output    ./results       \  # where to save results (default: eval_results/)
    --cache     ./.emb_cache    \  # embedding cache — skips recomputation on reruns
    --device    cuda            \  # or cpu (auto-detected if omitted)
    --max-pos   10              \  # max same-speaker trial pairs per speaker
    --neg-ratio 1.0             \  # negative-to-positive trial ratio
    --min-utts  2               \  # min audio files required per speaker
    --seed      42
```

**Or use it from Python:**

```python
from evaluate_speaker_embeddings import run_evaluation
from pathlib import Path

results = run_evaluation(
    dataset_root=Path("my_speakers/"),
    model_types=["pyannote", "speechbrain_ecapa"],
    output_dir=Path("results/"),
    cache_dir=Path(".emb_cache/"),
)
```

---

## Output

**Console table** printed after each run:

```
┌──┬────────────────────┬─────┬───────┬──────────┬─────────┬─────────┬────────┬───────────┐
│ #│ Model              │ Dim │ EER ↓ │ minDCF ↓ │ Intra ↑ │ Inter ↓ │  Sep ↑ │ ms/file ↓ │
├──┼────────────────────┼─────┼───────┼──────────┼─────────┼─────────┼────────┼───────────┤
│ 1│ nemo_titanet       │ 192 │ 3.2%  │  0.0821  │  0.710  │  0.184  │  0.526 │      48.3 │
│ 2│ speechbrain_ecapa  │ 192 │ 5.1%  │  0.1204  │  0.640  │  0.223  │  0.417 │      31.7 │
│ 3│ pyannote           │ 512 │ 7.4%  │  0.1893  │  0.554  │  0.271  │  0.283 │      62.1 │
└──┴────────────────────┴─────┴───────┴──────────┴─────────┴─────────┴────────┴───────────┘
```

**Saved files** in `--output` directory:

- `results.json` — full metrics for every model
- `summary.md` — the same table in markdown

**Embedding cache** in `--cache` directory:

- `.npy` files per audio clip per model
- Re-running skips already-computed embeddings

---

## Metrics Explained

| Metric      | Meaning                                                     | Better |
| ----------- | ----------------------------------------------------------- | ------ |
| **EER**     | Equal Error Rate — where false accepts = false rejects      | Lower  |
| **minDCF**  | Minimum Detection Cost (NIST standard, p_target=0.01)       | Lower  |
| **Intra**   | Avg cosine similarity between clips of the _same_ speaker   | Higher |
| **Inter**   | Avg cosine similarity between clips of _different_ speakers | Lower  |
| **Sep**     | Intra − Inter — the discrimination gap                      | Higher |
| **ms/file** | Average embedding extraction time per audio file            | Lower  |

---

## Tips

- **Start small** — run 2–3 models on a small dataset first to check everything loads.
- **Use `--cache`** — embeddings are slow to compute; cache lets you re-score freely.
- **More utterances = more reliable results** — aim for 5+ clips per speaker.
- **`--neg-ratio 2.0`** — double the negative trials for a harder, more realistic benchmark.
- GPU is strongly recommended for NeMo TitaNet; pyannote and SpeechBrain run fine on CPU.
