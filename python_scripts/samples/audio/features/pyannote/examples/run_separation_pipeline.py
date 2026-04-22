import os
import torch
import torchaudio
import scipy.io.wavfile
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pathlib import Path
import shutil

# ── Windows symlink workaround ────────────────────────────────────────────────
# Tell HuggingFace hub to store files directly (no symlinks) on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Tell SpeechBrain where to save its model files so savedir is never None.
_SCRIPT_DIR = r"C:\Users\druiv\.cache\pretrained_models\spkrec-ecapa-voxceleb"
_SB_CACHE   = Path(_SCRIPT_DIR) / "speechbrain"
_SB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SPEECHBRAIN_LOCALDIR", str(_SB_CACHE))
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers_mono_16k.wav"
HF_TOKEN   = os.getenv("HF_TOKEN")
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading speech-separation-ami-1.0 pipeline …")
pipeline = Pipeline.from_pretrained(
    "pyannote/speech-separation-ami-1.0",
    token=HF_TOKEN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Running on: {device}")
pipeline.to(device)

print("Pre-loading audio into memory …")
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

print("Running pipeline (this may take a while for long files) …")
with ProgressHook() as hook:
    diarization, sources = pipeline(audio_in_memory, hook=hook)

rttm_path = OUTPUT_DIR / "diarization.rttm"
with open(rttm_path, "w") as f:
    diarization.write_rttm(f)
print(f"  Saved diarization → {rttm_path}")

print("Saving per-speaker WAV files …")
labels = diarization.labels()
for s, speaker in enumerate(labels):
    spk_audio = sources.data[:, s]
    max_val = abs(spk_audio).max()
    if max_val > 0:
        spk_audio = spk_audio / max_val
    spk_audio_int16 = (spk_audio * 32767).astype("int16")
    out_path = OUTPUT_DIR / f"{speaker}.wav"
    scipy.io.wavfile.write(out_path, 16_000, spk_audio_int16)
    print(f"  Saved {out_path}")

print("\n── Diarization Timeline ──────────────────────────────────")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s → {turn.end:6.1f}s]  {speaker}")

print("\nDone! Check the rttm file")
print(rttm_path)