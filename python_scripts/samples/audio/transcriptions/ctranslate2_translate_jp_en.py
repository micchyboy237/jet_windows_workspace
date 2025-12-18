"""
Docstring for python_scripts.samples.audio.ctranslate2_translate_jp_en

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir C:/Users/druiv/.cache/hf_ctranslate2_models/faster-whisper-large-v3-int8_float16
"""

import ctranslate2
import librosa
import transformers

audio_file = "C:/Users/druiv/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio/data/1.wav"
model_path = "C:/Users/druiv/.cache/hf_ctranslate2_models/faster-whisper-large-v3-int8_float16"

# Load and resample the audio file.
audio, _ = librosa.load(audio_file, sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v3")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on CPU.
model = ctranslate2.models.Whisper(model_path)

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))

# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

# Run generation for the 30-second window.
results = model.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
print(transcription)