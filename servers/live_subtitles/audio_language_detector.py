import sys
import torchaudio
from pathlib import Path
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

class AudioLanguageDetector:
    """
    A reusable class for detecting spoken language from audio using
    SpeechBrain's ECAPA-TDNN models.
    """
    def __init__(self, model_source="speechbrain/lang-id-voxlingua107-ecapa"):
        """
        Initializes the language detector with a pre-trained model.

        Args:
            model_source (str): HuggingFace model identifier.
                - "speechbrain/lang-id-voxlingua107-ecapa": Supports 107 languages.
                - "speechbrain/lang-id-commonlanguage_ecapa": Supports 45 languages.
        """
        # The key fix: explicitly pass local_strategy to avoid symlinks on Windows
        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=Path("~/.cache/pretrained_models").expanduser() / model_source.split('/')[-1],
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY  # This is the crucial parameter
        )

    def detect_from_file(self, audio_path):
        """
        Detects language from an audio file path.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Detected language label.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
        return text_lab[0]

    def detect_from_bytes(self, audio_bytes, sample_rate=16000):
        """
        Detects language from raw audio bytes or a PyTorch tensor.

        Args:
            audio_bytes (torch.Tensor): Audio tensor.
            sample_rate (int): Sample rate of the audio.

        Returns:
            str: Detected language label.
        """
        # Ensure tensor is 2D (batch, samples)
        if audio_bytes.dim() == 1:
            audio_bytes = audio_bytes.unsqueeze(0)
        
        # Ensure correct format: normalize to 16kHz mono if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_bytes = resampler(audio_bytes)

        out_prob, score, index, text_lab = self.classifier.classify_batch(audio_bytes)
        return text_lab[0]

# Example Usage
if __name__ == "__main__":
    # Initialize once (the model will be downloaded and cached)
    print("Initializing AudioLanguageDetector...")
    detector = AudioLanguageDetector()
    print("Detector initialized successfully!\n")

    # Detect 'en' from file
    audio_path = "C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FunAudioLLM_SenseVoice/example/en.wav"
    if Path(audio_path).exists():
        language = detector.detect_from_file(audio_path)
        print(f"English audio - Detected language: {language}")

    # Detect 'ja' from file
    audio_path = "C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FunAudioLLM_SenseVoice/example/ja.wav"
    if Path(audio_path).exists():
        language = detector.detect_from_file(audio_path)
        print(f"Japanese audio - Detected language: {language}")
