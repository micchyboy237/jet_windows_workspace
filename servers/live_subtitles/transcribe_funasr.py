from typing import Literal
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile
import os

SupportedLanguage = Literal["auto", "en", "ja", "ko", "yue"]


class SenseVoiceTranscriber:
    """Reusable transcriber class to avoid reloading the model each time."""
    
    def __init__(self, device: str = "cuda:0", model_name: str = "iic/SenseVoiceSmall"):
        self.model = AutoModel(
            model=model_name,
            trust_remote_code=True,
            device=device,
        )

        print(f"Model successfully loaded on: {device}")
        if device == "cuda:0":
            print("🎉 CUDA GPU acceleration is active.")
        else:
            print("Running on CPU (slower but always stable).")
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: SupportedLanguage = "auto",
        use_itn: bool = True,
        suffix: str = ".wav"
    ) -> str:
        """
        Transcribe audio from raw bytes using SenseVoice.
        
        Args:
            audio_bytes: Raw audio data as bytes (WAV, MP3, etc.).
            language: Target language ("auto", "en", "ja", "ko", "yue").
            use_itn: Apply Inverse Text Normalization if True.
            suffix: Temporary file suffix matching the audio format.
            
        Returns:
            The transcribed text.
        """
        tmp_path = None
        try:
            # Write bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Generate transcription
            raw_results = self.model.generate(
                input=tmp_path,
                cache={},
                language=language,
                use_itn=use_itn,
                output_timestamp=True,
            )
            first = raw_results[0]

            print(f"First result:\n{first!r}")
            
            # Extract clean text
            clean_text = rich_transcription_postprocess(first["text"])
            return clean_text
            
        finally:
            # Clean up the temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def transcribe_audio(
    audio_bytes: bytes,
    language: SupportedLanguage = "auto",
    *,
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
    **kwargs,
) -> str:
    """
    Transcribe raw PCM int16 bytes.
    Designed for live server usage.
    """
    
    result = transcriber.transcribe_bytes(audio_bytes, language=language)

    return result


# Initialize once and reuse
transcriber = SenseVoiceTranscriber()
model = transcriber.model

# ===============================================
# Example Usage
# ===============================================
if __name__ == "__main__":
    audio_file_en = f"{model.model_path}/example/en.mp3"
    audio_file_ja = f"{model.model_path}/example/ja.mp3"

    # Example 1: English
    with open(audio_file_en, "rb") as f:
        audio_bytes = f.read()
    transcription = transcribe_audio(audio_bytes, language="en")
    print(f"EN Transcription: {transcription}")
    
    # Example 2: Japanese
    with open(audio_file_ja, "rb") as f:
        audio_bytes = f.read()
    transcription = transcribe_audio(audio_bytes, language="ja")
    print(f"JA Transcription: {transcription}")
