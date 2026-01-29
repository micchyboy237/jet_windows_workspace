# demo_python_interpreter.py
from smolagents import PythonInterpreterTool


def demo_python_interpreter():
    tool = PythonInterpreterTool(
        authorized_imports=["math", "random", "datetime"], timeout_seconds=15
    )

    # Simple calculation
    result = tool("x = 42 * 13 + 7; print(x)")
    print("Result 1:", result)

    # Multi-line with import
    code = """
import math
print(math.sqrt(169))
print(math.pi)
    """.strip()
    result = tool(code)
    print("Result 2:", result)

    # Using state preservation (multiple calls)
    tool("items = ['apple', 'banana', 'cherry']")
    result = tool("print(sorted(items))")
    print("Result 3 (using previous state):", result)


# demo_final_answer.py
from smolagents import FinalAnswerTool


def demo_final_answer():
    tool = FinalAnswerTool()

    # These are the kinds of calls agents usually make
    print(tool(42))
    print(tool("The capital of Japan is Tokyo"))
    print(tool({"answer": "yes", "confidence": 0.92}))
    print(tool(["Paris", "London", "Berlin"]))


# demo_user_input.py
from smolagents import UserInputTool


def get_input_with_default(
    question: str,
    hint: str | None = None,
    default: str | None = None,
) -> str:
    """
    Helper to ask question with optional hint and default value when empty input.

    Returns stripped user input or default when empty.
    """
    prompt_parts = [question]

    if hint:
        prompt_parts.append(f"  [hint: {hint}]")

    if default is not None:
        prompt_parts.append(f"  (press Enter to use '{default}')")
    elif hint:
        prompt_parts.append("  (press Enter to skip)")

    full_prompt = "".join(prompt_parts) + " → "

    user_input = input(full_prompt).strip()

    # If user gave nothing (empty after strip) → use default or empty
    if not user_input:
        return default if default is not None else ""

    return user_input


def demo_user_input():
    tool = UserInputTool()

    print("\n=== Improved interactive input examples ===\n")

    # Example 1: with hint + default
    answer1 = get_input_with_default(
        question="What is your favorite color?",
        hint="e.g. blue, forest green, #FF69B4",
        default="not specified",
    )
    print(f"You answered: {answer1!r}\n")

    # Example 2: required feeling, no default
    answer2 = get_input_with_default(
        question="How was your day in one word?",
        hint="great / okay / rough / chaotic",
        # no default → empty is allowed
    )
    print(f"You answered: {answer2!r}\n")

    # Example 3: using the original tool directly (for comparison)
    print("Original tool style (no default/hint):")
    answer3 = tool("Quick: cats or dogs?")
    print(f"You answered: {answer3!r}\n")


# demo_duckduckgo_search.py
from smolagents import DuckDuckGoSearchTool


def demo_duckduckgo():
    tool = DuckDuckGoSearchTool(max_results=6, rate_limit=1.5)

    queries = [
        "best lightweight python web framework 2025",
        "current version of PyTorch",
        "smolagents huggingface",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        print(tool(q))
        print()


# demo_google_search_serpapi.py
# (requires SERPAPI_API_KEY environment variable)
from smolagents import GoogleSearchTool


def demo_google_serpapi():
    try:
        tool = GoogleSearchTool(provider="serpapi")
        result = tool("Python 3.13 new features", filter_year=2024)
        print(result)
    except ValueError as e:
        print("Google SerpApi demo skipped — missing/invalid API key:", e)


# demo_api_web_search_brave.py
# (requires BRAVE_API_KEY environment variable)
from smolagents import ApiWebSearchTool


def demo_brave_search():
    try:
        tool = ApiWebSearchTool(
            rate_limit=2.0,
            # endpoint and headers already default to Brave
        )
        print(tool("AI agents open source frameworks comparison 2025"))
    except Exception as e:
        print("Brave Search demo skipped:", str(e))


# demo_visit_webpage.py
from smolagents import VisitWebpageTool


def demo_visit_webpage():
    tool = VisitWebpageTool(max_output_length=12000)

    urls = [
        "https://huggingface.co/docs/hub/index",
        "https://peps.python.org/pep-0745/",
        "https://pytorch.org/blog/pytorch-2.6/",
    ]

    for url in urls:
        print(f"\n=== {url} ===\n")
        content = tool(url)
        print(content[:400], "..." if len(content) > 400 else "")
        print()


# demo_wikipedia_search.py
from smolagents import WikipediaSearchTool


def demo_wikipedia():
    tool = WikipediaSearchTool(
        user_agent="DemoBot (demo@example.com)",
        language="en",
        content_type="summary",  # or "text"
        extract_format="WIKI",
    )

    topics = [
        "Large language model",
        "Retrieval-augmented generation",
        "Mixture of Experts",
    ]

    for topic in topics:
        print(f"\nWikipedia → {topic}")
        print("-" * 50)
        print(tool(topic))
        print()


# demo_speech_to_text.py
# from smolagents import SpeechToTextTool

# Custom SpeechToTextTool with proper audio loading and faster-whisper support
from typing import Any, Dict, Tuple, Union, Literal, Optional

import numpy as np
import soundfile as sf
import io
import requests

from smolagents.tools import PipelineTool
from smolagents.agent_types import AgentAudio

DeviceType = Literal["auto", "cpu", "cuda", "mps"]
ComputeType = Literal["default", "float32", "float16", "int8", "int8_float16"]

class SpeechToTextTool(PipelineTool):
    """
    Tool that transcribes audio to text using faster-whisper (or falls back to transformers).
    Handles various audio input formats: local path, URL, numpy array, bytes.

    Now defaults to a local CTranslate2-converted model folder for faster loading.
    You can still pass a model size string ("large-v3", "medium", ...) to download from HF,
    or any other local path / HF repo ID.
    """
    default_checkpoint = r"C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-large-v3-ct2"
    description = "Transcribes an audio file or buffer into text. Returns the transcribed string."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "any",
            "description": "Audio source: local file path, http/https URL, numpy array, or bytes",
        }
    }
    output_type = "string"

    use_faster_whisper: bool = True

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[DeviceType] = None,
        compute_type: Optional[ComputeType] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.device = device
        self.compute_type = compute_type

    def __new__(cls, *args, **kwargs):
        if not cls.use_faster_whisper:
            from transformers.models.whisper import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
            cls.pre_processor_class = WhisperProcessor
            cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def setup(self) -> None:
        if getattr(self, 'is_initialized', False):
            return

        if self.use_faster_whisper:
            try:
                import torch
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                ) from exc

            # If user didn't pass model → use our local large-v3-ct2 default
            # If user passes e.g. "large-v3" → downloads from HF
            # If user passes another local folder path → loads that
            model_id = self.model if self.model is not None else self.default_checkpoint

            if self.device is None or self.device == "auto":
                if torch.cuda.is_available():
                    selected_device = "cuda"
                elif hasattr(torch, "backends") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    selected_device = "mps"
                else:
                    selected_device = "cpu"
            else:
                selected_device = self.device

            compute_type = self.compute_type if self.compute_type is not None else "default"

            self.model = WhisperModel(
                model_id,
                device=selected_device,
                compute_type=compute_type,
                cpu_threads=0,           # 0 = auto
                num_workers=1,
            )
        else:
            super().setup()

        self.is_initialized = True

    # ──────────────────────────────────────────────────────────────
    # Rest of class (_load_audio_raw, _prepare_audio, encode, forward, decode) unchanged
    # ──────────────────────────────────────────────────────────────

    def _load_audio_raw(self, audio: Any) -> Tuple[np.ndarray, int]:
        """
        Load audio data → returns (waveform: float64, sample_rate: int)
        Supports: path (str), URL (str), bytes/bytearray, np.ndarray
        """
        if isinstance(audio, np.ndarray):
            # Assume already loaded – we still want sample rate info later
            # For simplicity we return dummy sr=16000 – faster-whisper can handle it
            return audio, 16000

        if isinstance(audio, (bytes, bytearray, io.BytesIO)):
            file_like = io.BytesIO(audio) if not isinstance(audio, io.BytesIO) else audio
            waveform, sample_rate = sf.read(file_like)
            return waveform, sample_rate

        if isinstance(audio, str):
            if audio.startswith(("http://", "https://")):
                response = requests.get(audio, timeout=30)
                response.raise_for_status()
                waveform, sample_rate = sf.read(io.BytesIO(response.content))
                return waveform, sample_rate
            else:
                # local file path
                waveform, sample_rate = sf.read(audio)
                return waveform, sample_rate

        # Try AgentAudio fallback if available
        try:
            audio_obj = AgentAudio(audio)
            raw = audio_obj.to_raw()  # assuming this returns np.ndarray
            return raw, 16000
        except (AttributeError, ImportError, TypeError):
            pass

        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    def _prepare_audio(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert to float32, normalize to [-1.0, 1.0] range (required by faster-whisper / ONNX)
        """
        if waveform.ndim > 1:
            # Convert multi-channel → mono (average)
            waveform = np.mean(waveform, axis=1)

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        max_abs = np.max(np.abs(waveform))
        if max_abs > 1.0 + 1e-6:
            waveform /= max_abs
        elif max_abs < 1e-8:
            # Protect against division by zero / silent audio
            waveform.fill(0.0)

        return waveform

    def encode(self, audio: Any) -> Dict[str, Any]:
        """
        Prepare audio for transcription.
        faster-whisper .transcribe() accepts np.ndarray directly (float32, mono, [-1,1])
        """
        waveform, _sample_rate = self._load_audio_raw(audio)
        prepared = self._prepare_audio(waveform)
        return {"audio": prepared}

    def forward(self, inputs: Dict[str, Any]) -> Any:
        if self.use_faster_whisper:
            segments, info = self.model.transcribe(
                inputs["audio"],
                language="en",
                beam_size=5,
                log_progress=True,
                # vad_filter=True,
                # vad_parameters=dict(min_silence_duration_ms=500),
                # without_timestamps=False,
            )
            return {"segments": list(segments), "info": info}
        else:
            # transformers path
            return self.model.generate(inputs["input_features"])

    def decode(self, outputs: Any) -> str:
        if self.use_faster_whisper:
            segments = outputs["segments"]
            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text and segment.text.strip()
            )
            return text.strip()
        else:
            return self.pre_processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()

def demo_transcriber():
    try:
        tool = SpeechToTextTool(device="cuda", compute_type="int8_float16")

        # Examples of inputs the tool accepts:
        #   - local path
        #   - http/https url
        #   - numpy array
        #   - bytes

        result = tool(
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
            # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\sample_en_sound_1.flac"
        )
        print("Transcription:", result)

        # Local file example (uncomment if you have the file)
        # result = tool("./my_recording.mp3")
        # print(result)

    except ImportError:
        print("Transformers + torch not installed → skipping transcriber demo")
    except Exception as e:
        print("Transcriber demo failed:", str(e))


if __name__ == "__main__":
    print("═══════════════════════════════════════════════")
    print("   Smolagents default tools — demo usage")
    print("═══════════════════════════════════════════════\n")

    demos = [
        ("PythonInterpreterTool", demo_python_interpreter),
        ("FinalAnswerTool", demo_final_answer),
        ("UserInputTool", demo_user_input),
        ("DuckDuckGoSearchTool", demo_duckduckgo),
        # ("GoogleSearchTool (SerpApi)", demo_google_serpapi),        # needs API key
        # ("ApiWebSearchTool (Brave)", demo_brave_search),           # needs API key
        ("VisitWebpageTool", demo_visit_webpage),
        ("WikipediaSearchTool", demo_wikipedia),
        ("SpeechToTextTool (Whisper)", demo_transcriber),
    ]

    for name, func in demos:
        print(f"\n► {name}")
        print("─" * (len(name) + 4))
        try:
            func()
        except Exception as e:
            print(f"  → Demo failed: {type(e).__name__}: {e}")
        print()
