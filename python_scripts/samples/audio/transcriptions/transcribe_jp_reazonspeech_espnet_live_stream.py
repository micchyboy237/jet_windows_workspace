"""
Live microphone transcription using ReazonSpeech ESPnet ASR + sounddevice.
Captures audio from mic, feeds into the ASR model in real-time, and yields
transcription segments as they become available.

Usage:
    python transcribe_jp_reazonspeech_espnet_live_stream.py [--device DEVICE_ID]
    python transcribe_jp_reazonspeech_espnet_live_stream.py -l  # List devices
    python transcribe_jp_reazonspeech_espnet_live_stream.py -q  # Quiet mode
"""

import argparse
import sys
import time
import queue
import numpy as np
import sounddevice as sd
from reazonspeech.espnet.asr import load_model
from reazonspeech.espnet.asr.audio import norm_audio, audio_from_numpy
from reazonspeech.espnet.asr.ctc import split_text, find_blank
from reazonspeech.espnet.asr.interface import Segment, AudioData

# ─── Configuration ───────────────────────────────────────────────────────────
SAMPLERATE = 16000  # Must match model's expected sample rate
BLOCK_SIZE = 4096   # Samples per sounddevice callback (~256ms @ 16kHz)
WINDOW_SECONDS = 12.0
MIN_CHUNK_SECONDS = 2.5
PADDING = (16000, 8000)


class LiveTranscriber:
    """
    Captures live audio from microphone and transcribes it in real-time
    using the ReazonSpeech ESPnet ASR model.

    Audio pipeline:
        Mic → numpy array → AudioData → norm_audio (ensure 16kHz mono) → pad → model
    """

    def __init__(self, model, device=None, verbose=True):
        """
        Args:
            model: Loaded ReazonSpeech ASR model (Speech2Text)
            device: sounddevice input device ID (None = default)
            verbose: Whether to print status messages
        """
        self.model = model
        self.device = device
        self.verbose = verbose

        # Audio buffer management
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.stream = None

        # State tracking
        self.is_running = False
        self.total_samples = 0

        # Timing constants (derived from config)
        self.window_samples = int(WINDOW_SECONDS * SAMPLERATE)
        self.min_chunk_samples = int(MIN_CHUNK_SECONDS * SAMPLERATE)

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream — runs in high-priority audio thread."""
        if status:
            print(f"⚠️  Audio callback status: {status}", file=sys.stderr)
        # Thread-safe: push a copy into the queue
        self.audio_queue.put(indata.copy().flatten().astype(np.float32))

    def start_capture(self):
        """Start capturing audio from the microphone."""
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLERATE,
            device=self.device,
            channels=1,
            dtype=np.float32,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self.stream.start()

        if self.verbose:
            device_info = (
                sd.query_devices(self.device)
                if self.device is not None
                else sd.query_devices(sd.default.device[0])
            )
            print(f"🎤 Recording from: {device_info['name']}")
            print(f"   Sample rate: {SAMPLERATE} Hz, Block size: {BLOCK_SIZE}")
            print("   Speak now... (Ctrl+C to stop)\n")

    def stop_capture(self):
        """Stop capturing audio and close the stream."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            if self.verbose:
                print("⏹️  Audio capture stopped.")

    def _collect_audio(self, timeout=0.1):
        """
        Drain the audio queue into the internal buffer.

        Args:
            timeout: How long to wait for new blocks (seconds)

        Returns:
            Current buffer length in samples
        """
        try:
            while True:
                data = self.audio_queue.get(timeout=timeout)
                self.audio_buffer = np.concatenate([self.audio_buffer, data])
        except queue.Empty:
            pass
        return len(self.audio_buffer)

    def _process_samples(self, samples):
        """
        Normalize and transcribe a chunk of audio samples.

        Args:
            samples: numpy array of raw audio samples

        Yields:
            Segment objects with timestamps and text
        """
        # Step 1: Normalize — ensures 16kHz mono (matches transcribe_stream internals)
        audio_chunk = AudioData(samples, SAMPLERATE)
        audio_chunk = norm_audio(audio_chunk)
        normalized_samples = audio_chunk.waveform

        # Step 2: Pad and run ASR
        padded = np.pad(normalized_samples, PADDING, mode="constant")
        asr_text = self.model(padded)[0][0]

        if self.verbose and asr_text.strip():
            print(f"   📝 Recognized: {asr_text}")

        # Step 3: Split into timestamped segments
        for start, end, text in split_text(self.model, normalized_samples, asr_text):
            segment = Segment(
                start_seconds=(self.total_samples + start) / SAMPLERATE,
                end_seconds=(self.total_samples + end) / SAMPLERATE,
                text=text.strip(),
            )
            yield segment

    def process_and_transcribe(self):
        """
        Main loop: collect audio, process when enough is available,
        and yield transcription segments as they become ready.

        Yields:
            Segment objects
        """
        if self.verbose:
            print("🔄 Starting transcription loop...")

        while self.is_running:
            # Collect all available audio from the queue
            buffer_len = self._collect_audio(timeout=0.1)

            # Wait until we have enough audio for a meaningful chunk
            if buffer_len < self.min_chunk_samples:
                time.sleep(0.05)
                continue

            # Extract a window for blank detection
            window = (
                self.audio_buffer[:self.window_samples]
                if buffer_len > self.window_samples
                else self.audio_buffer
            )

            # Find silence boundary to determine chunk endpoint
            blank = find_blank(self.model, window)
            chunk_end = max(
                int((blank.start + blank.end) / 2),
                self.min_chunk_samples,
            )

            # Safety: don't process if chunk is still too small
            if chunk_end < self.min_chunk_samples:
                time.sleep(0.1)
                continue

            # Slice out the chunk to process
            samples_to_process = self.audio_buffer[:chunk_end]

            if self.verbose:
                duration = len(samples_to_process) / SAMPLERATE
                print(f"\n🔍 Processing chunk: {duration:.2f}s")

            # Transcribe and yield segments
            try:
                yield from self._process_samples(samples_to_process)
            except Exception as e:
                print(f"❌ ASR error: {e}", file=sys.stderr)

            # Advance the buffer: discard processed samples, keep the rest
            self.total_samples += len(samples_to_process)
            self.audio_buffer = self.audio_buffer[chunk_end:]

            # Brief pause to avoid CPU spinning
            time.sleep(0.01)

    def flush_remaining(self):
        """
        Process any remaining audio left in the buffer after capture stops.

        Yields:
            Segment objects
        """
        buffer_len = len(self.audio_buffer)

        if buffer_len < self.min_chunk_samples:
            if self.verbose:
                print(f"   Buffer too small to flush ({buffer_len} samples), skipping.")
            return

        if self.verbose:
            duration = buffer_len / SAMPLERATE
            print(f"🔚 Flushing remaining {duration:.2f}s...")

        try:
            yield from self._process_samples(self.audio_buffer)
        except Exception as e:
            print(f"❌ Flush error: {e}", file=sys.stderr)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def list_input_devices():
    """Print all available audio input devices."""
    print("\n📋 Available audio input devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        in_channels = dev.get('max_input_channels', 0)
        if in_channels > 0:
            default_mark = " ← default" if i == sd.default.device[0] else ""
            print(
                f"  [{i}] {dev['name']}"
                f"  (in: {in_channels} ch, {dev['default_samplerate']} Hz)"
                f"{default_mark}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Live microphone transcription using ReazonSpeech ESPnet ASR."
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="Input device ID (use -l to list devices)",
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output (only print segments)",
    )
    args = parser.parse_args()

    # ── List devices mode ──
    if args.list_devices:
        list_input_devices()
        return

    # ── Load model ──
    print("📦 Loading ReazonSpeech model...")
    model = load_model()
    print("✅ Model loaded.\n")

    # ── Create transcriber ──
    transcriber = LiveTranscriber(
        model=model,
        device=args.device,
        verbose=not args.quiet,
    )

    # ── Run ──
    transcriber.start_capture()

    try:
        for segment in transcriber.process_and_transcribe():
            print(
                f"🎯 [{segment.start_seconds:.2f}s - {segment.end_seconds:.2f}s] "
                f"{segment.text}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user. Stopping...")

    finally:
        transcriber.stop_capture()

        # Flush any remaining audio in the buffer
        print("🔚 Processing final audio...")
        for segment in transcriber.flush_remaining():
            print(
                f"🎯 [{segment.start_seconds:.2f}s - {segment.end_seconds:.2f}s] "
                f"{segment.text}"
            )

        print("✅ Done.")


if __name__ == "__main__":
    main()