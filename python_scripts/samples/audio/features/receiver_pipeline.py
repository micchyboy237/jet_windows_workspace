import asyncio
import ctypes
import json
import platform
import os
import shutil
import sys
import time
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import websockets
from rich.console import Console

sys.path.append(r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\ten-vad\include")
from ten_vad import TenVad

# ====================== CONFIG ======================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 320  # Must match sender's OPUS_FRAME_SIZE
DTYPE = "int16"
DEBUG_DECODE = True
DEBUG_VAD = True

# Minimum speech duration (seconds) to consider a segment valid
MIN_SPEECH_SEC = 0.3

# New constants for accurate segmentation
VAD_HOP_SIZE = 160
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT_SEC = 2.0  # how long without packets = end of segment

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

SEGMENTS_DIR = OUTPUT_DIR / "segments"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_DURATION_SEC = 300  # 5 minutes
MAX_ROLLING_SAMPLES = SAMPLE_RATE * ROLLING_DURATION_SEC


console = Console()


def log(level: str, message: str):
    """Rich colored logging that works on Windows + macOS + Linux"""
    # Windows does not support %f in strftime → use datetime for milliseconds
    from datetime import datetime
    now = datetime.now()
    ts = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"

    color_map = {
        "INIT": "cyan",
        "CONN": "green",
        "PACKET": "blue",
        "DECODE": "magenta",
        "SEGMENT": "yellow",
        "SHUTDOWN": "bright_black",
        "INFO": "white",
        "ERROR": "red bold",
    }
    color = color_map.get(level.upper(), "white")
    console.print(f"[{ts}] [[bold {color}]{level.upper():<8}[/]] {message}")


# ====================== MODULAR FUNCTIONS (one job each) ======================
def compute_speech_probs_and_energies(
    pcm: np.ndarray, vad: TenVad
) -> tuple[list[float], list[float]]:
    """Run TenVad + RMS energy on every 10 ms hop. Returns two clean lists.
    Also logs each frame's values if DEBUG_VAD is enabled."""
    probs = []
    energies = []
    hop = VAD_HOP_SIZE
    for i in range(0, len(pcm), hop):
        frame = pcm[i : i + hop]
        if len(frame) < hop:
            frame = np.pad(frame, (0, hop - len(frame)), mode="constant")
        prob, _ = vad.process(frame)  # we only need the probability
        rms = float(np.sqrt(np.mean(frame.astype(float) ** 2)))
        probs.append(prob)
        energies.append(rms)

        if DEBUG_VAD:
            log(
                "VAD",
                f"frame={i // hop:04d} prob={prob:.4f} rms={rms:.2f}",
            )

    return probs, energies


def detect_speech_boundaries(
    probs: list[float], energies: list[float], hop_sec: float = 0.01
) -> dict:
    """Best-accuracy logic: state machine with threshold + hangover (same style as sender)."""
    is_speaking = False
    hangover_counter = 0
    start_frame = None
    end_frame = None
    HANGOVER = 12
    THRESHOLD = VAD_THRESHOLD

    for i, prob in enumerate(probs):
        voice = prob > THRESHOLD
        if voice:
            if not is_speaking:
                is_speaking = True
                start_frame = i
            hangover_counter = HANGOVER
        elif is_speaking:
            if hangover_counter > 0:
                hangover_counter -= 1
            else:
                is_speaking = False
                end_frame = i
                break  # one main utterance per buffer

    if is_speaking:  # still speaking at the end of buffer
        end_frame = len(probs) - 1

    if start_frame is None:  # fallback – treat whole buffer as speech
        start_frame = 0
        end_frame = len(probs) - 1

    start_sec = round(start_frame * hop_sec, 3)
    end_sec = round(end_frame * hop_sec, 3)
    duration_sec = round(end_sec - start_sec, 3)

    active_probs = probs[start_frame : end_frame + 1]
    active_rms = energies[start_frame : end_frame + 1]
    avg_prob = round(sum(active_probs) / len(active_probs), 4) if active_probs else 0.0
    avg_rms = round(sum(active_rms) / len(active_rms), 4) if active_rms else 0.0

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "has_valid_speech": duration_sec >= MIN_SPEECH_SEC,
        "average_prob": avg_prob,
        "average_rms": avg_rms,
        "total_duration_sec": round(len(probs) * hop_sec, 3),
        "num_frames": len(probs),
    }


def save_segment_files(
    segment_dir: Path,
    pcm: np.ndarray,
    probs: list[float],
    energies: list[float],
    metadata: dict,
):
    """Saves exactly the 5 files you asked for under each segment_### folder."""
    # 1. sound.wav
    wav_path = segment_dir / "sound.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.astype(np.int16).tobytes())

    # 2. speech_probs.json
    (segment_dir / "speech_probs.json").write_text(json.dumps(probs, indent=2))

    # 3. energies.json
    (segment_dir / "energies.json").write_text(json.dumps(energies, indent=2))

    # 4. segment.json
    (segment_dir / "segment.json").write_text(json.dumps(metadata, indent=2))

    # 5. speech_and_energy.png (two clean charts)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    times = np.arange(len(probs)) * (VAD_HOP_SIZE / SAMPLE_RATE)
    axs[0].plot(times, probs, color="blue")
    axs[0].set_title("Speech Probabilities")
    axs[0].set_ylabel("Probability")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(times, energies, color="orange")
    axs[1].set_title("Energy RMS")
    axs[1].set_ylabel("RMS")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(segment_dir / "speech_and_energy.png")
    plt.close()

    log("SEGMENT", f"✅ All 5 files saved → {segment_dir.name}")


def save_global_files(segments: list, output_dir: Path):
    """Update the two global files in OUTPUT_DIR."""
    (output_dir / "segments.json").write_text(json.dumps(segments, indent=2))

    total_segments = len(segments)
    total_dur = sum(s.get("duration_sec", 0) for s in segments)
    summary = {
        "total_segments": total_segments,
        "total_speech_duration_sec": round(total_dur, 3),
        "average_segment_duration_sec": round(total_dur / total_segments, 3)
        if total_segments
        else 0,
        "average_prob": round(
            sum(s.get("average_prob", 0) for s in segments) / total_segments, 4
        )
        if total_segments
        else 0,
        "average_rms": round(
            sum(s.get("average_rms", 0) for s in segments) / total_segments, 4
        )
        if total_segments
        else 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # === Cleanup old segments not in last 5 minutes ===
    cleanup_old_segments(output_dir)


def save_rolling_audio(rolling_pcm: list[np.ndarray], output_dir: Path):
    """Save the continuous last 5 minutes of audio as sound_last_5_mins.wav"""
    if not rolling_pcm:
        return
    full_pcm = np.concatenate([arr.flatten() for arr in rolling_pcm])
    wav_path = output_dir / "sound_last_5_mins.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(full_pcm.astype(np.int16).tobytes())
    log(
        "SEGMENT",
        f"✅ Updated rolling 5-min audio → {wav_path.name} ({len(full_pcm) / SAMPLE_RATE:.1f}s)",
    )


def cleanup_old_segments(output_dir: Path):
    """Remove segment folders whose audio is no longer present in the last 5 minutes
    of sound_last_5_mins.wav. Uses global timestamps stored in segments.json."""
    segments_json_path = output_dir / "segments.json"
    if not segments_json_path.exists():
        return
    try:
        segments = json.loads(segments_json_path.read_text())
        if not segments:
            return

        # Calculate the current global session time and rolling window start (in sec)
        current_global_sec = time.time() - pipeline.session_start_time
        min_keep_global_sec = max(0, current_global_sec - ROLLING_DURATION_SEC)

        to_keep = []
        removed_count = 0
        for seg in segments:
            seg_dir_name = seg.get("segment_dir")
            if not seg_dir_name:
                continue
            seg_path = SEGMENTS_DIR / seg_dir_name
            # Use the global timestamps we stored when the segment was created
            global_start = seg.get("global_start_sec")
            global_end = seg.get("global_end_sec")
            if global_start is None or global_end is None:
                # fallback for any legacy segments (should never happen after patch)
                to_keep.append(seg)
                continue
            # Keep only segments that overlap with the current rolling window
            # (i.e. their audio still exists inside sound_last_5_mins.wav)
            if global_end > min_keep_global_sec:
                to_keep.append(seg)
            else:
                if seg_path.exists():
                    shutil.rmtree(seg_path, ignore_errors=True)
                    removed_count += 1
                    log(
                        "SEGMENT",
                        f"🗑️ Removed old segment: {seg_dir_name} "
                        f"(global {global_start:.1f}s–{global_end:.1f}s)",
                    )
        # Update segments.json with only kept segments
        if removed_count > 0:
            (output_dir / "segments.json").write_text(json.dumps(to_keep, indent=2))
            log(
                "SEGMENT",
                f"✅ Cleaned up {removed_count} old segment(s) outside last 5 minutes",
            )

    except Exception as e:
        log("ERROR", f"Failed to cleanup old segments: {e}")


class OpusDecoder:
    def __init__(self):
        self.lib = None
        self.decoder = None
        self._load_opus_library()
        self._create_decoder()

    def _load_opus_library(self):
        system = platform.system().lower()

        if system == "darwin":  # macOS
            possible_paths = [
                "/opt/homebrew/lib/libopus.dylib",   # Apple Silicon Homebrew
                "/usr/local/lib/libopus.dylib",      # Intel Homebrew
            ]
            lib_name = "libopus.dylib"
        elif system == "linux":
            possible_paths = ["/usr/lib/x86_64-linux-gnu/libopus.so"]
            lib_name = "libopus.so"
        elif system == "windows":
            opus_dll_dir = str(Path("~/.cache/opus_dll").expanduser().resolve())
            possible_paths = [
                os.path.join(opus_dll_dir, "opus.dll"),           # same folder as script (recommended)
                os.path.join(opus_dll_dir, "libopus.dll"),
                r"C:\Program Files\opus\bin\opus.dll",                         # common install location
                r"C:\Windows\System32\opus.dll",
            ]
            lib_name = "opus.dll"
        else:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")

        # Try possible paths first
        for p in possible_paths:
            if os.path.exists(p):
                log("INIT", f"Loading Opus from: {p}")
                self.lib = ctypes.CDLL(p)
                return

        # If not found in known paths, try loading by name (lets Windows search PATH)
        try:
            log("INIT", f"Opus not found in standard paths. Trying to load '{lib_name}' via system PATH")
            self.lib = ctypes.CDLL(lib_name)
            return
        except OSError as e:
            pass

        # Final helpful error
        if system == "windows":
            raise FileNotFoundError(
                "libopus (opus.dll) not found.\n\n"
                "How to fix on Windows:\n"
                "1. Download Opus prebuilt binaries from https://opus-codec.org/downloads/\n"
                "2. Extract and copy 'opus.dll' (from bin/ folder) into the same folder as receiver_pipeline.py\n"
                "   OR add the bin/ folder to your Windows PATH.\n"
                "Recommended: place opus.dll next to this script."
            )
        else:
            raise FileNotFoundError(f"libopus not found. Tried paths for {platform.system()}")

    def _create_decoder(self):
        if not self.lib:
            raise RuntimeError("Opus library failed to load")

        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_short),
            ctypes.c_int,   # frame size
            ctypes.c_int,   # decode_fec
        ]
        self.lib.opus_decode.restype = ctypes.c_int

        self.decoder = self.lib.opus_decoder_create(SAMPLE_RATE, CHANNELS)
        if not self.decoder:
            raise RuntimeError("Failed to create Opus decoder")
        log("INIT", "✅ Opus decoder created successfully")

    def decode(self, opus_data: bytes) -> np.ndarray:
        # unchanged
        if not opus_data:
            log("DECODE", "Received empty opus data")
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        data_ptr = (ctypes.c_ubyte * len(opus_data)).from_buffer_copy(opus_data)
        pcm_buffer = (ctypes.c_short * (BLOCK_SIZE * CHANNELS))()

        length = self.lib.opus_decode(
            self.decoder, data_ptr, len(opus_data), pcm_buffer, BLOCK_SIZE, 0
        )

        if length < 0:
            log(
                "DECODE",
                f"ERROR: opus_decode returned {length} (input={len(opus_data)} bytes)",
            )
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        pcm = np.ctypeslib.as_array(pcm_buffer, shape=(BLOCK_SIZE, CHANNELS)).astype(
            DTYPE
        )

        if DEBUG_DECODE:
            rms = np.sqrt(np.mean(pcm.astype(float) ** 2))
            log(
                "DECODE",
                f"SUCCESS → {len(opus_data)} bytes → {BLOCK_SIZE} samples | RMS={rms:.2f}",
            )

        return pcm


class ReceiverPipeline:
    def __init__(self):
        self.decoder = OpusDecoder()
        self.last_seq = -1
        # Buffers & tools for segmentation
        self.audio_buffer: list[np.ndarray] = []
        self.is_speaking = False
        self.silence_counter = 0
        self.segment_num = 0
        self.segments: list[dict] = []
        self.min_speech_sec = MIN_SPEECH_SEC
        self.vad = TenVad(hop_size=VAD_HOP_SIZE, threshold=VAD_THRESHOLD)
        self.rolling_pcm: list[np.ndarray] = []  # rolling buffer for last 5 min
        # Global monotonic timeline for accurate cleanup of old segments
        self.session_start_time = time.time()
        self.monitor_task = None
        log("INIT", "✅ ReceiverPipeline initialized (segmentation mode)")

    def decode_packet(self, raw_msg: bytes):
        """Parse, decode, and now also buffer the PCM for segmentation and rolling 5-min file."""
        try:
            if not raw_msg:
                return

            parts = raw_msg.split(b"|", 1)
            if len(parts) != 2:
                return

            seq = int(parts[0])
            opus_data = parts[1]

            if seq > self.last_seq:
                pcm = self.decoder.decode(opus_data)

                # --- Quick VAD probe for debug logging ---
                pcm_flat = pcm.flatten()
                chunk_probs = []
                for i in range(0, len(pcm_flat), VAD_HOP_SIZE):
                    frame = pcm_flat[i : i + VAD_HOP_SIZE]
                    if len(frame) < VAD_HOP_SIZE:
                        frame = np.pad(frame, (0, VAD_HOP_SIZE - len(frame)))
                    prob, _ = self.vad.process(frame)
                    chunk_probs.append(prob)

                avg_prob = sum(chunk_probs) / len(chunk_probs) if chunk_probs else 0.0
                state = "SPEECH" if avg_prob > VAD_THRESHOLD else "SILENCE"

                # === Always append to rolling 5-min buffer ===
                self.rolling_pcm.append(pcm.copy())  # copy for safety
                total_samples = sum(len(arr) for arr in self.rolling_pcm)
                while total_samples > MAX_ROLLING_SAMPLES and self.rolling_pcm:
                    oldest = self.rolling_pcm.pop(0)
                    total_samples -= len(oldest)
                # =====================================================

                # VAD-driven segmentation
                if avg_prob > VAD_THRESHOLD:
                    if not self.is_speaking:
                        log("SEGMENT", "🎙️ Speech started")
                        self.is_speaking = True
                        self.audio_buffer = []
                    self.silence_counter = 0
                    self.audio_buffer.append(pcm)
                else:
                    if self.is_speaking:
                        self.silence_counter += 1
                        if self.silence_counter > 12:
                            log("SEGMENT", "🛑 Speech ended → saving segment")
                            self.is_speaking = False
                            self.process_current_segment()
                            self.audio_buffer = []
                            self.silence_counter = 0

                self.last_seq = seq

                rms = np.sqrt(np.mean(pcm.astype(float) ** 2))
                log(
                    "DECODE",
                    f"SUCCESS → {len(opus_data)} bytes → {BLOCK_SIZE} samples | "
                    f"RMS={rms:.2f} | VAD={avg_prob:.3f} [{state}]",
                )
            else:
                log("PACKET", f"Ignored duplicate/out-of-order seq={seq}")
        except Exception as e:
            log("ERROR", f"decode_packet failed: {e}")

    def process_current_segment(self):
        """Called when silence gap is detected (or at shutdown)."""
        if not self.audio_buffer:
            log("SEGMENT", "No audio buffer to process")
            return
        log("SEGMENT", f"Processing segment {self.segment_num:03d} ...")

        # Concatenate all chunks into one clean 1D array
        pcm_full = np.concatenate([arr.flatten() for arr in self.audio_buffer])

        # Compute speech probabilities & energies
        probs, energies = compute_speech_probs_and_energies(pcm_full, self.vad)
        metadata = detect_speech_boundaries(probs, energies)

        duration = metadata["duration_sec"]

        # === Record global timestamps so cleanup can know exactly which segments
        #     are still present in the rolling 5-min WAV ===
        global_start_sec = time.time() - self.session_start_time
        metadata["global_start_sec"] = round(global_start_sec, 3)
        metadata["global_end_sec"] = round(global_start_sec + duration, 3)

        # === FIX: Skip segments that don't have enough actual speech ===
        if duration < self.min_speech_sec:
            log(
                "SEGMENT",
                f"⏭️  Skipping short segment (duration={duration:.3f}s < {self.min_speech_sec}s)",
            )
            self.audio_buffer = []
            self.silence_counter = 0
            return

        # Rich logging with avg and duration info
        log(
            "SEGMENT",
            f"✅ Saving segment | avg_prob={metadata['average_prob']:.4f} "
            f"avg_rms={metadata['average_rms']:.2f} duration={duration:.3f}s",
        )

        # Create segment folder
        segment_dir = SEGMENTS_DIR / f"segment_{self.segment_num:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        save_segment_files(
            segment_dir,
            pcm_full,
            probs,
            energies,
            metadata,  # now already contains global_start_sec / global_end_sec
        )

        # Add to global segment list and update global files
        self.segments.append({**metadata, "segment_dir": segment_dir.name})
        save_global_files(self.segments, OUTPUT_DIR)
        save_rolling_audio(self.rolling_pcm, OUTPUT_DIR)  # update 5-min file

        self.segment_num += 1

    async def _monitor_silence(self):
        """Deprecated: segmentation is now VAD-driven."""
        while True:
            await asyncio.sleep(1)


# Global pipeline instance
pipeline = ReceiverPipeline()


async def websocket_receiver(websocket):
    # unchanged – just calls decode_packet
    peer = websocket.remote_address
    log("CONN", f"Client connected from {peer}")

    try:
        async for message in websocket:
            pipeline.decode_packet(message)
    except websockets.exceptions.ConnectionClosedOK:
        log("CONN", "Connection closed cleanly by sender")
    except websockets.exceptions.ConnectionClosedError as e:
        log("CONN", f"Connection closed with error: {e}")
    except Exception as e:
        log("ERROR", f"WebSocket handler error: {e}")
    finally:
        log("CONN", "WebSocket receiver handler ended")


async def main():
    log("START", "Starting TEN-VAD Receiver (segmentation mode) on port 8766")
    async with websockets.serve(websocket_receiver, "0.0.0.0", 8766):
        log("START", "✅ WebSocket server is running")
        log("INFO", "→ No audio playback – saving speech segments instead")

        # Start background silence monitor
        pipeline.monitor_task = asyncio.create_task(pipeline._monitor_silence())

        try:
            await asyncio.Future()  # run forever
        finally:
            # Graceful shutdown
            if pipeline.monitor_task:
                pipeline.monitor_task.cancel()
            log("SHUTDOWN", "Processing any remaining audio buffer...")
            save_rolling_audio(pipeline.rolling_pcm, OUTPUT_DIR)  # final save
            pipeline.process_current_segment()  # force-save last piece
            if pipeline.audio_buffer:
                pipeline.audio_buffer = []
            log(
                "SHUTDOWN",
                f"✅ Finished – {pipeline.segment_num} speech segments saved!",
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("SHUTDOWN", "Stopped by user")
    except Exception as e:
        log("ERROR", f"Unexpected error: {e}")
