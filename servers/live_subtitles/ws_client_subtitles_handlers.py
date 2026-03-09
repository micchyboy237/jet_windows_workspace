# ws_client_subtitles_handlers.py

import asyncio
import datetime
import json
import os

import websockets
from jet.audio.helpers.energy import rms_to_loudness_label
from jet.audio.speech.firered.speech_types import WordSegment

# from rich.logging import RichHandler
from jet.logger import logger as log
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from ws_client_subtitles_utils import find_segments_subdir

# Global SRT sequence counter (1-based)
srt_sequence = 1
latest_transcription_text: str = ""


class WSClientLiveSubtitleHandlers:
    def __init__(self, output_dir: str, overlay: LiveSubtitlesOverlay) -> None:
        self.output_dir = output_dir
        self.overlay = overlay

    # =============================
    # Subtitle SRT writing helpers
    # =============================

    def write_srt_block(
        self,
        sequence: int,
        start_sec: float,
        duration_sec: float,
        ja: str,
        en: str,
        file_path: str | os.PathLike,
    ) -> None:
        """Append one subtitle block to an SRT file."""
        start_dt = datetime.datetime.fromtimestamp(start_sec)
        end_dt = datetime.datetime.fromtimestamp(start_sec + duration_sec)

        start_str = (
            start_dt.strftime("%H:%M:%S") + f",{int(start_dt.microsecond / 1000):03d}"
        )
        end_str = end_dt.strftime("%H:%M:%S") + f",{int(end_dt.microsecond / 1000):03d}"

        block = f"{sequence}\n{start_str} --> {end_str}\n{ja.strip()}\n{en.strip()}\n\n"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(block)

        # log.info("[SRT] Appended block #%d to %s", sequence, os.path.basename(file_path))

    # =============================
    # Thin message receiver — receives WebSocket messages and dispatches to handler
    # =============================

    async def receive_messages(self, ws) -> None:
        """Thin receiver: recv → parse → dispatch by type"""
        async for msg in ws:
            try:
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "final_subtitle":
                    await self.handle_final_subtitle(data)

                elif msg_type == "speaker_update":
                    await self.handle_speaker_update(data)

                elif msg_type == "emotion_classification_update":
                    await self.handle_emotion_classification_update(data)

                elif msg_type == "partial_subtitle":
                    await self.handle_partial_subtitle(data)

                else:
                    log.warning("[ws] Ignoring unknown message type: %s", msg_type)

            except websockets.ConnectionClosed:
                log.info("[receive] WebSocket connection closed cleanly")
                break
            except json.JSONDecodeError:
                log.error("[receive] Invalid JSON received")
            except Exception as e:
                log.error("[receive] Error in receive loop: %s", e)
                await asyncio.sleep(0.3)  # prevent tight loop on repeated errors

    async def handle_partial_subtitle(self, data: dict) -> None:
        """Handle incremental/partial transcription updates from server."""
        log.debug("[partial_subtitle] Received partial update")

        utterance_id = data.get("utterance_id")
        if utterance_id is None:
            log.warning("[partial_subtitle] Missing utterance_id")
            return

        ja = data.get("transcription_ja", "").strip()
        en = data.get("translation_en", "").strip()
        if not (ja or en):
            return

        # Reuse most of the same logic as final, but mark as partial
        segment_idx = data.get("segment_idx")
        segment_num = data.get("segment_num")
        chunk_index = data.get("chunk_index")
        start_time = data.get("start_time")
        segment_type = data.get("segment_type", "speech")
        duration_sec = data.get("duration_sec", 0.0)
        avg_vad_conf = data.get("avg_vad_confidence", 0.0)
        rms = data.get("rms", 0.0)
        trans_conf = data.get("transcription_confidence")
        trans_quality = data.get("transcription_quality")
        transl_conf = data.get("translation_confidence")
        transl_quality = data.get("translation_quality")
        server_meta = data.get("meta", {})
        server_segments: list[WordSegment] = server_meta.get("segments", [])

        relative_start = 0.0
        relative_end = duration_sec

        # Show partial result (can be overwritten by next partial or final)
        self.overlay.add_message(
            message_id=f"{utterance_id}_chunk{chunk_index}",
            utterance_id=utterance_id,
            source_text=ja,
            translated_text=en,
            start_sec=relative_start,
            end_sec=relative_end,
            duration_sec=duration_sec,
            segment_number=segment_num,
            avg_vad_confidence=avg_vad_conf,
            rms=rms,
            rms_label=rms_to_loudness_label(rms),
            transcription_confidence=trans_conf if trans_conf is not None else None,
            transcription_quality=trans_quality,
            translation_confidence=transl_conf if transl_conf is not None else None,
            translation_quality=transl_quality,
            is_partial=True,
            chunk_index=chunk_index,
        )

        log.info(
            "[partial] utt %s | JA: %s",
            utterance_id,
            ja[:60] + "..." if len(ja) > 60 else ja,
        )
        if en:
            log.debug("[partial] EN: %s", en[:60] + "..." if len(en) > 60 else en)

    async def handle_final_subtitle(self, data: dict) -> None:
        log.info("[final_subtitle] Handling new final subtitle update...")
        global srt_sequence, latest_transcription_text

        utterance_id = data.get("utterance_id")
        if utterance_id is None:
            log.warning("[final_subtitle] Missing utterance_id")
            return

        ja = data.get("transcription_ja", "").strip()
        en = data.get("translation_en", "").strip()
        if not (ja or en):
            log.debug("[final_subtitle] Empty text received for utt %s", utterance_id)
            return

        latest_transcription_text = ja

        segment_idx = data["segment_idx"]
        segment_num = data["segment_num"]
        segment_type = data["segment_type"]
        start_time = data["start_time"]

        duration_sec = data.get("duration_sec", 0.0)
        avg_vad_conf = data.get("avg_vad_confidence")
        rms = data.get("rms")
        trans_conf = data.get("transcription_confidence")
        trans_quality = data.get("transcription_quality")
        transl_conf = data.get("translation_confidence")
        transl_quality = data.get("translation_quality")
        server_meta = data.get("meta", {})
        server_segments: list[WordSegment] = server_meta.get("segments", [])

        log.info("[final_subtitle] utt %s | JA: %s", utterance_id, ja[:80])
        if en:
            log.info("[final_subtitle] EN: %s", en[:80])
        log.info(
            "[quality] Transc: %.3f %s | Transl: %.3f %s",
            trans_conf or 0.0,
            trans_quality or "N/A",
            transl_conf or 0.0,
            transl_quality or "N/A",
        )

        # Reuse the global for now, but this could be per-instance in the future.
        relative_start = 0.0
        relative_end = duration_sec

        entry = {
            "ja": ja,
            "en": en,
            "duration_sec": duration_sec,
            "start_wallclock": start_time,
            "relative_start": relative_start,
            "relative_end": relative_end,
            "segment_idx": segment_idx,
            "segment_num": segment_num,
            "segment_type": segment_type,
            "avg_vad_conf": avg_vad_conf,
            "rms": rms,
            "trans_conf": trans_conf,
            "trans_quality": trans_quality,
            "transl_conf": transl_conf,
            "transl_quality": transl_quality,
            "server_meta": server_meta,
            "srt_written": False,
        }

        # Show text immediately (no speaker info yet)
        await self._update_display_and_files(utterance_id, entry)

    async def handle_speaker_update(self, data: dict) -> None:
        log.info("[speaker_update] Handling new speaker update...")
        utterance_id = data.get("utterance_id")
        if utterance_id is None:
            log.warning("[speaker_update] Missing utterance_id")
            return

        segment_idx = data["segment_idx"]
        segment_num = data["segment_num"]
        segment_type = data["segment_type"]

        # Use shared segment directory finding logic for consistency
        segment_dir = find_segments_subdir(
            segments_root=os.path.join(self.output_dir, "segments"),
            utterance_id=data.get("utterance_id"),
            chunk_index=data.get("chunk_index", 0),
            create_if_missing=False,
        )

        speaker_clusters = data.get("cluster_speakers", [])
        speaker_meta = {
            "segment_idx": segment_idx,
            "segment_num": segment_num,
            "segment_type": segment_type,
            "is_same_as_prev": data.get("is_same_speaker_as_prev"),
            "similarity_prev": data.get("similarity_prev"),
        }
        speaker_clusters_path = os.path.join(segment_dir, "speaker.json")
        speaker_meta_path = os.path.join(segment_dir, "speaker_meta.json")
        with open(speaker_clusters_path, "w", encoding="utf-8") as f:
            json.dump(speaker_clusters, f, indent=2, ensure_ascii=False)
        with open(speaker_meta_path, "w", encoding="utf-8") as f:
            json.dump(speaker_meta, f, indent=2, ensure_ascii=False)

    async def handle_emotion_classification_update(self, data: dict) -> None:
        log.info(
            "[emotion_classification_update] Handling new emotion classification update..."
        )
        utterance_id = data.get("utterance_id")
        if utterance_id is None:
            log.warning("[emotion_classification_update] Missing utterance_id")
            return

        segment_idx = data["segment_idx"]
        segment_num = data["segment_num"]
        segment_type = data["segment_type"]
        base_dir = "segments" if segment_type == "speech" else "segments_non_speech"
        segments_root = os.path.join(self.output_dir, base_dir)

        segment_dir = find_segments_subdir(
            segments_root=segments_root,
            utterance_id=data.get("utterance_id"),
            chunk_index=data.get("chunk_index", 0),
            create_if_missing=False,
        )

        emotion_classification_all = data.get("emotion_all")
        emotion_top_label = data.get("emotion_top_label")
        emotion_top_score = data.get("emotion_top_score")
        emotion_classification_meta = {
            "segment_idx": segment_idx,
            "segment_num": segment_num,
            "segment_type": segment_type,
            "emotion_top_label": emotion_top_label,
            "emotion_top_score": emotion_top_score,
        }

        emotion_classification_all_path = os.path.join(segment_dir, "emotion.json")
        emotion_classification_meta_path = os.path.join(
            segment_dir, "emotion_meta.json"
        )
        with open(emotion_classification_all_path, "w", encoding="utf-8") as f:
            json.dump(emotion_classification_all, f, indent=2, ensure_ascii=False)
        with open(emotion_classification_meta_path, "w", encoding="utf-8") as f:
            json.dump(emotion_classification_meta, f, indent=2, ensure_ascii=False)

    async def _update_display_and_files(self, utt_id: str | int, entry: dict) -> None:
        # Require text to proceed with display & SRT
        if "ja" not in entry:
            return

        ja = entry["ja"]
        en = entry["en"]
        duration_sec = entry["duration_sec"]
        relative_start = entry["relative_start"]
        relative_end = entry["relative_end"]
        is_final = entry.get("is_final", True)  # fallback to True
        segment_idx = entry["segment_idx"]
        segment_num = entry["segment_num"]
        chunk_index = entry.get("chunk_index", None)
        segment_type = entry["segment_type"]
        start_time = entry["start_wallclock"]

        avg_vad_conf = entry["avg_vad_conf"]
        rms = entry["rms"]

        trans_conf = entry.get("trans_conf")
        trans_quality = entry.get("trans_quality")
        transl_conf = entry.get("transl_conf")
        transl_quality = entry.get("transl_quality")

        subdir = "segments" if segment_type == "speech" else "segments_non_speech"
        segments_root = os.path.join(self.output_dir, subdir)

        segment_dir = find_segments_subdir(
            segments_root=segments_root,
            utterance_id=str(utt_id) if utt_id else None,
            chunk_index=entry.get("chunk_index", 0),
            create_if_missing=False,
        )

        self.overlay.add_message(
            message_id=f"{utt_id}_chunk{chunk_index}",
            utterance_id=utt_id,
            source_text=ja,
            translated_text=en,
            start_sec=relative_start,
            end_sec=relative_end,
            duration_sec=duration_sec,
            segment_number=segment_num,
            chunk_index=chunk_index,
            is_partial=not is_final,
            avg_vad_confidence=avg_vad_conf,
            rms=rms,
            rms_label=rms_to_loudness_label(rms),
            transcription_confidence=trans_conf if trans_conf is not None else None,
            transcription_quality=trans_quality,
            translation_confidence=transl_conf if transl_conf is not None else None,
            translation_quality=transl_quality,
        )

        per_seg_srt = os.path.join(segment_dir, "subtitles.srt")
        all_srt_path = os.path.join(self.output_dir, "all_subtitles.srt")

        if not entry.get("srt_written", False):
            global srt_sequence
            self.write_srt_block(
                srt_sequence, start_time, duration_sec, ja, en, per_seg_srt
            )
            self.write_srt_block(
                srt_sequence, start_time, duration_sec, ja, en, all_srt_path
            )
            srt_sequence += 1
            entry["srt_written"] = True
