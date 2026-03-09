"""
Modern live subtitles server using per-utterance LLM transcription + translation.
Compatible with newer client message format (speech_chunk / complete_utterance).
"""

import asyncio
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rich.logging import RichHandler

# Centralized logger import (assume logger.py in project root)
from logger import logger

executor_fast = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FastProc")
executor_slow = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SlowProc")

DEFAULT_OUT_DIR: Path | None = None

from processing.fast_processor import process_fast_llm
from processing.slow_processor import process_slow
from ws_server_subtitles_handlers import handler, connected_states

async def main():
    from websockets.asyncio.server import serve

    async with serve(
        handler,
        host="0.0.0.0",
        port=8765,
        ping_interval=20,
        ping_timeout=60,
    ) as server:
        logger.info("WebSocket server listening → ws://0.0.0.0:8765")
        await server.serve_forever()

if __name__ == "__main__":
    out_dir_str = os.getenv("UTTERANCE_OUT_DIR")
    if out_dir_str:
        out_dir_path = Path(out_dir_str).resolve()
        shutil.rmtree(out_dir_path, ignore_errors=True)        # clean on start (common in dev)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUT_DIR = out_dir_path
        logger.info(f"Permanent utterance storage enabled: {DEFAULT_OUT_DIR}")
    else:
        DEFAULT_OUT_DIR = None
        logger.info("Using temporary files (set UTTERANCE_OUT_DIR for permanent storage)")

    try:
        logger.info("Live subtitles server (LLM mode) starting...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception("Fatal startup error")
        raise