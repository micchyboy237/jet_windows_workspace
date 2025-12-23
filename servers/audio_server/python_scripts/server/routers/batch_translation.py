# router.py
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from python_scripts.server.services.translator.batch_translation_service import TranslationRequest, stream_batch_translation

router = APIRouter(prefix="/translate", tags=["translation"])


def _sse_generator(stream: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    """Wraps string generator to bytes and adds required SSE headers."""
    async def event_stream() -> AsyncGenerator[bytes, None]:
        async for chunk in stream:
            yield chunk.encode("utf-8")
    return event_stream()


@router.post("/batch", response_class=StreamingResponse)
async def batch_translate(request: Request, body: TranslationRequest):
    """
    Endpoint for batch Japanese → English translation with token streaming.
    
    Client should connect using EventSource (SSE) to receive real-time partial and final results.
    
    Response format (SSE):
    - partial updates: {"partial": "token chunk", "sentence": "original"}
    - final per sentence: {"done": "full translation", "sentence": "original"}
    - error: {"error": "message"}
    """
    # Allow client to disconnect gracefully
    async def wrapped_stream() -> AsyncGenerator[str, None]:
        try:
            async for item in stream_batch_translation(body):
                if await request.is_disconnected():
                    break
                yield item
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _sse_generator(wrapped_stream()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx)
        },
    )