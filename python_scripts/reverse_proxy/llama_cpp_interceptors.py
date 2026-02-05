import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from rich.logging import RichHandler
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
import uvicorn

import logging
import uuid
import ftfy

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
logger = logging.getLogger("llama_proxy")
console = Console()


def create_app(
    upstream_url: str,
    log_dir: Union[Path, str],
) -> FastAPI:
    app = FastAPI(
        title="LLaMA.cpp Reverse Proxy (OpenAI client)",
        description="Thin proxy with rich logging & request tracing using official OpenAI SDK",
        version="0.4.0",
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    upstream_base = upstream_url.rstrip("/")

    # Single shared async client — base_url points to llama.cpp server
    client = AsyncOpenAI(
        base_url=f"{upstream_base}/v1",  # most common prefix — adjust if needed
        api_key="sk-no-key-required",     # llama.cpp usually ignores / accepts dummy key
        timeout=None,                     # no client-level timeout
    )

    @app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
    @app.api_route("/", methods=["GET", "POST", "OPTIONS"])
    async def proxy(request: Request, path: str = ""):
        request_id = str(uuid.uuid4())[:8]

        path_with_query = request.url.path
        if request.url.query:
            path_with_query += "?" + request.url.query

        logger.info(
            f"[bold green]→[/] {request.method} {path_with_query} • [dim]id={request_id}[/]"
        )

        # Only handle chat completions streaming for now (most common use-case)
        # You can extend later for /models, /completions, etc.
        if request.method != "POST" or "chat/completions" not in path.lower():
            return Response("Only chat/completions endpoint is proxied via OpenAI client", 501)

        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return Response("Invalid JSON", 400)

        # Remove 'stream' from payload if present — we control it explicitly
        stream_requested = payload.pop("stream", False)

        # Prepare forwarded / tracing headers
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            forwarded_for += ", "
        forwarded_for += request.client.host or "unknown"

        extra_headers: Dict[str, str] = {
            "X-Forwarded-For": forwarded_for,
            "X-Forwarded-Proto": request.headers.get("x-forwarded-proto", "http"),
            "X-Forwarded-Host": request.headers.get("host", request.url.hostname),
            "X-Request-ID": request_id,
        }

        # Optional: pass any other client-provided headers
        for k, v in request.headers.items():
            if k.lower() not in {
                "host", "content-length", "transfer-encoding",
                "connection", "keep-alive", "upgrade"
            }:
                extra_headers[k] = v

        start = datetime.now(timezone.utc)
        duration_ms: float = 0
        usage_info: Optional[Dict[str, Any]] = None
        full_content: str = ""

        try:
            if stream_requested:
                stream = await client.chat.completions.create(
                    **payload,
                    stream=True,
                    extra_headers=extra_headers,  # type: ignore
                )

                duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                logger.info(
                    f"[bold cyan]←[/] streaming started • {duration_ms:.0f} ms   id={request_id}"
                )

                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S%z")
                log_file = log_dir / f"{timestamp}_{request_id}.json"

                async def stream_and_log() -> AsyncGenerator[bytes, None]:
                    chunks: list[bytes] = []
                    bytes_received = 0
                    last_preview_time = datetime.now(timezone.utc)
                    preview_throttle_sec = 1.5

                    full_content_pieces = []
                    usage_info_inner: Optional[Dict[str, Any]] = None

                    try:
                        async for chunk in stream:
                            # chunk is ChatCompletionChunk object

                            # Guard: skip if no choices (common in final usage chunk or server quirks)
                            if not getattr(chunk, "choices", None):
                                # Possible usage chunk (last one)
                                if hasattr(chunk, "usage") and chunk.usage is not None:
                                    usage_info_inner = chunk.usage.model_dump()
                                    logger.info(
                                        f"[dim]usage captured[/] prompt={usage_info_inner.get('prompt_tokens')} "
                                        f"completion={usage_info_inner.get('completion_tokens')} "
                                        f"total={usage_info_inner.get('total_tokens')} • id={request_id}"
                                    )
                                # Still forward the chunk as SSE
                                chunk_data = chunk.model_dump_json(exclude_none=True)
                                sse_line = f"data: {chunk_data}\n\n"
                                sse_bytes = sse_line.encode("utf-8")
                                bytes_received += len(sse_bytes)
                                chunks.append(sse_bytes)
                                yield sse_bytes
                                logger.debug("[empty choices chunk forwarded] id=%s", request_id)
                                continue

                            delta = chunk.choices[0].delta

                            # Collect content for reconstruction / final logging
                            if getattr(delta, "content", None) is not None:
                                # Fix potential mojibake / encoding glitches from upstream
                                fixed_content = ftfy.fix_text(delta.content)
                                full_content_pieces.append(fixed_content)

                            # Reconstruct SSE (same as before)
                            chunk_data = chunk.model_dump_json(exclude_none=True)
                            sse_line = f"data: {chunk_data}\n\n"
                            sse_bytes = sse_line.encode("utf-8")

                            bytes_received += len(sse_bytes)
                            chunks.append(sse_bytes)

                            logger.debug("[chunk] %d bytes • id=%s", len(sse_bytes), request_id)

                            # Throttled preview — only if there's content
                            now = datetime.now(timezone.utc)
                            if (now - last_preview_time).total_seconds() >= preview_throttle_sec:
                                if getattr(delta, "content", None):
                                    preview = ftfy.fix_text(delta.content.strip())[:120]
                                    if len(preview) == 120:
                                        preview += "…"
                                    if preview.strip():
                                        logger.info(
                                            f"[dim]chunk preview[/] {preview} "
                                            f"[dim]({bytes_received:,} bytes so far)[/]"
                                        )
                                last_preview_time = now

                            yield sse_bytes

                        # After stream ends, send final [DONE]
                        done = b"data: [DONE]\n\n"
                        yield done
                        chunks.append(done)
                        bytes_received += len(done)

                    except Exception as exc:
                        logger.exception("Streaming error", extra={"request_id": request_id})
                        raise

                    finally:
                        full_body = b"".join(chunks)
                        # Reconstruct approximate response body for logging
                        response_text = {
                            "content": "".join(full_content_pieces),
                            "usage": usage_info_inner,  # will be None if not present
                            "_note": "reconstructed from chunks — full raw SSE in chunks log",
                        }

                        req_json = payload

                        record = {
                            "request_id": request_id,
                            "timestamp": timestamp,
                            "method": request.method,
                            "path": path_with_query,
                            "client_ip": request.client.host,
                            "upstream_url": str(client.base_url),
                            "request_headers": extra_headers,
                            "request_body": req_json,
                            "status_code": 200,               # streaming always 200
                            "response_headers": {},           # we don't have them here
                            "response_body": response_text,
                            "duration_ms": round(duration_ms, 1),
                            "bytes_received": bytes_received,
                            "was_streaming": True,
                        }

                        try:
                            log_file.write_text(
                                json.dumps(record, indent=2, ensure_ascii=False),
                                encoding="utf-8",  # Force UTF-8 on Windows to support emojis & full Unicode
                            )
                        except Exception as exc:
                            logger.error(f"Failed to write log {log_file}", exc_info=exc)

                        # ─── Final rich console output ─────────────────────────────
                        table = Table(title=f"LLM Call  [dim]{request_id}[/]", show_header=False)
                        table.add_row("[bold]Request[/]", f"{request.method} {path_with_query}")
                        table.add_row("[bold]Client[/]", request.client.host or "?")
                        table.add_row("[bold]Upstream[/]", str(client.base_url))
                        table.add_row("[bold]Status[/]", "200 (streaming)")
                        table.add_row("[bold]Duration[/]", f"{duration_ms:.0f} ms")
                        table.add_row("[bold]Bytes[/]", f"{bytes_received:,}")
                        if usage_info_inner:
                            table.add_row("[bold]Usage[/]", f"prompt={usage_info_inner.get('prompt_tokens')} | "
                                         f"completion={usage_info_inner.get('completion_tokens')} | "
                                         f"total={usage_info_inner.get('total_tokens')}")

                        console.rule(f"[bold cyan]LLM Call • {request_id}", style="cyan")
                        console.print(table)

                        if req_json:
                            console.print("[bold green]→ Request body[/]")
                            console.print(Syntax(json.dumps(req_json, indent=2), "json", word_wrap=True))

                        console.print("[bold blue]← Response content (reconstructed & fixed)[/]")
                        preview_text = response_text["content"][:800]
                        if len(response_text["content"]) > 800:
                            preview_text += " … [truncated in console – full in log file]"
                        console.print(preview_text)

                return StreamingResponse(
                    stream_and_log(),
                    status_code=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",  # important for nginx
                    },
                    media_type="text/event-stream",
                )

            else:
                # ── Non-streaming path ──
                response: ChatCompletion = await client.chat.completions.create(
                    **payload,
                    stream=False,
                    extra_headers=extra_headers,  # type: ignore
                )

                duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                logger.info(
                    f"[bold cyan]←[/] completed in {duration_ms:.0f} ms   id={request_id}"
                )

                if response.usage:
                    usage_info = response.usage.model_dump()

                # For non-streaming we can get content directly
                if response.choices and response.choices[0].message.content is not None:
                    full_content = ftfy.fix_text(response.choices[0].message.content)

                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S%z")
                log_file = log_dir / f"{timestamp}_{request_id}.json"

                record = {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "method": request.method,
                    "path": path_with_query,
                    "client_ip": request.client.host,
                    "upstream_url": str(client.base_url),
                    "request_headers": extra_headers,
                    "request_body": payload,
                    "status_code": 200,
                    "response_headers": {},
                    "response_body": {
                        "content": full_content,
                        "usage": usage_info,
                    },
                    "duration_ms": round(duration_ms, 1),
                    "bytes_received": len(json.dumps(response.model_dump()).encode()),
                    "was_streaming": False,
                }

                try:
                    log_file.write_text(
                        json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.error(f"Failed to write log {log_file}", exc_info=exc)

                # ── Rich console output for non-streaming ──
                table = Table(title=f"LLM Call  [dim]{request_id}[/]", show_header=False)
                table.add_row("[bold]Request[/]", f"{request.method} {path_with_query}")
                table.add_row("[bold]Client[/]", request.client.host or "?")
                table.add_row("[bold]Upstream[/]", str(client.base_url))
                table.add_row("[bold]Status[/]", "200")
                table.add_row("[bold]Duration[/]", f"{duration_ms:.0f} ms")
                table.add_row("[bold]Bytes[/]", f"{len(json.dumps(response.model_dump()).encode()):,}")
                if usage_info:
                    table.add_row("[bold]Usage[/]", f"prompt={usage_info.get('prompt_tokens')} | "
                                 f"completion={usage_info.get('completion_tokens')} | "
                                 f"total={usage_info.get('total_tokens')}")

                console.rule(f"[bold cyan]LLM Call • {request_id}", style="cyan")
                console.print(table)

                if payload:
                    console.print("[bold green]→ Request body[/]")
                    console.print(Syntax(json.dumps(payload, indent=2), "json", word_wrap=True))

                console.print("[bold blue]← Response content[/]")
                preview_text = full_content[:800]
                if len(full_content) > 800:
                    preview_text += " … [truncated in console]"
                console.print(preview_text)

                # Return the original OpenAI-compatible response
                return JSONResponse(
                    content=response.model_dump(),
                    status_code=200,
                )

        except Exception as exc:
            logger.exception(f"OpenAI client failed • id={request_id}")
            return Response(f"Upstream error: {exc}", 502)

        # (keep your existing stream_and_log() function)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse proxy for LLaMA servers with rich logging (OpenAI SDK version)",
    )

    parser.add_argument(
        "--upstream-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the upstream LLaMA server",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the proxy server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the proxy (default 8080)",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(r"C:\Users\druiv\.cache\logs\llama.cpp\interceptors\logs"),
        help="Directory to store request/response logs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = create_app(
        upstream_url=args.upstream_url,
        log_dir=args.log_dir,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
