import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import traceback
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
    log_dir: Path | str,
) -> FastAPI:
    app = FastAPI(
        title="LLaMA.cpp Reverse Proxy (OpenAI client)",
        description="Thin proxy with rich logging & request tracing using official OpenAI SDK",
        version="0.5.0",
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir.resolve()  # make sure we have absolute path
    upstream_base = upstream_url.rstrip("/")

    # Shared async client pointing to llama.cpp server
    client = AsyncOpenAI(
        base_url=f"{upstream_base}/v1",
        api_key="sk-no-key-required",
        timeout=None,
        max_retries=0,
    )

    async def log_success(
        request_id: str,
        timestamp: str,
        request: Request,
        path_with_query: str,
        payload: dict,
        duration_ms: float,
        usage_info: Optional[Dict[str, Any]],
        content: str,
        bytes_transferred: int,
        was_streaming: bool,
        log_file: Path,
    ) -> None:
        """Shared logging logic for both streaming and non-streaming (success)."""
        record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "method": request.method,
            "path": path_with_query,
            "client_ip": request.client.host,
            "upstream_url": str(client.base_url),
            "request_headers": dict(request.headers),
            "request_body": payload,
            "status_code": 200,
            "response_headers": {},
            "response_body": {
                "content": content,
                "usage": usage_info,
            },
            "duration_ms": round(duration_ms, 1),
            "bytes_received": bytes_transferred,
            "was_streaming": was_streaming,
        }

        try:
            log_file.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(f"Failed to write log {log_file}", exc_info=exc)

        # Console output
        table = Table(title=f"LLM Call  [dim]{request_id}[/]", show_header=False)
        table.add_row("[bold]Request[/]", f"{request.method} {path_with_query}")
        table.add_row("[bold]Client[/]", request.client.host or "?")
        table.add_row("[bold]Upstream[/]", str(client.base_url))
        table.add_row("[bold]Status[/]", "200")
        table.add_row("[bold]Duration[/]", f"{duration_ms:.0f} ms")
        table.add_row("[bold]Bytes[/]", f"{bytes_transferred:,}")
        if usage_info:
            table.add_row(
                "[bold]Usage[/]",
                f"prompt={usage_info.get('prompt_tokens')} | "
                f"completion={usage_info.get('completion_tokens')} | "
                f"total={usage_info.get('total_tokens')}",
            )

        console.rule(f"[bold cyan]LLM Call • {request_id}", style="cyan")
        console.print(table)

        if payload:
            console.print("[bold green]→ Request body[/]")
            console.print(Syntax(json.dumps(payload, indent=2), "json", word_wrap=True))

        console.print("[bold blue]← Response content[/]")
        preview_text = content[:800]
        if len(content) > 800:
            preview_text += " … [truncated in console – full in log file]"
        console.print(preview_text)

    async def log_error(
        request_id: str,
        timestamp: str,
        request: Request,
        path_with_query: str,
        payload: dict,
        exc: Exception,
        daily_dir: Path,
    ) -> Path:
        error_file = daily_dir / f"ERROR_{timestamp}_{request_id}.json"

        record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "method": request.method,
            "path": path_with_query,
            "client_ip": request.client.host,
            "upstream_url": str(client.base_url),
            "request_headers": dict(request.headers),
            "request_body": payload,
            "status_code": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                # Fragile & broken — replace with standard traceback
                "traceback": traceback.format_exc() if hasattr(exc, '__traceback__') else None,
            },
            "success": False,
        }

        try:
            error_file.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as write_exc:
            logger.error(f"Failed to write error log {error_file}", exc_info=write_exc)

        logger.exception(f"Request failed • id={request_id}")
        return error_file

    @app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
    @app.api_route("/", methods=["GET", "POST", "OPTIONS"])
    async def proxy(request: Request, path: str = ""):
        request_id = str(uuid.uuid4())[:8]
        path_with_query = str(request.url.path)
        if request.url.query:
            path_with_query += "?" + request.url.query

        logger.info(
            f"[bold green]→[/] {request.method} {path_with_query} • [dim]id={request_id}[/]"
        )

        # Allow both /chat/completions and /embeddings endpoints
        if request.method != "POST" or not any(x in path.lower() for x in ["chat/completions", "embeddings"]):
            return Response("Only chat/completions and embeddings endpoints are proxied", 501)

        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return Response("Invalid JSON", 400)

        is_embeddings = "embeddings" in path.lower()
        stream_requested = payload.pop("stream", False) if not is_embeddings else False

        # Prepare forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for", request.client.host or "unknown")
        if forwarded_for:
            forwarded_for += ", "
        forwarded_for += request.client.host or "unknown"

        # Remove duplicate forwarded_for logic
        extra_headers = {
            "X-Forwarded-For": forwarded_for,
            "X-Forwarded-Proto": request.headers.get("x-forwarded-proto", "http"),
            "X-Forwarded-Host": request.headers.get("host", request.url.hostname),
            "X-Request-ID": request_id,
        }

        for k, v in request.headers.items():
            if k.lower() not in {
                "host", "content-length", "transfer-encoding",
                "connection", "keep-alive", "upgrade"
            }:
                extra_headers[k] = v

        start = datetime.now(timezone.utc)
        timestamp = start.strftime("%Y-%m-%dT%H-%M-%S%z")

        date_str = start.strftime("%Y-%m-%d")
        daily_dir = log_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        success_log_file = daily_dir / f"{timestamp}_{request_id}.json"

        if is_embeddings:
            # ── Embeddings path ─────────────────────────────────────────────
            # (embeddings block unchanged except log file & error logging)
            try:
                response = await client.embeddings.create(
                    **payload,
                    # embeddings do not support stream=True in OpenAI spec
                )

                duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                logger.info(
                    f"[bold cyan]←[/] embeddings completed in {duration_ms:.0f} ms   id={request_id}"
                )  

                # For logging: take first embedding or summary
                content_summary = ""
                if response.data and response.data[0].embedding:
                    emb = response.data[0].embedding
                    content_summary = f"embedding vector (dim={len(emb)}) • first 8: {emb[:8]}"

                bytes_transferred = len(json.dumps(response.model_dump()).encode("utf-8"))

                await log_success(
                    request_id=request_id,
                    timestamp=timestamp,
                    request=request,
                    path_with_query=path_with_query,
                    payload=payload,
                    duration_ms=duration_ms,
                    usage_info=None,  # embeddings usually have usage
                    content=content_summary,
                    bytes_transferred=bytes_transferred, 
                    was_streaming=False,
                    log_file=success_log_file,
                )

                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    status_code=200,
                )

            except Exception as exc:
                await log_error(
                    request_id, timestamp, request, path_with_query, payload, exc, daily_dir
                )
                raise
                # Alternative: return Response(f"Upstream error: {exc}", 502)

        try:
            if stream_requested:
                # ── Streaming chat path ── (existing)
                stream = await client.chat.completions.create(
                    **payload,
                    stream=True,
                    extra_headers=extra_headers,
                )

                # duration measured later for streaming
                logger.info(
                    f"[bold cyan]←[/] streaming started   id={request_id}"
                )

                full_content_pieces: list[str] = []
                usage_info: Optional[Dict[str, Any]] = None
                chunks: list[bytes] = []
                bytes_received = 0

                async def stream_and_log() -> AsyncGenerator[bytes, None]:
                    nonlocal bytes_received, usage_info
                    last_preview_time = datetime.now(timezone.utc)
                    preview_throttle_sec = 1.5

                    try:
                        async for chunk in stream:
                            if not getattr(chunk, "choices", None):
                                if hasattr(chunk, "usage") and chunk.usage:
                                    usage_info = chunk.usage.model_dump()
                                    logger.info(
                                        f"[dim]usage captured[/] prompt={usage_info.get('prompt_tokens')} "
                                        f"completion={usage_info.get('completion_tokens')} "
                                        f"total={usage_info.get('total_tokens')} • id={request_id}"
                                    )
                                chunk_data = chunk.model_dump_json(exclude_none=True)
                                sse_line = f"data: {chunk_data}\n\n"
                                sse_bytes = sse_line.encode("utf-8")
                                bytes_received += len(sse_bytes)
                                chunks.append(sse_bytes)
                                yield sse_bytes
                                continue

                            delta = chunk.choices[0].delta
                            if delta.content is not None:
                                fixed = ftfy.fix_text(delta.content)
                                full_content_pieces.append(fixed)

                            chunk_data = chunk.model_dump_json(exclude_none=True)
                            sse_line = f"data: {chunk_data}\n\n"
                            sse_bytes = sse_line.encode("utf-8")
                            bytes_received += len(sse_bytes)
                            chunks.append(sse_bytes)

                            now = datetime.now(timezone.utc)
                            if (now - last_preview_time).total_seconds() >= preview_throttle_sec:
                                if delta.content:
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

                        done = b"data: [DONE]\n\n"
                        yield done
                        chunks.append(done)
                        bytes_received += len(done)

                    except Exception as exc:
                        logger.exception("Streaming error", extra={"request_id": request_id})
                        await log_error(
                            request_id, timestamp, request, path_with_query, payload, exc, daily_dir
                        )
                        raise

                    # finally block already logs success
                    finally:
                        full_content = "".join(full_content_pieces)
                        end = datetime.now(timezone.utc)
                        duration_ms = (end - start).total_seconds() * 1000
                        await log_success(
                            request_id=request_id,
                            timestamp=timestamp,
                            request=request,
                            path_with_query=path_with_query,
                            payload=payload,
                            duration_ms=duration_ms,
                            usage_info=usage_info,
                            content=full_content,
                            bytes_transferred=bytes_received, 
                            was_streaming=True,
                            log_file=success_log_file,
                        )

                return StreamingResponse(
                    stream_and_log(),
                    status_code=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                    media_type="text/event-stream",
                )

            else:
                # ── Non-streaming chat path ──────────────────────────────────────
                response: ChatCompletion = await client.chat.completions.create(
                    **payload,
                    stream=False,
                    extra_headers=extra_headers,
                )

                duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                logger.info(
                    f"[bold cyan]←[/] completed in {duration_ms:.0f} ms   id={request_id}"
                )

                usage_info = response.usage.model_dump() if response.usage else None
                content = ""
                if response.choices and response.choices[0].message.content is not None:
                    content = ftfy.fix_text(response.choices[0].message.content)

                # Approximate bytes for logging
                bytes_transferred = len(json.dumps(response.model_dump()).encode("utf-8"))

                await log_success(
                    request_id=request_id,
                    timestamp=timestamp,
                    request=request,
                    path_with_query=path_with_query,
                    payload=payload,
                    duration_ms=duration_ms,
                    usage_info=usage_info,
                    content=content,
                    bytes_transferred=bytes_transferred,
                    was_streaming=False,
                    log_file=success_log_file,
                )

                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    status_code=200,
                )

        except Exception as exc:  # outer catch-all
            await log_error(
                request_id, timestamp, request, path_with_query, payload, exc, daily_dir
            )
            raise
            # or: return Response(f"Upstream error: {exc}", 502)

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
        default=Path.home() / ".cache" / "logs" / "llama.cpp" / "proxy",
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
