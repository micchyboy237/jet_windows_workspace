import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import traceback
import shutil
from typing import Dict, Any

from openai import AsyncOpenAI
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from rich.logging import RichHandler
from rich.console import Console
import uvicorn

import logging
import uuid

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
        title="LLaMA.cpp Transparent Reverse Proxy",
        description="Forwards all OpenAI-compatible requests with minimal interference + rich request logging",
        version="0.6.0",
    )

    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    upstream_base = upstream_url.rstrip("/")

    # Official OpenAI client (used mostly for base_url + internal httpx client)
    openai_client = AsyncOpenAI(
        base_url=f"{upstream_base}/v1",
        api_key="sk-no-key-required",
        timeout=None,
        max_retries=0,
    )

    # Low-level httpx client for transparent forwarding
    http_client = openai_client._client

    def build_upstream_url(path: str, query: str | None = None) -> str:
        """Build correct upstream URL without duplicating /v1"""
        clean_path = path.lstrip("/")
        # If incoming path starts with v1/ → strip it (since base_url already has /v1)
        if clean_path.startswith("v1/"):
            clean_path = clean_path[3:]
        url = f"{upstream_base}/{clean_path.lstrip('/')}"
        if query:
            url += "?" + query
        return url

    async def write_log_background(
        log_file: Path,
        record: Dict[str, Any],
    ) -> None:
        """Non-blocking log writer – fire and forget"""
        try:
            log_file.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Background log write failed: {e}")

    # ────────────────────────────────────────────────
    #           NEW: Log cleanup logic
    # ────────────────────────────────────────────────

    async def cleanup_old_logs(max_age_days: int = 14):
        """Delete daily log folders older than max_age_days"""
        if max_age_days < 1:
            return
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        for item in list(log_dir.iterdir()):
            if not item.is_dir():
                continue
            if not item.name.startswith("20") or len(item.name) != 10:
                continue  # skip non-date-looking folders
            try:
                folder_date = datetime.strptime(item.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if folder_date < cutoff_date:
                    logger.info(
                        f"Cleaning up old logs: removing {item} (older than {max_age_days} days)"
                    )
                    shutil.rmtree(item, ignore_errors=True)
            except ValueError:
                continue  # not a valid date folder
            except Exception as e:
                logger.warning(f"Failed to clean directory {item}: {e}")

    # Periodic cleanup task
    async def periodic_log_cleanup():
        while True:
            try:
                await cleanup_old_logs(max_age_days=14)  # ← you can change this number
            except Exception as e:
                logger.error(f"Log cleanup task failed: {e}")
            await asyncio.sleep(3600)  # every 1 hour

    # ────────────────────────────────────────────────
    #           FastAPI lifecycle hooks
    # ────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup_event():
        # Clean old logs once when the server starts
        await cleanup_old_logs(max_age_days=14)
        # Start periodic cleanup in background
        asyncio.create_task(periodic_log_cleanup())
        logger.info("Log cleanup tasks started (keeping last 14 days)")

    # ────────────────────────────────────────────────
    #           Your existing proxy endpoint
    # ────────────────────────────────────────────────

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    @app.api_route(
        "/", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
    )
    async def proxy_all(request: Request, path: str = ""):
        request_id = str(uuid.uuid4())[:8]
        path_with_query = str(request.url.path)
        if request.url.query:
            path_with_query += "?" + request.url.query

        logger.info(
            f"[bold green]→[/] {request.method} {path_with_query} • [dim]id={request_id}[/]"
        )

        # Prepare forwarded headers (remove hop-by-hop)
        headers_to_forward: Dict[str, str] = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in {
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "keep-alive",
                "upgrade",
                "te",
            }
        }

        # Forwarding chain info
        forwarded_for = request.headers.get(
            "x-forwarded-for", request.client.host or "unknown"
        )
        if forwarded_for:
            forwarded_for += ", "
        forwarded_for += request.client.host or "unknown"

        headers_to_forward["X-Forwarded-For"] = forwarded_for
        headers_to_forward["X-Forwarded-Proto"] = request.headers.get(
            "x-forwarded-proto", "http"
        )
        headers_to_forward["X-Forwarded-Host"] = request.headers.get(
            "host", request.url.hostname
        )
        headers_to_forward["X-Request-ID"] = request_id

        start = datetime.now(timezone.utc)
        timestamp = start.strftime("%Y-%m-%dT%H-%M-%S%z")
        date_str = start.strftime("%Y-%m-%d")
        daily_dir = log_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        log_file = daily_dir / f"{timestamp}_{request_id}.json"

        try:
            target_url = build_upstream_url(path, request.url.query)

            upstream_resp = await http_client.request(
                method=request.method,
                url=target_url,
                headers=headers_to_forward,
                content=await request.body(),
                timeout=None,
                follow_redirects=True,
            )

            duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            # Minimal structured log
            record = {
                "request_id": request_id,
                "timestamp": timestamp,
                "method": request.method,
                "path": path_with_query,
                "client_ip": request.client.host,
                "upstream_url": target_url,
                "status_code": upstream_resp.status_code,
                "duration_ms": round(duration_ms, 1),
                "forwarded_headers": {
                    k: v
                    for k, v in headers_to_forward.items()
                    if k.lower().startswith("x-")
                },
            }

            # Non-blocking log write
            asyncio.create_task(write_log_background(log_file, record))

            logger.info(
                f"[bold cyan]←[/] {upstream_resp.status_code} "
                f"{duration_ms:.0f} ms   [dim]id={request_id}[/]"
            )

            # Forward response — streaming if applicable
            content_type = upstream_resp.headers.get("content-type", "")
            is_stream = (
                "text/event-stream" in content_type.lower()
                or "application/x-ndjson" in content_type.lower()
            )

            if is_stream:
                return StreamingResponse(
                    upstream_resp.aiter_bytes(),
                    status_code=upstream_resp.status_code,
                    headers=dict(upstream_resp.headers),
                    media_type=content_type,
                )
            else:
                content = await upstream_resp.aread()
                return Response(
                    content=content,
                    status_code=upstream_resp.status_code,
                    headers=dict(upstream_resp.headers),
                    media_type=upstream_resp.headers.get("content-type"),
                )

        except Exception as exc:
            duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            error_record = {
                "request_id": request_id,
                "timestamp": timestamp,
                "method": request.method,
                "path": path_with_query,
                "client_ip": request.client.host,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "duration_ms": round(duration_ms, 1),
                "success": False,
            }

            asyncio.create_task(write_log_background(log_file, error_record))
            logger.exception(f"Proxy error • id={request_id}")

            return Response(
                content=f"Upstream error: {str(exc)}",
                status_code=502,
            )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transparent reverse proxy for llama.cpp / OpenAI-compatible servers with rich logging",
    )
    parser.add_argument(
        "--upstream-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the upstream llama.cpp server (without /v1)",
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
        default=Path.home() / ".cache" / "logs" / "llama_proxy",
        help="Directory to store request logs",
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
