from fastapi import FastAPI
from server.router import router
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("subtitle_server")

app = FastAPI(title="Live Subtitle Server")

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    log.info("[bold green]Live Subtitle Server started on http://0.0.0.0:8002[/bold green]")
    log.info("[bold cyan]WebSocket endpoint: ws://<your-ip>:8002/ws/subtitles[/bold cyan]")