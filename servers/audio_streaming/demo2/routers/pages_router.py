import logging
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

logger = logging.getLogger(__name__)
router = APIRouter()

# Custom Jinja2 environment
env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html', 'xml']),
    extensions=['jinja2.ext.debug', 'jinja2.ext.loopcontrols'],
    trim_blocks=True,
    lstrip_blocks=True,
)

templates = Jinja2Templates(env=env)

@router.get("/")
async def root_redirect(request: Request):
    """Redirect root to partial range demo."""
    logger.info(f"Redirecting root request from {request.client.host} to /partial")
    return RedirectResponse(url="/partial", status_code=302)

@router.get("/partial")
async def serve_partial_demo(request: Request):
    """Serves the partial-range progressive streaming demo."""
    logger.info(f"Serving partial range demo to {request.client.host}")
    return templates.TemplateResponse(
        "partial.jinja",
        {
            "request": request,
            "current_page": "partial"
        }
    )

@router.get("/full")
async def serve_full_demo(request: Request):
    """Serves the full-range Web Audio API demo."""
    logger.info(f"Serving full range demo to {request.client.host}")
    return templates.TemplateResponse(
        "full.jinja",
        {
            "request": request,
            "current_page": "full"
        }
    )
