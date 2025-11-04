# main.py
import uvicorn
from fastapi import FastAPI
from app.routes import router
from app.config import settings

def create_app() -> FastAPI:
    app = FastAPI(title="Stanza NLP Server", version="1.0")
    app.include_router(router, prefix="/api/v1")
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app",
                host=settings.host,
                port=settings.port,
                log_level="info",
                workers=1)  # increase workers for production if desired