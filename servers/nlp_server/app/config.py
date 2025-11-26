# app/config.py
from typing import Literal
from pydantic_settings import BaseSettings  # ← Use this
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

class Settings(BaseSettings):
    language: str = Field("en", description="Language code for Stanza pipeline")
    processors: str = Field("tokenize,mwt,pos,lemma,ner,depparse",
                            description="Comma-separated Stanza processors")
    model_dir: str = Field("stanza_models", description="Directory where Stanza models are stored")
    use_gpu: bool = Field(False, description="Whether to use GPU for Stanza pipeline")
    host: str = Field("0.0.0.0", description="Host to serve API")
    port: int = Field(8000, description="Port for API")
    batch_size: int = Field(1, description="Batch size for processing multiple documents")

    class Config:
        env_prefix = "NLP_"

settings = Settings()