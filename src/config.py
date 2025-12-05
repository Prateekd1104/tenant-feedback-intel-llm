from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseModel):
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "llama-3-8b-instruct")
    chroma_dir: str = os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db"))
    project_name: str = "Tenant Feedback Intelligence Platform"


settings = Settings()
