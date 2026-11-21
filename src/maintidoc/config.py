"""Configuration centralisée (lue depuis .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PG = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=os.getenv("PG_PORT", "5432"),
    user=os.getenv("PG_USER", "maintidoc"),
    password=os.getenv("PG_PASSWORD", "maintidoc"),
    database=os.getenv("PG_DB", "maintidoc"),
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "qwen2.5:7b")
OLLAMA_VLM = os.getenv("OLLAMA_VLM", "qwen2.5vl:7b")

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-m3")
TEXT_EMBED_DIM = int(os.getenv("TEXT_EMBED_DIM", "1024"))
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_DIM = int(os.getenv("CLIP_DIM", "512"))

# Tables PGVector (LlamaIndex ajoute le préfixe "data_")
TEXT_TABLE = "maintidoc_text"
IMAGE_TABLE = "maintidoc_images"

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_prompt(name: str) -> dict:
    """Charge un prompt versionné depuis prompts/<name>.yaml."""
    import yaml
    with open(PROMPTS_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
