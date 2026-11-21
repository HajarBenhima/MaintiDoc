"""Descriptions textuelles des images avec un VLM (Qwen2.5-VL via Ollama)."""
import ollama

from maintidoc.config import OLLAMA_BASE_URL, OLLAMA_VLM, load_prompt

_PROMPT = load_prompt("image_caption_v1")
_client = ollama.Client(host=OLLAMA_BASE_URL)


def describe_image(path: str, page_text_hint: str = "") -> str:
    msg = _PROMPT["prompt"]
    if page_text_hint:
        # La légende de la figure se trouve souvent dans le texte de la page
        msg += f"\nContexte de la page (peut contenir la légende) : {page_text_hint[:800]}"
    r = _client.chat(model=OLLAMA_VLM,
                     messages=[{"role": "user", "content": msg, "images": [path]}])
    return r["message"]["content"].strip()
