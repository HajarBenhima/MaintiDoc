"""Lecture des images par un VLM via Ollama : description de figures et transcription
(OCR) des pages scannées."""
import ollama

from maintidoc.config import OLLAMA_BASE_URL, OLLAMA_VLM, load_prompt

_client = ollama.Client(host=OLLAMA_BASE_URL)


def describe_image(path: str, page_text_hint: str = "",
                   prompt_name: str = "image_caption_v1", max_tokens: int = 400) -> str:
    msg = load_prompt(prompt_name)["prompt"]
    if page_text_hint:
        # La légende et le titre de section se trouvent souvent dans le texte de la page
        msg += f"\nContexte de la page (peut contenir la légende) : {page_text_hint[:800]}"
    r = _client.chat(model=OLLAMA_VLM, options={"temperature": 0, "num_predict": max_tokens,
                              "repeat_penalty": 1.15},
                     messages=[{"role": "user", "content": msg, "images": [path]}])
    return r["message"]["content"].strip()


def transcribe_image(path: str) -> str:
    """Transcription d'une page scannée (tableaux en Markdown)."""
    return describe_image(path, prompt_name="ocr_transcription_v1", max_tokens=1500)
