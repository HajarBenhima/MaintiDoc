"""Encodeur CLIP (images et texte dans le même espace vectoriel)."""
from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from maintidoc.config import CLIP_MODEL

_device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def _load():
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(_device).eval()
    return model, CLIPProcessor.from_pretrained(CLIP_MODEL)


@torch.no_grad()
def embed_image(path: str) -> list[float]:
    model, proc = _load()
    inputs = proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to(_device)
    v = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()


@torch.no_grad()
def embed_text(query: str) -> list[float]:
    model, proc = _load()
    inputs = proc(text=[query], return_tensors="pt", padding=True, truncation=True).to(_device)
    v = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()
