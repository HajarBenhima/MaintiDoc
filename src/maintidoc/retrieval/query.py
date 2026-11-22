"""Reformulation de la requête (français -> anglais technique)."""
from functools import lru_cache

from maintidoc.config import QUERY_TRANSLATION, load_prompt


@lru_cache(maxsize=512)
def translate_query(question: str) -> str:
    """Requête anglaise équivalente, ou chaîne vide si désactivé / en échec."""
    if not QUERY_TRANSLATION:
        return ""
    from maintidoc.llm import get_llm
    prompt = load_prompt("query_rewrite_v1")["prompt"].format(question=question)
    try:
        out = get_llm(0.0, 60).invoke(prompt).content
    except Exception:          # la recherche continue avec la requête d'origine
        return ""
    line = out.strip().splitlines()[0] if out.strip() else ""
    return line.strip(" \"'«»`")
