"""Pipeline RAG simple : recherche -> contexte -> LLM (Qwen via Ollama)."""
from functools import lru_cache

from langchain_ollama import ChatOllama

from maintidoc.config import OLLAMA_BASE_URL, OLLAMA_LLM, load_prompt
from maintidoc.retrieval.search import format_context, search_images, search_text

P = load_prompt("answer_v1")


@lru_cache
def get_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_LLM, base_url=OLLAMA_BASE_URL, temperature=0.1)


def answer(question: str, k: int = 5, n_images: int = 2) -> dict:
    nodes = search_text(question, k=k)
    images = search_images(question, k=n_images)
    msgs = [("system", P["system"]),
            ("user", P["user"].format(context=format_context(nodes), question=question))]
    out = get_llm().invoke(msgs).content
    return {
        "answer": out,
        "sources": [{**n.node.metadata, "score": n.score} for n in nodes],
        "images": [{**img.metadata, "score": s} for img, s in images],
        "prompt_version": P["version"],
    }
