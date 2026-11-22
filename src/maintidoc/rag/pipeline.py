"""Pipeline RAG simple : recherche -> contexte -> LLM (Qwen via Ollama)."""
from maintidoc.config import load_prompt
from maintidoc.llm import get_llm
from maintidoc.retrieval.search import (format_context, search_images, search_queries,
                                        search_text)
from maintidoc.safety.guard import is_action_request, refusal_response

P = load_prompt("answer_v2")


def answer(question: str, k: int = 5, n_images: int = 2,
           doc_names: list[str] | None = None) -> dict:
    if is_action_request(question):          # garde-fou avant tout appel au LLM
        return refusal_response()
    nodes = search_text(question, k=k, doc_names=doc_names)
    # Images affichées : d'abord les figures dont la description a été retrouvée
    # (cohérentes avec la réponse) ; CLIP seulement à défaut (peu fiable sur les schémas).
    described = [n for n in nodes if n.node.metadata.get("type") == "image_description"]
    images = [(n.node.metadata, n.score) for n in described[:n_images]] or \
        [(img.metadata, s) for img, s in search_images(question, k=n_images, doc_names=doc_names)]
    msgs = [("system", P["system"]),
            ("user", P["user"].format(context=format_context(nodes), question=question))]
    out = get_llm(0.1).invoke(msgs).content
    return {
        "answer": out,
        "status": "answered",
        "sources": [{**n.node.metadata, "score": n.score} for n in nodes],
        "images": [{**m, "score": s} for m, s in images],
        "search_queries": search_queries(question),
        "prompt_version": P["version"],
    }
