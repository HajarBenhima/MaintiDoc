"""Outils de recherche exposés à l'agent LangChain.

Le filtre de documents (doc_names) est transmis par la configuration d'exécution
(config["configurable"]["doc_names"]) : il n'est pas choisi par le LLM.

Chaque outil renvoie (texte pour le LLM, artifact) : l'artifact contient les sources
et images trouvées, affichées par l'interface sans passer par le modèle.
"""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from maintidoc.retrieval.search import format_context, reference, search_images, search_text

MIN_TABLE_SCORE = 0.45   # en dessous, on complète par une recherche texte


def _artifact(nodes=(), images=()) -> dict:
    return {"sources": [{**n.node.metadata, "score": n.score} for n in nodes],
            "images": [{**m, "score": s} for m, s in images]}


def _docs(config: RunnableConfig | None) -> list[str] | None:
    return ((config or {}).get("configurable") or {}).get("doc_names") or None


@tool(response_format="content_and_artifact")
def rechercher_texte(query: str, config: RunnableConfig) -> tuple[str, dict]:
    """Recherche des passages textuels dans les documents techniques indexés.
    À utiliser pour les définitions, explications, procédures, consignes et valeurs
    mentionnées dans le texte."""
    nodes = search_text(query, k=5, type_="text", doc_names=_docs(config))
    return format_context(nodes) or "Aucun résultat.", _artifact(nodes)


@tool(response_format="content_and_artifact")
def rechercher_tableaux(query: str, config: RunnableConfig) -> tuple[str, dict]:
    """Recherche dans les TABLEAUX des documents (valeurs numériques, scores,
    paramètres, codes d'erreur, matrices de dépannage)."""
    docs = _docs(config)
    tables = search_text(query, k=3, type_="table", doc_names=docs)
    out, nodes = format_context(tables), list(tables)
    if not tables or (tables[0].score or 0) < MIN_TABLE_SCORE:
        extra = search_text(query, k=3, type_="text", doc_names=docs)
        nodes += extra
        out = (out + "\n\n" if out else "Aucun tableau pertinent.\n\n") + \
            "Passages texte complémentaires :\n" + format_context(extra)
    return out, _artifact(nodes)


@tool(response_format="content_and_artifact")
def rechercher_figures(query: str, config: RunnableConfig) -> tuple[str, dict]:
    """Recherche des FIGURES, schémas ou images et renvoie leur description et leur page.
    À utiliser quand la question porte sur un schéma, une figure ou une illustration."""
    docs = _docs(config)
    described = search_text(query, k=3, type_="image_description", doc_names=docs)
    clip = search_images(query, k=2, doc_names=docs)
    parts = [format_context(described)]
    for node, score in clip:
        m = node.metadata
        parts.append(f"{reference(m)} {m.get('figure_label', '')} (similarité CLIP {score:.2f})\n"
                     f"Légende : {m.get('figure_caption') or '—'}\nDescription : {m.get('caption', '')}")
    images = [(n.node.metadata, n.score) for n in described] + \
        [(node.metadata, score) for node, score in clip]
    return "\n\n".join(p for p in parts if p) or "Aucune figure.", \
        _artifact(described, images)


TOOLS = [rechercher_texte, rechercher_tableaux, rechercher_figures]
