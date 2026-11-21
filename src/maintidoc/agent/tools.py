"""Outils de recherche exposés à l'agent LangChain."""
from langchain_core.tools import tool

from maintidoc.retrieval.search import format_context, reference, search_images, search_text


@tool
def rechercher_texte(query: str) -> str:
    """Recherche des passages textuels dans les documents techniques indexés.
    À utiliser pour les définitions, explications et procédures."""
    return format_context(search_text(query, k=5, type_="text")) or "Aucun résultat."


@tool
def rechercher_tableaux(query: str) -> str:
    """Recherche dans les TABLEAUX des documents (valeurs numériques, scores,
    paramètres, codes d'erreur, caractéristiques techniques)."""
    return format_context(search_text(query, k=3, type_="table")) or "Aucun tableau."


@tool
def rechercher_figures(query: str) -> str:
    """Recherche des FIGURES, schémas ou images et renvoie leur description et leur page.
    À utiliser quand la question porte sur un schéma, une figure ou une illustration."""
    parts = [format_context(search_text(query, k=3, type_="image_description"))]
    for node, score in search_images(query, k=2):
        m = node.metadata
        parts.append(f"{reference(m)} {m.get('figure_label', '')} (similarité CLIP {score:.2f})\n"
                     f"Légende : {m.get('figure_caption') or '—'}\nDescription : {m.get('caption', '')}")
    return "\n\n".join(p for p in parts if p) or "Aucune figure."


TOOLS = [rechercher_texte, rechercher_tableaux, rechercher_figures]
