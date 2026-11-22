"""Fonctions de recherche (texte, tableaux, descriptions, CLIP).

- filtre optionnel par document (doc_names) ;
- requête multilingue : question d'origine + reformulation anglaise, résultats fusionnés.
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores import (FilterOperator, MetadataFilter,
                                            MetadataFilters, VectorStoreQuery)

from maintidoc.retrieval.query import translate_query
from maintidoc.retrieval.stores import image_store, text_embed_model, text_store


def build_filters(type_: str | None = None,
                  doc_names: list[str] | None = None) -> MetadataFilters | None:
    filters = []
    if type_:
        filters.append(MetadataFilter(key="type", value=type_))
    if doc_names:
        filters.append(MetadataFilter(key="doc_name", value=list(doc_names),
                                      operator=FilterOperator.IN))
    return MetadataFilters(filters=filters) if filters else None


def search_queries(query: str, translate: bool = True) -> list[str]:
    queries = [query]
    en = translate_query(query) if translate else ""
    if en and en.lower() != query.lower():
        queries.append(en)
    return queries


def search_text(query: str, k: int = 5, type_: str | None = None,
                doc_names: list[str] | None = None,
                translate: bool = True) -> list[NodeWithScore]:
    """type_ : None (tout), 'text', 'table' ou 'image_description'."""
    index = VectorStoreIndex.from_vector_store(text_store(), embed_model=text_embed_model())
    retriever = index.as_retriever(similarity_top_k=k, filters=build_filters(type_, doc_names))
    best: dict[str, NodeWithScore] = {}
    for q in search_queries(query, translate):
        for n in retriever.retrieve(q):
            if n.node.node_id not in best or (n.score or 0) > (best[n.node.node_id].score or 0):
                best[n.node.node_id] = n
    return sorted(best.values(), key=lambda n: n.score or 0, reverse=True)[:k]


def search_images(query: str, k: int = 3, doc_names: list[str] | None = None) -> list[tuple]:
    """Recherche texte -> image avec CLIP (requête anglaise si disponible)."""
    from maintidoc.retrieval.clip_encoder import embed_text
    q = search_queries(query)[-1]          # CLIP comprend surtout l'anglais
    res = image_store().query(VectorStoreQuery(query_embedding=embed_text(q),
                                               similarity_top_k=k,
                                               filters=build_filters(None, doc_names)))
    return list(zip(res.nodes or [], res.similarities or []))


def reference(meta: dict) -> str:
    return f"[{meta.get('source')} | p.{meta.get('page')} | {meta.get('element_id', meta.get('type'))}]"


def format_context(nodes: list[NodeWithScore]) -> str:
    """Bloc de contexte avec identifiants de source citables."""
    return "\n\n---\n\n".join(
        f"{reference(n.node.metadata)}\n{n.node.get_content(metadata_mode=MetadataMode.NONE)}"
        for n in nodes)
