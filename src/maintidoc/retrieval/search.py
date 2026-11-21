"""Fonctions de recherche (texte, tableaux, descriptions, CLIP)."""
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores import (ExactMatchFilter, MetadataFilters,
                                            VectorStoreQuery)

from maintidoc.retrieval.stores import image_store, text_embed_model, text_store


def search_text(query: str, k: int = 5, type_: str | None = None) -> list[NodeWithScore]:
    """type_ : None (tout), 'text', 'table' ou 'image_description'."""
    index = VectorStoreIndex.from_vector_store(text_store(), embed_model=text_embed_model())
    filters = (MetadataFilters(filters=[ExactMatchFilter(key="type", value=type_)])
               if type_ else None)
    return index.as_retriever(similarity_top_k=k, filters=filters).retrieve(query)


def search_images(query: str, k: int = 3) -> list[tuple]:
    """Recherche texte -> image avec CLIP. Renvoie [(node, similarité)]."""
    from maintidoc.retrieval.clip_encoder import embed_text
    res = image_store().query(VectorStoreQuery(query_embedding=embed_text(query),
                                               similarity_top_k=k))
    return list(zip(res.nodes or [], res.similarities or []))


def reference(meta: dict) -> str:
    return f"[{meta.get('source')} | p.{meta.get('page')} | {meta.get('element_id', meta.get('type'))}]"


def format_context(nodes: list[NodeWithScore]) -> str:
    """Bloc de contexte avec identifiants de source citables."""
    return "\n\n---\n\n".join(
        f"{reference(n.node.metadata)}\n{n.node.get_content(metadata_mode=MetadataMode.NONE)}"
        for n in nodes)
