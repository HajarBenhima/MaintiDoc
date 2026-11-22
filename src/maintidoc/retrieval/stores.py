"""Connexions PGVector et modèle d'embeddings texte."""
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

from maintidoc.config import (CLIP_DIM, IMAGE_TABLE, PG, TEXT_EMBED_DIM,
                              TEXT_EMBED_MODEL, TEXT_TABLE)


@lru_cache
def text_store() -> PGVectorStore:
    return PGVectorStore.from_params(**PG, table_name=TEXT_TABLE, embed_dim=TEXT_EMBED_DIM)


@lru_cache
def image_store() -> PGVectorStore:
    return PGVectorStore.from_params(**PG, table_name=IMAGE_TABLE, embed_dim=CLIP_DIM)


@lru_cache
def text_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=TEXT_EMBED_MODEL)


def _engine():
    url = (f"postgresql+psycopg2://{PG['user']}:{PG['password']}"
           f"@{PG['host']}:{PG['port']}/{PG['database']}")
    return create_engine(url)


def delete_document(document_id: str) -> int:
    """Supprime toutes les lignes d'un document (avant réindexation).

    L'identifiant est stocké sous la clé « doc_name » : LlamaIndex réserve
    « document_id », « doc_id » et « ref_doc_id » et les écrase à l'insertion."""
    removed = 0
    with _engine().begin() as conn:
        for table in (TEXT_TABLE, IMAGE_TABLE):
            name = f"data_{table}"
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": name}).scalar() is None:
                continue
            res = conn.execute(
                text(f"DELETE FROM {name} WHERE metadata_->>'doc_name' = :d"),
                {"d": document_id})
            removed += res.rowcount
    return removed


def list_documents() -> list[str]:
    """Identifiants (doc_name) des documents présents dans l'index."""
    name = f"data_{TEXT_TABLE}"
    with _engine().connect() as conn:
        if conn.execute(text("SELECT to_regclass(:t)"), {"t": name}).scalar() is None:
            return []
        rows = conn.execute(text(
            f"SELECT DISTINCT metadata_->>'doc_name' FROM {name} "
            "WHERE metadata_->>'doc_name' IS NOT NULL ORDER BY 1"))
        return [r[0] for r in rows]
