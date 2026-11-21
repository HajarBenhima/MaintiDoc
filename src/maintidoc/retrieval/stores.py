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
    """Supprime toutes les lignes d'un document (avant réindexation)."""
    removed = 0
    with _engine().begin() as conn:
        for table in (TEXT_TABLE, IMAGE_TABLE):
            name = f"data_{table}"
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": name}).scalar() is None:
                continue
            res = conn.execute(
                text(f"DELETE FROM {name} WHERE metadata_->>'document_id' = :d"),
                {"d": document_id})
            removed += res.rowcount
    return removed
