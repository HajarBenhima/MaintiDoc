"""Embeddings et stockage dans PGVector (texte/tableaux/descriptions + CLIP)."""
import hashlib
import json
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode

from maintidoc.config import CHUNK_OVERLAP, CHUNK_SIZE
from maintidoc.ingestion.parse_pdf import ParsedPDF
from maintidoc.retrieval.stores import image_store, text_embed_model, text_store

# Métadonnées non injectées dans le texte embeddé / envoyé au LLM
_HIDDEN = ["image_path", "width", "height", "caption", "figure_caption", "document_id"]


def _cached_description(image_path: str, hint: str) -> str:
    """Description VLM mise en cache (captions.json à côté des images), clé = hash du fichier."""
    from maintidoc.config import OLLAMA_VLM
    from maintidoc.ingestion.captions import describe_image

    cache_file = Path(image_path).parent / "captions.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{OLLAMA_VLM}:{hashlib.sha1(Path(image_path).read_bytes()).hexdigest()}"
    if key not in cache:
        cache[key] = describe_image(image_path, hint)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache[key]


def index_text_and_tables(parsed: ParsedPDF) -> int:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(parsed.text_docs)
    for i, n in enumerate(nodes):
        n.metadata["element_id"] = f"chunk_{n.metadata['page']}_{i}"
    # Un tableau = un seul nœud (on ne le coupe pas)
    nodes += [TextNode(text=d.text, metadata=d.metadata) for d in parsed.table_docs]
    for n in nodes:
        n.excluded_embed_metadata_keys = _HIDDEN
        n.excluded_llm_metadata_keys = _HIDDEN

    storage = StorageContext.from_defaults(vector_store=text_store())
    VectorStoreIndex(nodes, storage_context=storage,
                     embed_model=text_embed_model(), show_progress=True)
    return len(nodes)


def index_images(parsed: ParsedPDF, with_captions: bool = True) -> int:
    if not parsed.images:
        return 0
    from maintidoc.retrieval.clip_encoder import embed_image

    embed = text_embed_model()
    page_texts = parsed.page_texts
    clip_nodes, caption_nodes = [], []
    for im in parsed.images:
        hint = im["figure_caption"] or page_texts.get(im["page"], "")
        caption = _cached_description(im["image_path"], hint) if with_captions else ""
        meta = {**im, "caption": caption}
        # a) vecteur CLIP -> table images
        clip_nodes.append(ImageNode(image_path=im["image_path"], text=caption,
                                    metadata=meta, embedding=embed_image(im["image_path"])))
        # b) description -> table texte (interrogeable par le RAG classique)
        if caption:
            label = f"{im['figure_label']} ({im['element_id']}, page {im['page']})" \
                if im["figure_label"] else f"Figure {im['element_id']} (page {im['page']})"
            text = f"{label}\nLégende : {im['figure_caption'] or '—'}\nDescription : {caption}"
            node = TextNode(text=text, metadata={**meta, "type": "image_description"},
                            embedding=embed.get_text_embedding(text))
            node.excluded_llm_metadata_keys = _HIDDEN
            caption_nodes.append(node)
        print(f"  image {im['element_id']} indexée")

    image_store().add(clip_nodes)
    if caption_nodes:
        text_store().add(caption_nodes)
    return len(clip_nodes)
