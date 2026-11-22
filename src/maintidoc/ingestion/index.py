"""Embeddings et stockage dans PGVector (texte/tableaux/descriptions + CLIP)."""
import hashlib
import json
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode

from maintidoc.config import CHUNK_OVERLAP, CHUNK_SIZE, OCR_CHUNK_OVERLAP, OCR_CHUNK_SIZE
from maintidoc.ingestion.parse_pdf import ParsedPDF
from maintidoc.retrieval.stores import image_store, text_embed_model, text_store

# Métadonnées non injectées dans le texte embeddé / envoyé au LLM
_HIDDEN = ["image_path", "width", "height", "caption", "figure_caption", "doc_name"]


def _cached_description(image_path: str, hint: str, ocr: bool = False) -> str:
    """Lecture VLM mise en cache (captions.json à côté des images), clé = hash du fichier.
    ocr=True : transcription d'une page scannée au lieu d'une description."""
    from maintidoc.config import OLLAMA_VLM
    from maintidoc.ingestion.captions import describe_image, transcribe_image

    cache_file = Path(image_path).parent / "captions.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    prefix = "ocr_transcription_v1:" if ocr else ""
    key = f"{prefix}{OLLAMA_VLM}:{hashlib.sha1(Path(image_path).read_bytes()).hexdigest()}"
    if key not in cache:
        cache[key] = transcribe_image(image_path) if ocr else describe_image(image_path, hint)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache[key]


def index_text_and_tables(parsed: ParsedPDF) -> int:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    ocr_splitter = SentenceSplitter(chunk_size=OCR_CHUNK_SIZE, chunk_overlap=OCR_CHUNK_OVERLAP)
    regular = [d for d in parsed.text_docs if not d.metadata.get("ocr")]
    scanned = [d for d in parsed.text_docs if d.metadata.get("ocr")]
    nodes = splitter.get_nodes_from_documents(regular) + \
        ocr_splitter.get_nodes_from_documents(scanned)
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
        ocr = bool(im.get("needs_ocr"))
        hint = f"{im['figure_caption']}\n{page_texts.get(im['page'], '')}".strip()
        caption = _cached_description(im["image_path"], hint, ocr=ocr) if with_captions else ""
        meta = {**im, "caption": caption}
        # a) vecteur CLIP -> table images
        clip_nodes.append(ImageNode(image_path=im["image_path"], text=caption,
                                    metadata=meta, embedding=embed_image(im["image_path"])))
        # b) description -> table texte (interrogeable par le RAG classique)
        if caption and ocr:
            # page scannée : la transcription est indexée comme un tableau/texte
            text = f"Page scannée {im['page']} ({im['element_id']}) — transcription :\n{caption}"
            node = TextNode(text=text, metadata={**meta, "type": "table", "ocr": True},
                            embedding=embed.get_text_embedding(text))
            node.excluded_llm_metadata_keys = _HIDDEN
            caption_nodes.append(node)
        elif caption:
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
