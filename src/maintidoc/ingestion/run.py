"""Point d'entrée de l'ingestion.

Usage :
    python -m maintidoc.ingestion.run "data/pdfs/attention is all you need.pdf"
    python -m maintidoc.ingestion.run data/pdfs/manuel.pdf --no-captions
"""
import argparse

from maintidoc.ingestion.index import index_images, index_text_and_tables
from maintidoc.ingestion.parse_pdf import parse_pdf
from maintidoc.retrieval.stores import delete_document


def ingest(pdf_path: str, with_captions: bool = True) -> dict:
    parsed = parse_pdf(pdf_path)
    removed = delete_document(parsed.document_id)   # idempotence : pas de doublons
    n_text = index_text_and_tables(parsed)
    n_img = index_images(parsed, with_captions=with_captions)
    stats = {"document_id": parsed.document_id, "pages": len(parsed.text_docs),
             "tables": len(parsed.table_docs), "images": n_img,
             "text_nodes": n_text, "old_rows_removed": removed}
    print(stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Indexe un PDF dans PGVector")
    ap.add_argument("pdf")
    ap.add_argument("--no-captions", action="store_true",
                    help="ne pas générer les descriptions d'images (plus rapide)")
    args = ap.parse_args()
    ingest(args.pdf, with_captions=not args.no_captions)
