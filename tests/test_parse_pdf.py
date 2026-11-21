"""Tests d'extraction (sans base de données ni LLM)."""
from pathlib import Path

import pytest

from maintidoc.config import PDF_DIR
from maintidoc.ingestion.parse_pdf import parse_pdf

PDF = Path(PDF_DIR) / "attention is all you need.pdf"


@pytest.fixture(scope="module")
def parsed():
    if not PDF.exists():
        pytest.skip("PDF de référence absent")
    return parse_pdf(PDF)


def test_text_pages(parsed):
    assert len(parsed.text_docs) >= 10
    assert all(d.metadata["page"] >= 1 for d in parsed.text_docs)
    assert "Attention" in parsed.text_docs[0].text


def test_images_extracted(parsed):
    # Figure 1 (p.3) et Figure 2 (p.4)
    pages = {im["page"] for im in parsed.images}
    assert {3, 4} <= pages
    assert all(Path(im["image_path"]).exists() for im in parsed.images)


def test_tables_extracted(parsed):
    ids = {d.metadata["element_id"] for d in parsed.table_docs}
    assert {"table_1", "table_2", "table_3"} <= ids
    assert all(d.metadata["type"] == "table" for d in parsed.table_docs)


def test_table_content(parsed):
    t1 = next(d for d in parsed.table_docs if d.metadata["element_id"] == "table_1")
    assert "Self-Attention" in t1.text and "Recurrent" in t1.text
    t2 = next(d for d in parsed.table_docs if d.metadata["element_id"] == "table_2")
    assert "28.4" in t2.text   # BLEU EN-DE du Transformer (big)


def test_metadata_for_citations(parsed):
    for d in parsed.table_docs:
        assert {"document_id", "source", "page", "element_id"} <= d.metadata.keys()


def test_figure_labels(parsed):
    labels = {im["element_id"]: im["figure_label"] for im in parsed.images}
    assert labels["figure_3_0"] == "Figure 1"
    assert labels["figure_4_0"] == "Figure 2" and labels["figure_4_1"] == "Figure 2"
