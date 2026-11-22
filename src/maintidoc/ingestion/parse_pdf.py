"""Extraction du texte, des tableaux et des images d'un PDF.

Texte : PyMuPDFReader (LlamaIndex), un Document par page.
Tableaux : deux méthodes complémentaires
  1. ancrage sur la légende « Table N: » + filets horizontaux (tableaux style LaTeX/booktabs,
     sans bordures verticales, que find_tables() découpe mal) ;
  2. page.find_tables() pour les tableaux à bordures (fréquents dans les manuels techniques).
Images :
  - images bitmap : extract_image -> PNG dans data/images/<document_id>/ ;
  - figures vectorielles : zone de dessin (cluster_drawings) au-dessus de la légende
    « FIGURE N », rendue en PNG ;
  - pages scannées (peu de texte extractible) : les images sont marquées « needs_ocr »
    pour être transcrites par le VLM lors de l'indexation.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import shutil

import pymupdf
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

from maintidoc.config import IMAGE_DIR

MIN_IMG_SIZE = 100       # ignore logos / icônes
CAPTION_RE = re.compile(r"^(Table|Tableau)\s+(\d+)\s*[:.]", re.IGNORECASE)
FIGURE_RE = re.compile(r"^(Figure|Fig\.|Schéma)\s*(\d+)\s*(?:[:.]|$)", re.IGNORECASE)
FIG_MAX_GAP = 60         # écart max (pt) entre une légende de figure et son dessin
FIG_MIN_SIZE = 40        # taille min (pt) d'un groupe de tracés
RENDER_DPI = 150
SCANNED_TEXT_MAX = 150   # en dessous de ce nb de caractères, la page est considérée scannée
OCR_DPI = 300
OCR_MAX_CELLS = 5        # au-delà, une ligne OCR est une matrice de cases illisible
OCR_LANG = os.getenv("OCR_LANG", "eng")   # ex. "eng+fra" si le pack français est installé
RULE_MAX_GAP = 90        # écart max (pt) entre deux filets d'un même tableau
CAPTION_MAX_GAP = 40     # écart max (pt) entre la légende et le premier filet
COL_GAP = 8              # écart horizontal (pt) qui sépare deux colonnes


@dataclass
class ParsedPDF:
    document_id: str
    source: str
    text_docs: list[Document] = field(default_factory=list)
    table_docs: list[Document] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)

    @property
    def page_texts(self) -> dict[int, str]:
        return {d.metadata["page"]: d.text for d in self.text_docs}


def make_document_id(pdf_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", pdf_path.stem).strip("_")


# ---------------------------------------------------------------- tableaux
def _horizontal_rules(page) -> list[float]:
    ys = set()
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "re":
                r = it[1]
            elif it[0] == "l":
                r = pymupdf.Rect(it[1], it[2])
            else:
                continue
            if r.height < 2 and r.width > 50:
                ys.add(round(r.y0, 1))
    return sorted(ys)


def _chain(rules: list[float], start: float, forward: bool) -> list[float]:
    """Suite de filets proches les uns des autres à partir de `start`."""
    seq = rules if forward else list(reversed(rules))
    near = [y for y in seq if (0 <= y - start <= CAPTION_MAX_GAP if forward
                               else 0 <= start - y <= CAPTION_MAX_GAP)]
    if not near:
        return []
    out = [near[0]]
    for y in seq[seq.index(near[0]) + 1:]:
        if abs(y - out[-1]) > RULE_MAX_GAP:
            break
        out.append(y)
    return sorted(out)


def _rows_text(page, rect=None, textpage=None) -> str:
    """Reconstitue les lignes (tableau ou page OCR) ; « | » entre colonnes éloignées."""
    words = sorted(page.get_text("words", clip=rect, textpage=textpage),
                   key=lambda w: (round(w[3]), w[0]))
    lines, current, last_y = [], [], None
    for w in words:
        if last_y is not None and abs(w[3] - last_y) > 3:
            lines.append(current)
            current = []
        current.append(w)
        last_y = w[3]
    if current:
        lines.append(current)
    out = []
    for ln in lines:
        ln.sort(key=lambda w: w[0])
        s = ln[0][4]
        for prev, w in zip(ln, ln[1:]):
            s += (" | " if w[0] - prev[2] > COL_GAP else " ") + w[4]
        out.append(s)
    return "\n".join(out)


def _caption_tables(page) -> list[tuple[str, str, pymupdf.Rect]]:
    """Tableaux repérés par leur légende. Renvoie [(légende, contenu, bbox)]."""
    rules = _horizontal_rules(page)
    found = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not CAPTION_RE.match(text.strip()):
            continue
        chain = _chain(rules, y1, forward=True) or _chain(rules, y0, forward=False)
        if len(chain) < 2:
            continue
        rect = pymupdf.Rect(page.rect.x0, chain[0] - 1, page.rect.x1, chain[-1] + 1)
        content = _rows_text(page, rect)
        if content.strip():
            found.append((" ".join(text.split()), content, rect))
    return found


def _extract_tables(page, pno: int, base_meta: dict) -> list[Document]:
    docs = []
    caption_rects = []
    for caption, content, rect in _caption_tables(page):
        num = CAPTION_RE.match(caption).group(2)
        caption_rects.append(rect)
        docs.append(Document(
            text=f"{caption}\n(page {pno})\n{content}",
            metadata={**base_meta, "page": pno, "type": "table",
                      "element_id": f"table_{num}", "caption": caption[:300]},
        ))
    # Tableaux à bordures non déjà couverts par la méthode légende
    for t_idx, tab in enumerate(page.find_tables().tables):
        bbox = pymupdf.Rect(tab.bbox)
        if any(bbox.intersects(r) for r in caption_rects):
            continue
        md = tab.to_markdown()
        if md.strip():
            docs.append(Document(
                text=f"Tableau (page {pno}) :\n{md}",
                metadata={**base_meta, "page": pno, "type": "table",
                          "element_id": f"table_p{pno}_{t_idx}"},
            ))
    return docs


# ---------------------------------------------------------------- OCR
def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def is_scanned(page) -> bool:
    """Page dont le texte n'est pas extractible mais qui contient des images."""
    return len(page.get_text().strip()) < SCANNED_TEXT_MAX and bool(page.get_images())


def _is_noise(line: str) -> bool:
    """Ligne OCR inexploitable : matrice de cases (beaucoup de « | ») ou peu de vrais mots."""
    if line.count("|") > OCR_MAX_CELLS:
        return True
    return len(line.strip()) < 6 or not re.search(r"[A-Za-zÀ-ÿ]{4,}", line)


def ocr_page(page) -> str:
    """OCR Tesseract de la page entière, lignes reconstruites avec séparateurs de colonnes.
    Les lignes de bruit (matrices de « X » mal reconnues) sont écartées."""
    tp = page.get_textpage_ocr(dpi=OCR_DPI, full=True, language=OCR_LANG)
    lines = _rows_text(page, textpage=tp).splitlines()
    return "\n".join(l for l in lines if not _is_noise(l))


# ---------------------------------------------------------------- images
def _figure_captions(page) -> list[tuple[str, str, pymupdf.Rect]]:
    """Légendes « Figure N: » de la page : [(numéro, texte, bbox)]."""
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        m = FIGURE_RE.match(text.strip())
        if m:
            out.append((m.group(2), " ".join(text.split())[:400], pymupdf.Rect(x0, y0, x1, y1)))
    return out


def _closest_caption(captions, img_rect):
    """Légende la plus proche, en privilégiant celles situées sous l'image."""
    if not captions:
        return None
    if img_rect is None:
        return captions[0] if len(captions) == 1 else None

    def dist(c):
        below = c[2].y0 >= img_rect.y1 - 5
        return (0 if below else 1, abs(c[2].y0 - img_rect.y1))
    return min(captions, key=dist)


def _vector_figure_rect(page, caption_rect, clusters, upper_limit=0.0,
                        lower_limit=None) -> pymupdf.Rect | None:
    """Zone de dessin associée à une légende : groupes de tracés juste au-dessus
    (ou, à défaut, juste en dessous), chaînés tant qu'ils restent proches.
    Les groupes au-delà des légendes voisines (upper/lower_limit) sont ignorés."""
    lower_limit = page.rect.y1 if lower_limit is None else lower_limit

    def grow(candidates, above: bool):
        region = None
        for r in sorted(candidates, key=lambda r: -r.y1 if above else r.y0):
            edge = caption_rect.y0 if region is None and above else (
                caption_rect.y1 if region is None else (region.y0 if above else region.y1))
            close = (r.y1 >= edge - FIG_MAX_GAP) if above else (r.y0 <= edge + FIG_MAX_GAP)
            if close:
                region = r if region is None else region | r
        return region

    above = [r for r in clusters if upper_limit <= r.y0 < caption_rect.y0]
    below = [r for r in clusters if caption_rect.y1 - 5 <= r.y0 and r.y1 <= lower_limit]
    return grow(above, True) or grow(below, False)


def _extract_vector_figures(page, pno, base_meta, out_dir, captions, done_numbers) -> list[dict]:
    clusters = [pymupdf.Rect(r) for r in page.cluster_drawings()]
    clusters = [r for r in clusters if r.width > FIG_MIN_SIZE and r.height > FIG_MIN_SIZE]
    figures = []
    caps = sorted(captions, key=lambda c: c[2].y0)
    for i, (num, caption, cap_rect) in enumerate(caps):
        if num in done_numbers:          # déjà couverte par une image bitmap
            continue
        upper = caps[i - 1][2].y1 if i > 0 else 0.0
        lower = caps[i + 1][2].y0 if i + 1 < len(caps) else None
        region = _vector_figure_rect(page, cap_rect, clusters, upper, lower)
        if region is None:
            continue
        clip = (region | cap_rect) + (-5, -5, 5, 5)
        path = out_dir / f"p{pno}_figure{num}.png"
        page.get_pixmap(clip=clip & page.rect, dpi=RENDER_DPI).save(path)
        figures.append({
            **base_meta, "page": pno, "type": "image",
            "element_id": f"figure_{num}",
            "figure_label": f"Figure {num}", "figure_caption": caption,
            "image_path": str(path), "vector": True,
            "width": round(clip.width * RENDER_DPI / 72),
            "height": round(clip.height * RENDER_DPI / 72),
        })
    return figures


def _extract_images(pdf, page, pno: int, base_meta: dict, out_dir: Path) -> list[dict]:
    images = []
    captions = _figure_captions(page)
    # Page scannée : OCR Tesseract si disponible (voir parse_pdf), sinon transcription VLM
    if is_scanned(page) and _tesseract_available():
        return []
    scanned = is_scanned(page)
    for i_idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        info = pdf.extract_image(xref)
        if info["width"] < MIN_IMG_SIZE or info["height"] < MIN_IMG_SIZE:
            continue
        pix = pymupdf.Pixmap(pdf, xref)
        if pix.alpha:                       # supprimer le canal alpha
            pix = pymupdf.Pixmap(pix, 0)
        if pix.n >= 4:                      # CMYK -> RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        path = out_dir / f"p{pno}_img{i_idx}.png"
        pix.save(path)
        rects = page.get_image_rects(xref)
        cap = _closest_caption(captions, rects[0] if rects else None)
        images.append({
            **base_meta, "page": pno, "type": "image",
            "element_id": f"figure_{pno}_{i_idx}",
            "figure_label": f"Figure {cap[0]}" if cap else "",
            "figure_caption": cap[1] if cap else "",
            "image_path": str(path),
            "width": info["width"], "height": info["height"],
            "needs_ocr": scanned,
        })
    done = {im["figure_label"].split()[-1] for im in images if im["figure_label"]}
    images += _extract_vector_figures(page, pno, base_meta, out_dir, captions, done)
    return images


# ---------------------------------------------------------------- point d'entrée
def parse_pdf(pdf_path: str | Path, extract_images: bool = True) -> ParsedPDF:
    pdf_path = Path(pdf_path)
    doc_id = make_document_id(pdf_path)
    base_meta = {"doc_name": doc_id, "source": pdf_path.name}
    result = ParsedPDF(document_id=doc_id, source=pdf_path.name)

    # 1. Texte (un Document LlamaIndex par page)
    pages = PyMuPDFReader().load_data(file_path=pdf_path)
    for i, p in enumerate(pages, start=1):
        if p.text.strip():
            result.text_docs.append(
                Document(text=p.text, metadata={**base_meta, "page": i, "type": "text"})
            )

    # 2. Tableaux et 3. Images
    out_dir = Path(IMAGE_DIR) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            if is_scanned(page) and _tesseract_available():
                text = ocr_page(page)
                if text.strip():
                    result.text_docs = [d for d in result.text_docs if d.metadata["page"] != pno]
                    result.text_docs.append(Document(
                        text=f"Page {pno} (texte obtenu par OCR) :\n{text}",
                        metadata={**base_meta, "page": pno, "type": "text", "ocr": True}))
                continue
            result.table_docs += _extract_tables(page, pno, base_meta)
            if extract_images:
                result.images += _extract_images(pdf, page, pno, base_meta, out_dir)
    return result
