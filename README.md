# MaintiDoc — V1 (Étape 1 : RAG agentique multimodal)

Assistant documentaire pour techniciens de maintenance. Il répond à partir de PDF
(texte, tableaux, images) avec des références vérifiables (document, page, élément).
Il ne pilote aucun équipement et ne remplace pas un technicien habilité.

Stack : LlamaIndex + PyMuPDF · BGE-M3 · CLIP · PGVector (PostgreSQL) · Qwen2.5 / Qwen2.5-VL via Ollama · LangChain · Streamlit · FastAPI.

## Installation

```bash
# 1. Base PGVector
cp .env.example .env
docker compose up -d

# 2. Modèles Ollama
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b

# 3. Python (3.10–3.12)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Au premier lancement, BGE-M3 (~2,2 Go) et CLIP (~600 Mo) sont téléchargés depuis Hugging Face.

### ⚠️ Mac Intel (x86_64)

- **PyMuPDF** : les dernières versions n'ont pas de paquet précompilé pour Intel. Installer avec
  `pip install --only-binary=pymupdf -r requirements.txt` (sinon pip tente une compilation qui échoue).
- **PyTorch** : le dernier paquet pour Mac Intel est `torch==2.2.2`. Combinaison testée :

  ```bash
  pip install --only-binary=pymupdf,torch "torch==2.2.2" "numpy<2" \
      "transformers>=4.44,<4.50" "sentence-transformers>=3,<4" -r requirements.txt
  ```
- **Ollama sur CPU** : un modèle 7B est lent. Modèle testé : `qwen3-vl:4b-instruct`. Il gère la vision
  et l'appel d'outils, et sert donc à la fois de LLM et de VLM (`OLLAMA_LLM` et `OLLAMA_VLM` dans `.env`).

## Utilisation

```bash
# Indexer un PDF
python -m maintidoc.ingestion.run "data/pdfs/attention is all you need.pdf"
#   --no-captions : sans descriptions d'images (plus rapide)

# Interface
streamlit run src/maintidoc/app/streamlit_app.py

# API (optionnel)
uvicorn maintidoc.api.main:app --reload

# Tests d'extraction
pytest
```

Vérifier le contenu de la base :

```bash
docker exec -it maintidoc-db psql -U maintidoc -c \
  "SELECT metadata_->>'type' AS type, count(*) FROM data_maintidoc_text GROUP BY 1;"
```

## Structure

```
data/pdfs/            PDF à indexer
data/images/          images extraites (générées)
prompts/              prompts versionnés (answer_v1, image_caption_v1)
docker/init.sql       extension pgvector
src/maintidoc/
  config.py           paramètres (.env)
  ingestion/          parse_pdf · captions (VLM) · index (PGVector) · run
  retrieval/          stores · clip_encoder · search
  rag/pipeline.py     RAG simple
  agent/              outils + agent LangChain
  api/main.py         FastAPI
  app/                interface Streamlit
tests/                tests + questions de référence
```

## Tables PGVector

| Table | Dimension | Contenu |
|---|---|---|
| `data_maintidoc_text` | 1024 (BGE-M3) | passages `text`, tableaux `table`, descriptions `image_description` |
| `data_maintidoc_images` | 512 (CLIP) | vecteurs CLIP des images |

Métadonnées : `document_id`, `source`, `page`, `type`, `element_id` (+ `image_path`, `caption` pour les images).
