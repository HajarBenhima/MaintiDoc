"""Interface Streamlit MaintiDoc.

Lancement : streamlit run src/maintidoc/app/streamlit_app.py
"""
import uuid
from pathlib import Path

import streamlit as st

from maintidoc.config import PDF_DIR

st.set_page_config(page_title="MaintiDoc", page_icon="🔧", layout="wide")
st.title("🔧 MaintiDoc — Assistant documentaire de maintenance")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []

with st.sidebar:
    st.header("📄 Documents")
    pdf = st.file_uploader("Charger un PDF", type="pdf")
    captions = st.checkbox("Décrire les images (VLM, plus lent)", value=True)
    if pdf and st.button("Indexer", type="primary"):
        dest = Path(PDF_DIR) / pdf.name          # garde le nom d'origine pour les citations
        dest.write_bytes(pdf.getbuffer())
        from maintidoc.ingestion.run import ingest
        with st.spinner("Extraction, descriptions d'images, embeddings..."):
            stats = ingest(str(dest), with_captions=captions)
        st.success("Document indexé ✅")
        st.json(stats)

    st.divider()
    from maintidoc.retrieval.stores import list_documents
    try:
        available = list_documents()
    except Exception as e:                      # base non démarrée
        available = []
        st.error(f"Base PGVector inaccessible : {e}")
    doc_names = st.multiselect("Documents interrogés", available, default=available,
                               help="La recherche est limitée aux documents sélectionnés.")
    mode = st.radio("Mode", ["Agent (LangChain)", "RAG simple"])
    if st.button("Nouvelle conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
    st.caption("⚠️ Aide documentaire : ne pilote aucun équipement et "
               "ne remplace pas un technicien habilité.")


def show_sources(r: dict):
    if r.get("status") == "refused":
        st.caption("🛑 Demande d'action refusée par le garde-fou (aucun appel au modèle).")
    if r.get("search_queries") and len(r["search_queries"]) > 1:
        st.caption(f"🔎 Requêtes : {' · '.join(r['search_queries'])}")
    if r.get("tools_used"):
        with st.expander("🛠️ Outils utilisés"):
            st.write(", ".join(r["tools_used"]))
    if r.get("sources"):
        with st.expander("📚 Sources"):
            for s in r["sources"]:
                st.write(f"- **{s.get('source')}** p.{s.get('page')} — {s.get('type')} "
                         f"({s.get('element_id', '')}) · score {s['score']:.2f}")
    for im in r.get("images", []):
        if Path(im["image_path"]).exists():
            st.image(im["image_path"], width=350,
                     caption=f"p.{im['page']} · {im['element_id']} · CLIP {im['score']:.2f}")


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("extra"):
            show_sources(msg["extra"])

if q := st.chat_input("Posez votre question technique..."):
    st.session_state.history.append({"role": "user", "content": q})
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Recherche..."):
            if mode.startswith("Agent"):
                from maintidoc.agent.agent import ask
                r = ask(q, thread_id=st.session_state.session_id, doc_names=doc_names)
            else:
                from maintidoc.rag.pipeline import answer
                r = answer(q, doc_names=doc_names)
        st.markdown(r["answer"])
        show_sources(r)
    st.session_state.history.append({"role": "assistant", "content": r["answer"], "extra": r})
