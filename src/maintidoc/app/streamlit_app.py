"""Interface Streamlit MaintiDoc.

Lancement (depuis la racine du projet, pour charger .streamlit/config.toml) :
    streamlit run src/maintidoc/app/streamlit_app.py
"""
import html
import uuid
from pathlib import Path

import streamlit as st

from maintidoc.app.mascot import mascot_b64, user_b64
from maintidoc.config import PDF_DIR

st.set_page_config(page_title="MaintiDoc", page_icon=":material/build:", layout="wide")

MODES = {"Agent": "Choisit ses outils de recherche et garde le fil de la conversation",
         "RAG simple": "Une recherche, puis une réponse"}
EXAMPLES = {
    "pentair_pompe_centrifuge_F00637": [
        "La pompe ne délivre aucun débit : quelles sont les causes possibles ?",
        "Quelle figure montre la vue éclatée de la pompe ?",
        "Comment vérifier l'alignement de l'accouplement ?",
    ],
    "attention_is_all_you_need": [
        "Décris l'architecture du Transformer (Figure 1).",
        "Quel score BLEU obtient le Transformer (big) en anglais-allemand ?",
    ],
}

st.html("""
<style>
.block-container { padding-top: 4.5rem; max-width: 1100px; }

/* Cartouche : en-tête inspiré du cartouche des plans techniques */
.cartouche { display: grid; grid-template-columns: 2.2fr 1fr 1fr 1fr;
  border: 2px solid #1F2A33; background: #FFFFFF; margin-bottom: 1.5rem; }
.cartouche > div { padding: .55rem .9rem; border-left: 1px solid #1F2A33; }
.cartouche > div:first-child { border-left: none; }
.c-name { font-family: "Barlow Condensed", sans-serif; font-weight: 600;
  font-size: 2rem; line-height: 1; color: #1F2A33; letter-spacing: .01em; }
.c-sub { display: block; font-size: .9rem; color: #52606B; margin-top: .25rem; }
.c-k { display: block; font-size: .75rem; color: #52606B; }
.c-v { display: block; font-family: "Barlow Condensed", sans-serif;
  font-size: 1.2rem; font-weight: 500; color: #1F2A33; margin-top: .15rem; }
@media (max-width: 700px) {
  .cartouche { grid-template-columns: 1fr 1fr; }
  .cartouche > div:first-child { grid-column: 1 / -1; border-bottom: 1px solid #1F2A33; }
  .cartouche > div:nth-child(2) { border-left: none; }
  .cartouche > div:nth-child(4) { grid-column: 1 / -1; border-left: none;
    border-top: 1px solid #1F2A33; }
}

/* Références : étiquettes à cases (document | page | élément) */
.refs { display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .2rem; }
.ref { display: inline-flex; border: 1px solid #1D5C8C; font-size: .8rem;
  background: #FFFFFF; color: #1F2A33; }
.ref span { padding: .12rem .45rem; border-left: 1px solid #C9D1D7; }
.ref span:first-child { border-left: none; color: #1D5C8C; font-weight: 500;
  max-width: 22ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ref .score { color: #52606B; }

/* Refus de sécurité : seul usage du jaune signalisation */
.refusal { border: 1px solid #1F2A33; background: #FFFBEA; display: flex;
  margin-top: .4rem; font-size: .9rem; }
.refusal::before { content: ""; flex: 0 0 12px;
  background: repeating-linear-gradient(-45deg, #F2B705 0 6px, #1F2A33 6px 12px); }
.refusal p { margin: 0; padding: .45rem .75rem; }

.empty { color: #52606B; margin: .5rem 0 1rem; max-width: 60ch; }
/* Bulles de conversation */
.q-row { display: flex; align-items: center; gap: .75rem; margin: 1.4rem 0 .9rem; }
/* avatar du technicien (statique) */
.q-avatar { flex: 0 0 52px; height: 52px;
  background: url("data:image/svg+xml;base64,{USER}") center / contain no-repeat; }
.q-bubble { background: #1D5C8C; color: #FFFFFF; padding: .6rem 1.2rem;
  border-radius: 22px; border-bottom-left-radius: 6px; max-width: 70%;
  font-size: 1rem; line-height: 1.45; letter-spacing: .01em;
  box-shadow: 0 2px 6px rgba(29, 92, 140, .25); }
[class*="st-key-answer-"] { margin-left: auto; margin-right: 64px; max-width: 86%;
  position: relative; overflow: visible;
  border: 1.5px solid #1D5C8C; border-radius: 22px; border-top-right-radius: 6px;
  padding: .9rem 1.3rem 1rem; background: transparent; }
/* avatar de l'assistant (robot animé), à droite de la bulle */
[class*="st-key-answer-"]::after { content: ""; position: absolute; top: 0; right: -64px;
  width: 52px; height: 52px;
  background: url("data:image/svg+xml;base64,{BOT_SMALL}") center / contain no-repeat; }
.st-key-answer-pending::after { background-image:
  url("data:image/svg+xml;base64,{BOT_THINKING}"); }
[class*="st-key-answer-"] p, [class*="st-key-answer-"] li { color: #1F2A33; }
[class*="st-key-answer-"] h1, [class*="st-key-answer-"] h2,
[class*="st-key-answer-"] h3 { font-size: 1.25rem; padding: .6rem 0 .2rem; }
.st-key-answer-pending { width: fit-content; padding: .6rem 1.3rem; }
.st-key-examples button { min-height: 0; padding: .3rem 0; justify-content: flex-start; }
</style>
""".replace("{USER}", user_b64())
   .replace("{BOT_SMALL}", mascot_b64("idle"))
   .replace("{BOT_THINKING}", mascot_b64("thinking")))

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []

def _label(doc_name: str) -> str:
    text = doc_name.replace("_", " ")
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.subheader(":material/folder_open: Documents")
    pdf = st.file_uploader("Ajouter un PDF", type="pdf")
    captions = st.checkbox("Décrire les images avec le modèle de vision", value=True,
                           help="Plus lent, mais les figures deviennent interrogeables.")
    if pdf and st.button("Indexer le document", type="primary", icon=":material/upload_file:",
                         width="stretch"):
        dest = Path(PDF_DIR) / pdf.name          # garde le nom d'origine pour les citations
        dest.write_bytes(pdf.getbuffer())
        from maintidoc.ingestion.run import ingest
        with st.status("Indexation en cours", expanded=False) as status:
            stats = ingest(str(dest), with_captions=captions)
            status.update(label="Document indexé", state="complete")
            st.json(stats)

    from maintidoc.retrieval.stores import list_documents
    try:
        available = list_documents()
    except Exception as e:                      # base non démarrée
        available = []
        st.error(f"Base PGVector inaccessible. Lancez `docker compose up -d`.\n\n{e}")
    st.caption("Documents interrogés")
    doc_names = [d for d in available
                 if st.checkbox(_label(d), value=True, key=f"doc_{d}")]
    if available and not doc_names:
        st.warning("Sélectionnez au moins un document pour poser une question.",
                   icon=":material/info:")

    st.subheader(":material/tune: Mode de réponse")
    mode = st.radio("Mode de réponse", list(MODES), captions=list(MODES.values()),
                    label_visibility="collapsed")
    if st.button("Nouvelle conversation", icon=":material/restart_alt:",
                 width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []

    st.divider()
    st.caption(":material/shield: Aide documentaire uniquement. MaintiDoc ne pilote aucun "
               "équipement et ne remplace pas un technicien habilité.")

# ------------------------------------------------------------------------ en-tête
n_questions = sum(m["role"] == "user" for m in st.session_state.history)
st.html(f"""
<div class="cartouche">
  <div><span class="c-name">MaintiDoc</span>
       <span class="c-sub">Assistant documentaire de maintenance</span></div>
  <div><span class="c-k">Documents interrogés</span>
       <span class="c-v">{len(doc_names)} sur {len(available)}</span></div>
  <div><span class="c-k">Mode</span><span class="c-v">{html.escape(mode)}</span></div>
  <div><span class="c-k">Questions posées</span><span class="c-v">{n_questions}</span></div>
</div>
""")


def show_sources(r: dict):
    if r.get("status") == "refused":
        st.html('<div class="refusal"><p>Demande d\'action refusée par le garde-fou, '
                'avant tout appel au modèle.</p></div>')
        return
    queries = r.get("search_queries") or []
    if len(queries) > 1:
        st.caption(f":material/translate: Recherche aussi en anglais : *{queries[-1]}*")
    if r.get("tools_used"):
        tools = ", ".join(dict.fromkeys(t.replace("_", " ") for t in r["tools_used"]))
        st.caption(f":material/manage_search: Outils consultés : {tools}")
    if r.get("sources"):
        chips = "".join(
            f'<span class="ref"><span title="{html.escape(str(s.get("source")))}">'
            f'{html.escape(str(s.get("source")))}</span><span>p.{s.get("page")}</span>'
            f'<span>{html.escape(str(s.get("element_id", s.get("type"))))}</span>'
            f'<span class="score">{(s.get("score") or 0):.2f}</span></span>'
            for s in r["sources"][:6])
        st.html(f'<div class="refs">{chips}</div>')
    images = [im for im in r.get("images", []) if Path(im["image_path"]).exists()]
    if images:
        for col, im in zip(st.columns(len(images)), images):
            label = im.get("figure_label") or im["element_id"]
            col.image(im["image_path"], caption=f"{label}, page {im['page']}",
                      width="stretch")


# --------------------------------------------------------------------- conversation
def show_question(text: str):
    st.html(f'<div class="q-row"><div class="q-avatar"></div>'
            f'<div class="q-bubble">{html.escape(text)}</div></div>')


def show_answer(i: int, msg: dict):
    with st.container(key=f"answer-{i}"):
        st.markdown(msg["content"])
        if msg.get("extra"):
            show_sources(msg["extra"])


for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        show_question(msg["content"])
    else:
        show_answer(i, msg)

pending = None
if not st.session_state.history:
    st.html('<p class="empty">Posez une question sur les documents sélectionnés. '
            'Chaque réponse cite le document, la page et l\'élément utilisés '
            '(passage, tableau ou figure).</p>')
    examples = [q for d in doc_names for q in EXAMPLES.get(d, [])]
    if examples:
        st.caption("Exemples de questions")
    with st.container(gap=None, key="examples"):
        for i, ex in enumerate(examples):
            if st.button(ex, key=f"ex{i}", icon=":material/help_outline:", type="tertiary"):
                pending = ex

q = st.chat_input("Posez votre question technique", disabled=not doc_names) or pending
if q:
    st.session_state.history.append({"role": "user", "content": q})
    show_question(q)
    with st.container(key="answer-pending"):
        with st.spinner("Recherche dans la documentation"):
            if mode == "Agent":
                from maintidoc.agent.agent import ask
                r = ask(q, thread_id=st.session_state.session_id, doc_names=doc_names)
            else:
                from maintidoc.rag.pipeline import answer
                r = answer(q, doc_names=doc_names)
    st.session_state.history.append({"role": "assistant", "content": r["answer"], "extra": r})
    st.rerun()              # affiche la réponse, met à jour le cartouche, masque les exemples
