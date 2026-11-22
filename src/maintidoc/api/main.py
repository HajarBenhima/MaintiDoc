"""API FastAPI minimale (sera étendue aux étapes 3 et 4).

Lancement : uvicorn maintidoc.api.main:app --reload
"""
import shutil

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from maintidoc.config import PDF_DIR

app = FastAPI(title="MaintiDoc API", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    thread_id: str = "default"
    agentic: bool = True
    doc_names: list[str] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    from maintidoc.retrieval.stores import list_documents
    return list_documents()


@app.post("/ingest")
def ingest_pdf(file: UploadFile, with_captions: bool = True):
    from maintidoc.ingestion.run import ingest
    dest = PDF_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return ingest(str(dest), with_captions=with_captions)


@app.post("/ask")
def ask(req: AskRequest):
    if req.agentic:
        from maintidoc.agent.agent import ask as agent_ask
        return agent_ask(req.question, thread_id=req.thread_id, doc_names=req.doc_names)
    from maintidoc.rag.pipeline import answer
    return answer(req.question, doc_names=req.doc_names)
