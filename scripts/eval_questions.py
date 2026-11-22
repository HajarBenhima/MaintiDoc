"""Évalue les questions de référence (tests/questions_reference.yaml).

Usage :
    python scripts/eval_questions.py               # RAG simple + agent
    python scripts/eval_questions.py --mode agent  # agent seulement
    python scripts/eval_questions.py --ids Q1 Q3
    python scripts/eval_questions.py --questions tests/questions_pompe.yaml

Écrit un rapport Markdown dans reports/eval_<date>.md (réponses, sources,
outils utilisés, latence). Le verdict correct/incorrect reste à remplir à la main.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml

from maintidoc.config import OLLAMA_LLM, PROJECT_ROOT

QUESTIONS = PROJECT_ROOT / "tests" / "questions_reference.yaml"


def run(mode: str, question: str, doc_names: list[str] | None) -> dict:
    t0 = time.perf_counter()
    if mode == "agent":
        from maintidoc.agent.agent import ask
        r = ask(question, thread_id=f"eval-{time.time()}", doc_names=doc_names)
    else:
        from maintidoc.rag.pipeline import answer
        r = answer(question, doc_names=doc_names)
    r["latency_s"] = round(time.perf_counter() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "agent", "both"], default="both")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--questions", default=str(QUESTIONS), help="fichier YAML de questions")
    ap.add_argument("--docs", nargs="*", help="limiter la recherche à ces documents (doc_name)")
    args = ap.parse_args()

    qs = yaml.safe_load(open(args.questions, encoding="utf-8"))
    if args.ids:
        qs = [q for q in qs if q["id"] in args.ids]
    modes = ["rag", "agent"] if args.mode == "both" else [args.mode]

    lines = [f"# Évaluation — {Path(args.questions).name} — {datetime.now():%Y-%m-%d %H:%M}",
             f"\nModèle : `{OLLAMA_LLM}` — documents : {args.docs or 'tous'}\n"]
    for q in qs:
        lines.append(f"## {q['id']} — {q['question']}\n")
        lines.append(f"*Attendu :* {q['attendu']} — *source :* {q['source']}\n")
        for mode in modes:
            print(f"[{mode}] {q['id']} ...", flush=True)
            r = run(mode, q["question"], args.docs)
            lines.append(f"### {mode.upper()} ({r['latency_s']} s)\n")
            lines.append(r["answer"].strip() + "\n")
            if r.get("status") == "refused":
                lines.append("*Refus par le garde-fou (aucun appel au modèle).*\n")
            if r.get("search_queries"):
                lines.append(f"*Requêtes :* {' · '.join(r['search_queries'])}\n")
            if r.get("tools_used"):
                lines.append(f"*Outils :* {', '.join(r['tools_used'])}\n")
            if r.get("sources"):
                refs = [f"p.{s.get('page')} {s.get('element_id')} ({s['score']:.2f})"
                        for s in r["sources"]]
                lines.append(f"*Sources récupérées :* {'; '.join(refs)}\n")
            lines.append("*Verdict :* ☐ correct ☐ partiel ☐ incorrect\n")
    out = PROJECT_ROOT / "reports" / f"eval_{datetime.now():%Y%m%d_%H%M}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport : {out}")


if __name__ == "__main__":
    main()
