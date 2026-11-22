"""Agent RAG LangChain : choix des outils, mémoire de conversation, réponse sourcée."""
import re
from functools import lru_cache

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.agent.tools import TOOLS
from maintidoc.config import load_prompt
from maintidoc.llm import get_llm
from maintidoc.safety.guard import is_action_request, refusal_response

PROMPT = load_prompt("agent_v2")
RETRY = ("Tu as répondu sans consulter les documents. Utilise d'abord un outil de recherche, "
         "puis réponds à la question : {question}")
_NEGATIVE = re.compile(r"\b(aucune?|introuvable|pas trouvé|ne (?:peux|sais) pas|"
                       r"pas d'information)\b", re.I)


def _called_tool(msgs) -> bool:
    return any(getattr(m, "tool_calls", None) for m in msgs)


def _needs_search(msgs, start: int) -> bool:
    """Vrai si ce tour n'a appelé aucun outil alors qu'il le fallait :
    première question de la conversation, ou réponse négative (« aucune figure… »)."""
    if _called_tool(msgs[start:]):
        return False
    return not _called_tool(msgs[:start]) or bool(_NEGATIVE.search(msgs[-1].content or ""))


@lru_cache
def get_agent():
    return create_agent(model=get_llm(0.0), tools=TOOLS, system_prompt=PROMPT["system"],
                        checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "default",
        doc_names: list[str] | None = None) -> dict:
    if is_action_request(question):          # garde-fou avant tout appel au LLM
        return refusal_response()
    config = {"configurable": {"thread_id": thread_id, "doc_names": doc_names},
              "recursion_limit": 12}
    msgs = get_agent().invoke({"messages": [{"role": "user", "content": question}]},
                              config=config)["messages"]
    # début de CE tour (dernier message utilisateur)
    start = max(i for i, m in enumerate(msgs) if m.type == "human")
    if _needs_search(msgs, start):
        # le modèle a répondu sans chercher : une relance avec rappel
        msgs = get_agent().invoke({"messages": [{"role": "user", "content":
                                   RETRY.format(question=question)}]}, config=config)["messages"]
    turn = msgs[start:]
    tools_used = [tc["name"] for m in turn for tc in (getattr(m, "tool_calls", None) or [])]
    sources, images = {}, {}
    for m in turn:
        art = (getattr(m, "artifact", None) or {}) if m.type == "tool" else {}
        for s in art.get("sources", []):
            key = (s.get("source"), s.get("page"), s.get("element_id"))
            if key not in sources or (s["score"] or 0) > (sources[key]["score"] or 0):
                sources[key] = s
        for im in art.get("images", []):
            images.setdefault(im.get("image_path"), im)
    return {"answer": msgs[-1].content, "status": "answered", "tools_used": tools_used,
            "sources": sorted(sources.values(), key=lambda s: s["score"] or 0, reverse=True),
            "images": [im for p, im in images.items() if p][:3],
            "prompt_version": PROMPT["version"]}
