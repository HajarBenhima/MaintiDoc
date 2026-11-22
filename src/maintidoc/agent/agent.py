"""Agent RAG LangChain : choix des outils, mémoire de conversation, réponse sourcée."""
from functools import lru_cache

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.agent.tools import TOOLS
from maintidoc.config import load_prompt
from maintidoc.llm import get_llm
from maintidoc.safety.guard import is_action_request, refusal_response

PROMPT = load_prompt("agent_v2")


@lru_cache
def get_agent():
    return create_agent(model=get_llm(0.0), tools=TOOLS, system_prompt=PROMPT["system"],
                        checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "default",
        doc_names: list[str] | None = None) -> dict:
    if is_action_request(question):          # garde-fou avant tout appel au LLM
        return refusal_response()
    res = get_agent().invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id, "doc_names": doc_names},
                "recursion_limit": 12},
    )
    msgs = res["messages"]
    # outils appelés pendant CE tour (après le dernier message utilisateur)
    last_user = max(i for i, m in enumerate(msgs) if m.type == "human")
    tools_used = [tc["name"] for m in msgs[last_user:]
                  for tc in (getattr(m, "tool_calls", None) or [])]
    return {"answer": msgs[-1].content, "status": "answered", "tools_used": tools_used,
            "prompt_version": PROMPT["version"]}
