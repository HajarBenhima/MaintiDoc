"""Agent RAG LangChain : choix des outils, mémoire de conversation, réponse sourcée."""
from functools import lru_cache

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.agent.tools import TOOLS
from maintidoc.config import OLLAMA_BASE_URL, OLLAMA_LLM, load_prompt

SYSTEM = load_prompt("agent_v1")["system"]


@lru_cache
def get_agent():
    llm = ChatOllama(model=OLLAMA_LLM, base_url=OLLAMA_BASE_URL, temperature=0)
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM,
                        checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "default") -> dict:
    res = get_agent().invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    msgs = res["messages"]
    # outils appelés pendant CE tour (après le dernier message utilisateur)
    last_user = max(i for i, m in enumerate(msgs) if m.type == "human")
    tools_used = [tc["name"] for m in msgs[last_user:] for tc in (getattr(m, "tool_calls", None) or [])]
    return {"answer": msgs[-1].content, "tools_used": tools_used}
