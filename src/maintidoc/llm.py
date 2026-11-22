"""Accès au LLM local (Ollama) partagé par le RAG, l'agent et la reformulation."""
from functools import lru_cache

from langchain_ollama import ChatOllama

from maintidoc.config import LLM_NUM_CTX, OLLAMA_BASE_URL, OLLAMA_LLM


@lru_cache
def get_llm(temperature: float = 0.0, num_predict: int | None = None) -> ChatOllama:
    return ChatOllama(model=OLLAMA_LLM, base_url=OLLAMA_BASE_URL, temperature=temperature,
                      num_ctx=LLM_NUM_CTX, num_predict=num_predict)
