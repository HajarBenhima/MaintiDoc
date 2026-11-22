"""Relance de l'agent quand il répond sans consulter les documents (sans LLM)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from maintidoc.agent.agent import _needs_search

CALL = AIMessage(content="", tool_calls=[{"name": "rechercher_texte", "args": {"query": "q"},
                                          "id": "1"}])


def test_first_question_without_tool():
    assert _needs_search([HumanMessage("Q ?"), AIMessage("Réponse.")], 0)


def test_tool_called_this_turn():
    msgs = [HumanMessage("Q ?"), CALL, ToolMessage("ctx", tool_call_id="1"), AIMessage("R.")]
    assert not _needs_search(msgs, 0)


def test_follow_up_from_memory_is_ok():
    msgs = [HumanMessage("Q ?"), CALL, ToolMessage("ctx", tool_call_id="1"), AIMessage("R p.28"),
            HumanMessage("À quelle page ?"), AIMessage("Page 28.")]
    assert not _needs_search(msgs, 4)


def test_negative_answer_without_search():
    msgs = [HumanMessage("Q ?"), CALL, ToolMessage("ctx", tool_call_id="1"), AIMessage("R."),
            HumanMessage("Décris la Figure 1"), AIMessage("Aucune figure nommée Figure 1.")]
    assert _needs_search(msgs, 4)
