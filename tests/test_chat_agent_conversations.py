import json
import sys
import uuid
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents import chat_agent as chat_module  # noqa: E402
from app import smart_intent_system  # noqa: E402
from app.agents.chat_agent import ChatAgentState, chat_agent_graph, ensure_pydantic_state  # noqa: E402


def _extract_last_text(messages):
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, tuple):
        return last[-1]
    if isinstance(last, dict):
        return last.get("content", "")
    if hasattr(last, "content"):
        return last.content
    return str(last)


class DummyChatLLM:
    def __init__(self, reply_prefix: str = "stub-chat", *_, **__) -> None:
        self.reply_prefix = reply_prefix

    async def ainvoke(self, messages, **kwargs):
        text = _extract_last_text(messages)
        return SimpleNamespace(content=f"{self.reply_prefix}: {text}")

    def invoke(self, messages, **kwargs):
        text = _extract_last_text(messages)
        return SimpleNamespace(content=f"{self.reply_prefix}: {text}")


class DummyClarifyLLM:
    async def ainvoke(self, messages, **kwargs):
        payload = {
            "question": "Можете уточнить, что именно хотите найти?",
            "reason": "stub-clarify",
            "guidance_type": "question",
            "conversation_flow": "needs_clarification",
        }
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def stub_chat_dependencies(monkeypatch):
    from app.services import analytics_service as analytics_module
    from app.services import recommendation_service as recommendation_module
    from app.services import search_service as simple_module
    from app import smart_transition_system

    monkeypatch.setattr(chat_module, "ChatOpenAI", DummyChatLLM)
    monkeypatch.setattr(chat_module, "llm", DummyChatLLM())
    monkeypatch.setattr(chat_module, "clarify_llm", DummyClarifyLLM())

    smart_intent_system.enhanced_intent_detector.llm = DummyChatLLM()

    async def fake_simple_search(session_id: str, message: str):
        return {
            "response": f"search results for {message}",
            "results": [
                {
                    "title": "Dune",
                    "author": "Frank Herbert",
                    "content": "Stub summary",
                }
            ],
            "analysis": "stub-search-analysis",
            "intent": "search",
            "performance_metrics": {"stub": True},
        }

    async def fake_recommendations(session_id: str, message: str, conversation_history):
        return {
            "response": f"Вот подборка: {message}",
            "results": [
                {
                    "books": [
                        {
                            "title": "Foundation",
                            "author": "Isaac Asimov",
                            "reasoning": "Классика научной фантастики",
                        }
                    ]
                }
            ],
            "analysis": "stub-recommendation",
            "recommendation_stats": {"total": 1},
            "performance_metrics": {"stub": True},
        }

    async def fake_analytics(session_id: str, message: str, context=None):
        return {
            "response": f"Статистика по запросу: {message}",
            "analytics_stats": {"top_genre": "sci-fi"},
            "results": [
                {
                    "title": "Dune",
                    "author": "Frank Herbert",
                }
            ],
            "performance_metrics": {"stub": True},
        }

    monkeypatch.setattr(simple_module, "process_simple_search", fake_simple_search)
    monkeypatch.setattr(chat_module, "process_simple_search", fake_simple_search)

    monkeypatch.setattr(
        recommendation_module, "process_smart_recommendations", fake_recommendations
    )
    monkeypatch.setattr(chat_module, "process_smart_recommendations", fake_recommendations)

    monkeypatch.setattr(analytics_module, "process_smart_analytics", fake_analytics)
    monkeypatch.setattr(chat_module, "process_smart_analytics", fake_analytics)

    # Ensure formatter and analytics LLMs never make network calls if accessed
    monkeypatch.setattr(simple_module, "formatter_llm", DummyChatLLM())
    monkeypatch.setattr(analytics_module, "_analytics_llm", DummyChatLLM("stub-analytics"))

    def fake_decide_transition(self, current_node, intent_analysis, state_context):
        intent = (intent_analysis.get("intent") or "CHAT").upper()
        transition_type = intent_analysis.get("transition_type", "direct")

        if current_node == "recommendations":
            return {
                "next_node": "END",
                "transition_type": "handoff",
                "reasoning": "stub: recommendations complete the turn",
            }

        if current_node == "simple_search":
            return {
                "next_node": "END",
                "transition_type": transition_type,
                "reasoning": "stub: search completes conversation",
            }

        if current_node == "intent":
            mapping = {
                "RECOMMEND": "recommendations",
                "SEARCH": "simple_search",
                "ANALYTICS": "analytics",
                "CLARIFY": "clarify",
            }
            next_node = mapping.get(intent, "chat_response")
            return {
                "next_node": next_node,
                "transition_type": transition_type,
                "reasoning": f"stub: route to {next_node}",
            }

        return {
            "next_node": "chat_response",
            "transition_type": transition_type,
            "reasoning": "stub: default to chat",
        }

    monkeypatch.setattr(
        chat_module.smart_transition_manager,
        "decide_transition",
        MethodType(fake_decide_transition, chat_module.smart_transition_manager),
    )
    monkeypatch.setattr(
        smart_transition_system.smart_transition_manager,
        "decide_transition",
        MethodType(fake_decide_transition, smart_transition_system.smart_transition_manager),
    )


def make_intent_meta(intent: str, **overrides):
    base = {
        "intent": intent,
        "confidence": 0.9,
        "reason": f"stub-{intent.lower()}",
        "needs_clarification": False,
        "clarification_question": None,
        "context_summary": "",
        "conversation_stage": "exploring",
        "focus_entities": [],
        "transition_type": "direct",
    }
    base.update(overrides)
    return base


def build_recommendation_script():
    return {
        "name": "recommendation_flow",
        "turns": [
            {
                "user": "Привет! Я здесь впервые.",
                "detector": {
                    "intent": "CHAT",
                    "meta": make_intent_meta("CHAT", confidence=0.88, reason="greeting"),
                },
                "expected": {
                    "result_intent": "chat",
                    "reply_contains": "stub-chat",
                    "last_agent": "chat",
                },
            },
            {
                "user": "Посоветуй научно-фантастические книги в стиле Азимова.",
                "detector": {
                    "intent": "RECOMMEND",
                    "meta": make_intent_meta(
                        "RECOMMEND",
                        confidence=0.93,
                        reason="user asked for recommendations",
                        focus_entities=["фантастика"],
                    ),
                },
                "expected": {
                    "result_intent": "smart_recommendations",
                    "reply_contains": "Вот подборка",
                    "last_agent": "recommendations",
                    "books_expected": 1,
                },
            },
        ],
    }


def build_search_script():
    return {
        "name": "search_flow",
        "turns": [
            {
                "user": "Добрый день!",
                "detector": {
                    "intent": "CHAT",
                    "meta": make_intent_meta("CHAT", confidence=0.87, reason="small talk"),
                },
                "expected": {
                    "result_intent": "chat",
                    "reply_contains": "stub-chat",
                    "last_agent": "chat",
                },
            },
            {
                "user": "Найди факты про роман \"Дюна\".",
                "detector": {
                    "intent": "SEARCH",
                    "meta": make_intent_meta(
                        "SEARCH",
                        confidence=0.95,
                        reason="looking for facts about a specific title",
                    ),
                },
                "expected": {
                    "result_intent": "search",
                    "reply_contains": "search results for",
                    "last_agent": "search",
                    "search_results_length": 1,
                },
            },
        ],
    }


def build_clarify_script():
    return {
        "name": "clarify_multilingual_flow",
        "turns": [
            {
                "user": "Hola, necesito un livre about космос, maybe?",
                "detector": {
                    "intent": "CLARIFY",
                    "meta": make_intent_meta(
                        "CLARIFY",
                        confidence=0.5,
                        reason="ambiguous multilingual request",
                        needs_clarification=True,
                        clarification_question="Что именно вы хотите прочитать?",
                        transition_type="clarification",
                    ),
                },
                "expected": {
                    "result_intent": "clarify",
                    "reply_contains": "уточнить",
                    "last_agent": "clarify",
                    "clarification_attempts": 1,
                },
            }
        ],
    }


async def run_script(monkeypatch, script):
    intent_sequence = deque()
    for turn in script["turns"]:
        intent_data = {
            "intent": turn["detector"]["intent"],
            "meta": make_intent_meta(turn["detector"]["intent"]),
        }
        intent_data["meta"].update(turn["detector"].get("meta", {}))
        intent_sequence.append(intent_data)

    async def fake_detect(message, session_id, conversation_history=None, metadata=None, checkpointer=None):
        if not intent_sequence:
            raise AssertionError(f"Unexpected intent detection call in {script['name']}")
        step = intent_sequence.popleft()
        chat_module.enhanced_intent_detector.last_result = step["meta"]
        smart_intent_system.enhanced_intent_detector.last_result = step["meta"]
        return step["intent"]

    monkeypatch.setattr(chat_module.enhanced_intent_detector, "detect_intent", fake_detect)
    monkeypatch.setattr(smart_intent_system.enhanced_intent_detector, "detect_intent", fake_detect)

    thread_id = f"{script['name']}-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    states = []

    for index, turn in enumerate(script["turns"]):
        state = ChatAgentState(
            session_id=thread_id,
            messages=[HumanMessage(content=turn["user"])],
        )
        result = await chat_agent_graph.ainvoke(state, config=config)
        new_state = ensure_pydantic_state(result)
        states.append(new_state)

        expected = turn["expected"]
        result_payload = new_state.results[0]

        assert (
            result_payload.get("intent") == expected["result_intent"]
        ), (
            f"{script['name']} turn {index + 1}: expected intent "
            f"{expected['result_intent']}, got {result_payload.get('intent')}"
        )

        if "reply_contains" in expected:
            assert expected["reply_contains"] in result_payload.get("message", ""), (
                f"{script['name']} turn {index + 1}: reply missing fragment "
                f"'{expected['reply_contains']}'"
            )

        if "last_agent" in expected:
            assert new_state.last_agent == expected["last_agent"], (
                f"{script['name']} turn {index + 1}: expected last_agent "
                f"{expected['last_agent']}, got {new_state.last_agent}"
            )

        if expected.get("books_expected") is not None:
            assert (
                len(result_payload.get("books", [])) == expected["books_expected"]
            ), (
                f"{script['name']} turn {index + 1}: expected {expected['books_expected']} books"
            )


        if expected.get("clarification_attempts") is not None:
            assert (
                new_state.clarification_attempts == expected["clarification_attempts"]
            ), (
                f"{script['name']} turn {index + 1}: expected clarification attempts "
                f"{expected['clarification_attempts']}, got {new_state.clarification_attempts}"
            )

        if expected.get("search_results_length") is not None:
            assert (
                len(new_state.search_results) == expected["search_results_length"]
            ), (
                f"{script['name']} turn {index + 1}: expected search results length "
                f"{expected['search_results_length']}"
            )

    assert not intent_sequence, f"{script['name']} left intent steps unused"
    return states


@pytest.mark.asyncio
async def test_conversation_recommendations(monkeypatch):
    states = await run_script(monkeypatch, build_recommendation_script())
    final_state = states[-1]
    assert final_state.last_agent == "recommendations"
    assert len(final_state.recommendation_results) == 1
    assert final_state.recommendation_results[0]["books"][0]["title"] == "Foundation"


@pytest.mark.asyncio
async def test_conversation_search(monkeypatch):
    states = await run_script(monkeypatch, build_search_script())
    final_state = states[-1]
    assert final_state.last_agent == "search"
    assert len(final_state.search_results) == 1
    assert final_state.search_results[0]["title"] == "Dune"


@pytest.mark.asyncio
async def test_conversation_multilingual_clarify(monkeypatch):
    states = await run_script(monkeypatch, build_clarify_script())
    final_state = states[-1]
    assert final_state.last_agent == "clarify"
    assert final_state.clarification_attempts == 1
