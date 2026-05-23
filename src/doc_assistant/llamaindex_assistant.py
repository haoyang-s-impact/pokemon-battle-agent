"""
LlamaIndex Pokemon Document Assistant -- barebone easiest path.

Architecture: index.as_chat_engine(chat_mode="condense_plus_context")
Internally this runs: condense follow-up -> retrieve -> synthesize
The developer writes ~5 lines; the framework handles the rest.

Multi-turn: ChatMemoryBuffer per session, managed via a dict keyed by session_id.

What LlamaIndex gives you for free:
  - Query condensing for follow-ups (automatic LLM call)
  - Retrieval + synthesis with sensible defaults
  - ~5 lines of application code for a working multi-turn RAG assistant
  - Built-in chat modes (condense_plus_context, context, react, etc.)

Where LlamaIndex falls short for this task:
  - Multi-session persistence is manual (dict of ChatMemoryBuffer)
  - The pipeline is opaque -- condense/retrieve/synthesize happen internally
  - Customizing individual steps requires callbacks or subclassing
"""

from __future__ import annotations

import asyncio
import logging

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

from src.doc_assistant.shared.indexer import get_index, get_abilities_index

logger = logging.getLogger(__name__)

_LLM = OpenAI(model="gpt-4o-mini", temperature=0)
_TOP_K = 5
_MEMORY_TOKEN_LIMIT = 3000

_SYSTEM_PROMPT = (
    "You are a Pokemon battle strategy assistant. Answer questions about "
    "Pokemon types, moves, and competitive strategy using the provided context. "
    "Be specific and cite data (base power, type effectiveness multipliers, etc.) "
    "when available. If the context doesn't contain enough information, say so."
)

_sessions: dict[str, ChatMemoryBuffer] = {}
_engines: dict[str, object] = {}

# Extension 1: meta-question detection (same logic, applied as pre-processing)
_META_PHRASES = ["what can you help me with", "what do you know", "help", "hello", "hi"]

_META_RESPONSE = (
    "I can help you with Pokemon competitive strategy! Ask me about:\n"
    "- Type effectiveness (e.g., 'What is super effective against Water?')\n"
    "- Move stats and usage (e.g., 'Tell me about Earthquake')\n"
    "- Battle strategies (e.g., 'How to beat a Dragon type?')\n"
    "- Pokemon abilities (e.g., 'What does Intimidate do?')\n"
    "- Competitive concepts (STAB, entry hazards, setup sweeping, etc.)"
)

_ABILITY_KEYWORDS = ["ability", "abilities", "intimidate", "levitate", "protean",
                     "drizzle", "drought", "regenerator", "magic guard", "speed boost"]


def _is_meta_question(query: str) -> bool:
    return query.strip().lower().rstrip("?!.") in _META_PHRASES


def _is_ability_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _ABILITY_KEYWORDS)


def _combine_responses(responses) -> str:
    """Preserve response order while dropping exact duplicate text."""
    combined = []
    seen = set()
    for response in responses:
        text = str(response).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        combined.append(text)
    return "\n\n".join(combined)


def _build_router_query_engine():
    """Extension 2: RouterQueryEngine over main + abilities indices."""
    main_index = get_index()
    abilities_index = get_abilities_index()

    main_tool = QueryEngineTool.from_defaults(
        query_engine=main_index.as_query_engine(similarity_top_k=_TOP_K, llm=_LLM),
        description="Covers Pokemon type chart, competitive moves, and battle strategies.",
    )
    abilities_tool = QueryEngineTool.from_defaults(
        query_engine=abilities_index.as_query_engine(similarity_top_k=_TOP_K, llm=_LLM),
        description="Covers Pokemon abilities (Intimidate, Levitate, weather abilities, etc.).",
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=_LLM),
        query_engine_tools=[main_tool, abilities_tool],
    )


_router_engine = None
_ability_engines = None


def _get_router():
    global _router_engine
    if _router_engine is None:
        _router_engine = _build_router_query_engine()
    return _router_engine


def _get_ability_engines():
    """Return query engines for deterministic main+abilities retrieval."""
    global _ability_engines
    if _ability_engines is None:
        main_index = get_index()
        abilities_index = get_abilities_index()
        _ability_engines = (
            main_index.as_query_engine(similarity_top_k=_TOP_K, llm=_LLM),
            abilities_index.as_query_engine(similarity_top_k=_TOP_K, llm=_LLM),
        )
    return _ability_engines


def _get_engine(session_id: str):
    """Return (or create) a chat engine for the given session."""
    if session_id not in _engines:
        memory = ChatMemoryBuffer.from_defaults(token_limit=_MEMORY_TOKEN_LIMIT)
        _sessions[session_id] = memory

        index = get_index()
        engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            similarity_top_k=_TOP_K,
            llm=_LLM,
            system_prompt=_SYSTEM_PROMPT,
        )
        _engines[session_id] = engine
    return _engines[session_id]


async def ask(query: str, session_id: str = "default") -> str:
    """Ask a question. Returns the answer string."""
    # Extension 1: skip retrieval for meta-questions
    if _is_meta_question(query):
        return _META_RESPONSE

    engine = _get_engine(session_id)
    response = await engine.achat(query)
    logger.info("  [llamaindex] response length=%d", len(str(response)))
    return str(response)


async def ask_with_routing(query: str) -> str:
    """Extension 2: ask using routed retrieval (no multi-turn memory)."""
    if _is_meta_question(query):
        return _META_RESPONSE

    if _is_ability_query(query):
        # Hybrid ability questions need move/type/strategy context from the
        # main corpus as well as focused details from the abilities index.
        main_engine, abilities_engine = _get_ability_engines()
        responses = await asyncio.gather(
            main_engine.aquery(query),
            abilities_engine.aquery(query),
        )
        return _combine_responses(responses)

    router = _get_router()
    response = await router.aquery(query)
    return str(response)
