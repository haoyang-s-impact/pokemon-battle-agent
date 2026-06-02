"""
LangGraph Pokemon Document Assistant -- barebone easiest path.

Architecture: START -> retrieve -> generate -> END
State: messages (with checkpointer), query, retrieved_chunks, answer
Multi-turn: InMemorySaver checkpointer + thread_id. No query condensing --
follow-up questions retrieve against the raw string.

What LangGraph gives you for free:
  - Typed state schema enforced across nodes
  - Checkpointer-based persistence with zero application code
  - Graph topology as a first-class inspectable artifact
  - LangSmith tracing with node-level spans (via env vars)

Where LangGraph falls short for this task:
  - No automatic query condensing: follow-ups like "What about its resistances?"
    retrieve against the literal string, degrading retrieval quality
  - More ceremony than necessary for a 2-node linear pipeline
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.doc_assistant.shared.indexer import retrieve as shared_retrieve, retrieve_abilities

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Pokemon battle strategy assistant. Answer questions about "
    "Pokemon types, moves, and competitive strategy using the provided context. "
    "Be specific and cite data (base power, type effectiveness multipliers, etc.) "
    "when available. If the context doesn't contain enough information, say so."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_chunks: list[str]
    answer: str


# ---------------------------------------------------------------------------
# Extension 1: skip retrieval for meta-questions (control-flow extension)
# ---------------------------------------------------------------------------

_META_PHRASES = ["what can you help me with", "what do you know", "help", "hello", "hi"]

def _is_meta_question(query: str) -> bool:
    return query.strip().lower().rstrip("?!.") in _META_PHRASES


def direct_respond(state: AssistantState) -> dict:
    """Respond to meta-questions without retrieval or LLM call."""
    answer = (
        "I can help you with Pokemon competitive strategy! Ask me about:\n"
        "- Type effectiveness (e.g., 'What is super effective against Water?')\n"
        "- Move stats and usage (e.g., 'Tell me about Earthquake')\n"
        "- Battle strategies (e.g., 'How to beat a Dragon type?')\n"
        "- Competitive concepts (STAB, entry hazards, setup sweeping, etc.)"
    )
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=answer),
        ],
    }


def route_entry(state: AssistantState) -> str:
    """Route meta-questions to direct_respond, everything else to retrieve."""
    if _is_meta_question(state["query"]):
        return "direct_respond"
    return "retrieve"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_ABILITY_KEYWORDS = ["ability", "abilities", "intimidate", "levitate", "protean",
                     "drizzle", "drought", "regenerator", "magic guard", "speed boost"]

def _is_ability_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _ABILITY_KEYWORDS)


def retrieve_docs(state: AssistantState) -> dict:
    """Retrieve chunks from the shared index. No LLM call."""
    t0 = time.perf_counter()
    query = state["query"]
    # Extension 2: route ability queries to the abilities index
    if _is_ability_query(query):
        chunks = retrieve_abilities(query, top_k=5)
        logger.info("  [retrieve-abilities] query=%r  chunks=%d", query, len(chunks))
    else:
        chunks = shared_retrieve(query, top_k=5)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("  [retrieve] query=%r  chunks=%d  %.0fms", query, len(chunks), elapsed)
    return {"retrieved_chunks": chunks}


def generate_answer(state: AssistantState) -> dict:
    """Generate an answer using the LLM with retrieved context + full history."""
    t0 = time.perf_counter()

    context_block = "\n\n---\n\n".join(state["retrieved_chunks"])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=(
            f"[Retrieved context]\n{context_block}\n\n"
            f"[Question]\n{state['query']}"
        )),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("  [generate] %.0fms", elapsed)

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=response.content),
        ],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph assistant."""
    builder = StateGraph(AssistantState)
    builder.add_node("direct_respond", direct_respond)
    builder.add_node("retrieve", retrieve_docs)
    builder.add_node("generate", generate_answer)

    # Extension 1: conditional entry -- meta-questions skip retrieval
    builder.add_conditional_edges(START, route_entry, {
        "direct_respond": "direct_respond",
        "retrieve": "retrieve",
    })
    builder.add_edge("direct_respond", END)
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=InMemorySaver())


_graph = None
_session_locks: dict[str, asyncio.Lock] = {}


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def ask(query: str, session_id: str = "default") -> str:
    """Ask a question. Returns the answer string."""
    graph = get_graph()
    async with _get_session_lock(session_id):
        result = await graph.ainvoke(
            {"query": query},
            config={"configurable": {"thread_id": session_id}},
        )
    return result["answer"]
