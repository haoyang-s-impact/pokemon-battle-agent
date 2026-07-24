"""Regression: Settings.llm=None must not leave MockLLM prompt windows for synthesis."""

from __future__ import annotations

import importlib
import sys
import types
import unittest


def _install_import_stubs() -> None:
    """Avoid loading HuggingFace embeddings when importing the assistant module."""
    if "src.doc_assistant.shared.indexer" in sys.modules:
        return

    indexer = types.ModuleType("src.doc_assistant.shared.indexer")
    indexer.get_index = lambda: None
    indexer.get_abilities_index = lambda: None
    sys.modules["src.doc_assistant.shared.indexer"] = indexer


_install_import_stubs()

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.utils import get_prefix_messages_with_context
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.llms.openai import OpenAI

llamaindex_assistant = importlib.import_module("src.doc_assistant.llamaindex_assistant")


_CONTEXT_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


def _pollute_settings_with_mock_llm() -> None:
    """Reproduce the old indexer `Settings.llm = None` side effect."""
    Settings._prompt_helper = None
    Settings._chat_prompt_helper = None
    Settings.llm = None  # resolve_llm(None) → MockLLM, ~3900-token helper


def _history_near_memory_limit() -> list[ChatMessage]:
    msg = "Electric and Grass are super effective against Water. " * 20
    history: list[ChatMessage] = []
    # ~3000 content tokens — same regime as ChatMemoryBuffer(token_limit=3000)
    for _ in range(9):
        history.append(ChatMessage(role=MessageRole.USER, content=msg))
        history.append(ChatMessage(role=MessageRole.ASSISTANT, content=msg))
    return history


def _make_synth_with_history(history: list[ChatMessage]) -> CompactAndRefine:
    qa_messages = get_prefix_messages_with_context(
        PromptTemplate(_CONTEXT_PROMPT),
        llamaindex_assistant._SYSTEM_PROMPT,
        [],
        history,
        MessageRole.SYSTEM,
    )
    return CompactAndRefine(
        llm=OpenAI(model="gpt-4o-mini", api_key="sk-test"),
        callback_manager=CallbackManager([]),
        text_qa_template=ChatPromptTemplate.from_messages(qa_messages),
        refine_template=ChatPromptTemplate.from_messages(qa_messages),
    )


class SettingsPromptHelperTest(unittest.TestCase):
    def tearDown(self) -> None:
        Settings._llm = None
        Settings._prompt_helper = None
        Settings._chat_prompt_helper = None

    def test_configure_replaces_mock_llm_prompt_helper_window(self) -> None:
        _pollute_settings_with_mock_llm()
        self.assertIsInstance(Settings._llm, MockLLM)
        self.assertEqual(Settings.prompt_helper.context_window, 3900)

        llamaindex_assistant._configure_llama_settings()

        self.assertIs(Settings.llm, llamaindex_assistant._LLM)
        self.assertEqual(
            Settings.prompt_helper.context_window,
            llamaindex_assistant._LLM.metadata.context_window,
        )
        self.assertGreater(Settings.prompt_helper.context_window, 3900)

    def test_mock_llm_helper_crashes_near_memory_limit_but_configure_recovers(self) -> None:
        history = _history_near_memory_limit()
        chunks = ["Water resists Fire Steel Water. " * 40 for _ in range(5)]

        _pollute_settings_with_mock_llm()
        polluted = _make_synth_with_history(history)
        with self.assertRaises(ValueError):
            polluted._make_compact_text_chunks("What about its resistances?", chunks)

        llamaindex_assistant._configure_llama_settings()
        fixed = _make_synth_with_history(history)
        groups = fixed._make_compact_text_chunks("What about its resistances?", chunks)
        self.assertGreaterEqual(len(groups), 1)
        self.assertGreater(sum(len(g) for g in groups), 0)

    def test_indexer_no_longer_assigns_settings_llm_none(self) -> None:
        import ast
        from pathlib import Path

        source = Path("src/doc_assistant/shared/indexer.py").read_text()
        tree = ast.parse(source)
        assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "Settings"
                    and target.attr == "llm"
                ):
                    assignments.append(node)
        self.assertEqual(assignments, [], "indexer must not assign Settings.llm")


if __name__ == "__main__":
    unittest.main()
