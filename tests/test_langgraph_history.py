import importlib
import sys
import types
import unittest
from typing import get_type_hints


class _FakeBaseMessage:
    def __init__(self, content=""):
        self.content = content


class _FakeHumanMessage(_FakeBaseMessage):
    pass


class _FakeAIMessage(_FakeBaseMessage):
    pass


class _FakeSystemMessage(_FakeBaseMessage):
    pass


class _FakeStateGraph:
    def __init__(self, *_args, **_kwargs):
        pass

    def add_node(self, *_args, **_kwargs):
        pass

    def add_conditional_edges(self, *_args, **_kwargs):
        pass

    def add_edge(self, *_args, **_kwargs):
        pass

    def compile(self, *_args, **_kwargs):
        return object()


def _install_dependency_stubs():
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.AIMessage = _FakeAIMessage
    messages_module.BaseMessage = _FakeBaseMessage
    messages_module.HumanMessage = _FakeHumanMessage
    messages_module.SystemMessage = _FakeSystemMessage

    langchain_core_module = types.ModuleType("langchain_core")
    langchain_core_module.messages = messages_module

    graph_module = types.ModuleType("langgraph.graph")
    graph_module.END = "END"
    graph_module.START = "START"
    graph_module.StateGraph = _FakeStateGraph
    graph_module.add_messages = lambda left, right: list(left or []) + list(right or [])

    checkpoint_memory_module = types.ModuleType("langgraph.checkpoint.memory")
    checkpoint_memory_module.InMemorySaver = object

    checkpoint_module = types.ModuleType("langgraph.checkpoint")
    checkpoint_module.memory = checkpoint_memory_module

    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.graph = graph_module
    langgraph_module.checkpoint = checkpoint_module

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.ChatOpenAI = object

    indexer_module = types.ModuleType("src.doc_assistant.shared.indexer")
    indexer_module.retrieve = lambda *_args, **_kwargs: []
    indexer_module.retrieve_abilities = lambda *_args, **_kwargs: []

    stubs = {
        "langchain_core": langchain_core_module,
        "langchain_core.messages": messages_module,
        "langgraph": langgraph_module,
        "langgraph.graph": graph_module,
        "langgraph.checkpoint": checkpoint_module,
        "langgraph.checkpoint.memory": checkpoint_memory_module,
        "langchain_openai": langchain_openai_module,
        "src.doc_assistant.shared.indexer": indexer_module,
    }
    sys.modules.update(stubs)


class LangGraphHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_dependency_stubs()
        sys.modules.pop("src.doc_assistant.langgraph_assistant", None)
        cls.module = importlib.import_module("src.doc_assistant.langgraph_assistant")

    def test_state_uses_bounded_message_reducer(self):
        hints = get_type_hints(self.module.AssistantState, include_extras=True)
        self.assertIs(hints["messages"].__metadata__[0], self.module._add_trimmed_messages)

    def test_message_reducer_keeps_recent_history_bounded(self):
        history = []
        all_messages = []
        for turn in range(self.module._MAX_HISTORY_MESSAGES + 5):
            new_messages = [
                self.module.HumanMessage(content=f"question-{turn}"),
                self.module.AIMessage(content=f"answer-{turn}"),
            ]
            all_messages.extend(new_messages)
            history = self.module._add_trimmed_messages(history, new_messages)

        expected = all_messages[-self.module._MAX_HISTORY_MESSAGES:]
        self.assertEqual([m.content for m in history], [m.content for m in expected])
        self.assertLessEqual(len(history), self.module._MAX_HISTORY_MESSAGES)


if __name__ == "__main__":
    unittest.main()
