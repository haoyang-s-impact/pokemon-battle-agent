import importlib
import sys
import types
import unittest


def _install_dependency_stubs():
    messages_mod = types.ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content=None, **kwargs):
            self.content = content
            self.kwargs = kwargs

    messages_mod.AIMessage = _Message
    messages_mod.BaseMessage = _Message
    messages_mod.HumanMessage = _Message
    messages_mod.SystemMessage = _Message

    langchain_core_mod = types.ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod

    langchain_openai_mod = types.ModuleType("langchain_openai")

    class _ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    langchain_openai_mod.ChatOpenAI = _ChatOpenAI
    sys.modules["langchain_openai"] = langchain_openai_mod

    graph_mod = types.ModuleType("langgraph.graph")
    graph_mod.END = "__end__"
    graph_mod.START = "__start__"

    def add_messages(left, right):
        return [*(left or []), *(right or [])]

    class _StateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):
            return object()

    graph_mod.add_messages = add_messages
    graph_mod.StateGraph = _StateGraph

    checkpoint_memory_mod = types.ModuleType("langgraph.checkpoint.memory")

    class _InMemorySaver:
        pass

    checkpoint_memory_mod.InMemorySaver = _InMemorySaver

    langgraph_mod = types.ModuleType("langgraph")
    checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    checkpoint_mod.memory = checkpoint_memory_mod
    langgraph_mod.graph = graph_mod
    langgraph_mod.checkpoint = checkpoint_mod

    sys.modules["langgraph"] = langgraph_mod
    sys.modules["langgraph.graph"] = graph_mod
    sys.modules["langgraph.checkpoint"] = checkpoint_mod
    sys.modules["langgraph.checkpoint.memory"] = checkpoint_memory_mod

    indexer_mod = types.ModuleType("src.doc_assistant.shared.indexer")
    indexer_mod.retrieve = lambda query, top_k=5: []
    indexer_mod.retrieve_abilities = lambda query, top_k=5: []
    sys.modules["src.doc_assistant.shared.indexer"] = indexer_mod


class LangGraphRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_dependency_stubs()
        sys.modules.pop("src.doc_assistant.langgraph_assistant", None)
        cls.module = importlib.import_module("src.doc_assistant.langgraph_assistant")

    def test_ability_queries_merge_main_and_ability_context(self):
        calls = []

        def shared_retrieve(query, top_k):
            calls.append(("main", query, top_k))
            return ["main type/move context", "duplicate ability context"]

        def retrieve_abilities(query, top_k):
            calls.append(("abilities", query, top_k))
            return ["duplicate ability context", "ability-only context"]

        self.module.shared_retrieve = shared_retrieve
        self.module.retrieve_abilities = retrieve_abilities

        result = self.module.retrieve_docs({
            "query": "How does Levitate interact with Earthquake?",
        })

        self.assertEqual(
            result["retrieved_chunks"],
            ["main type/move context", "duplicate ability context", "ability-only context"],
        )
        self.assertEqual(
            calls,
            [
                ("main", "How does Levitate interact with Earthquake?", 5),
                ("abilities", "How does Levitate interact with Earthquake?", 5),
            ],
        )

    def test_non_ability_queries_use_main_context_only(self):
        calls = []

        def shared_retrieve(query, top_k):
            calls.append(("main", query, top_k))
            return ["main context"]

        def retrieve_abilities(query, top_k):
            calls.append(("abilities", query, top_k))
            return ["ability context"]

        self.module.shared_retrieve = shared_retrieve
        self.module.retrieve_abilities = retrieve_abilities

        result = self.module.retrieve_docs({
            "query": "What types are super effective against Water?",
        })

        self.assertEqual(result["retrieved_chunks"], ["main context"])
        self.assertEqual(
            calls,
            [("main", "What types are super effective against Water?", 5)],
        )


if __name__ == "__main__":
    unittest.main()
