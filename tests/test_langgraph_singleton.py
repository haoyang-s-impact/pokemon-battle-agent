import importlib
import sys
import threading
import time
import types
import unittest


_MISSING = object()


class LangGraphSingletonTests(unittest.TestCase):
    _STUBBED_MODULES = [
        "langchain_core",
        "langchain_core.messages",
        "langgraph",
        "langgraph.graph",
        "langgraph.checkpoint",
        "langgraph.checkpoint.memory",
        "langchain_openai",
        "src.doc_assistant.shared.indexer",
        "src.doc_assistant.langgraph_assistant",
    ]

    def setUp(self):
        self._original_modules = {
            name: sys.modules.get(name, _MISSING) for name in self._STUBBED_MODULES
        }
        for name in self._STUBBED_MODULES:
            sys.modules.pop(name, None)
        self._install_dependency_stubs()

    def tearDown(self):
        for name, module in self._original_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_dependency_stubs(self):
        messages_mod = types.ModuleType("langchain_core.messages")

        class _Message:
            def __init__(self, content=None, **kwargs):
                self.content = content

        messages_mod.AIMessage = type("AIMessage", (_Message,), {})
        messages_mod.BaseMessage = type("BaseMessage", (_Message,), {})
        messages_mod.HumanMessage = type("HumanMessage", (_Message,), {})
        messages_mod.SystemMessage = type("SystemMessage", (_Message,), {})
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
        sys.modules["langchain_core.messages"] = messages_mod

        graph_mod = types.ModuleType("langgraph.graph")
        graph_mod.add_messages = lambda *args, **kwargs: None
        graph_mod.END = "__end__"
        graph_mod.START = "__start__"

        class StateGraph:
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

        graph_mod.StateGraph = StateGraph
        sys.modules["langgraph"] = types.ModuleType("langgraph")
        sys.modules["langgraph.graph"] = graph_mod

        memory_mod = types.ModuleType("langgraph.checkpoint.memory")
        memory_mod.InMemorySaver = type("InMemorySaver", (), {})
        sys.modules["langgraph.checkpoint"] = types.ModuleType("langgraph.checkpoint")
        sys.modules["langgraph.checkpoint.memory"] = memory_mod

        openai_mod = types.ModuleType("langchain_openai")
        openai_mod.ChatOpenAI = type("ChatOpenAI", (), {})
        sys.modules["langchain_openai"] = openai_mod

        indexer_mod = types.ModuleType("src.doc_assistant.shared.indexer")
        indexer_mod.retrieve = lambda query, top_k=5: []
        indexer_mod.retrieve_abilities = lambda query, top_k=5: []
        sys.modules["src.doc_assistant.shared.indexer"] = indexer_mod

    def test_get_graph_builds_single_instance_for_concurrent_cold_start(self):
        module = importlib.import_module("src.doc_assistant.langgraph_assistant")
        module._graph = None

        build_calls = []
        build_calls_lock = threading.Lock()

        def slow_build_graph():
            time.sleep(0.05)
            graph = object()
            with build_calls_lock:
                build_calls.append(graph)
            return graph

        module.build_graph = slow_build_graph

        start = threading.Barrier(12)
        results = []
        errors = []

        def worker():
            try:
                start.wait(timeout=2)
                results.append(module.get_graph())
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertFalse(errors)
        self.assertEqual(12, len(results))
        self.assertEqual(1, len(build_calls))
        self.assertEqual(1, len({id(result) for result in results}))
        self.assertIs(module._graph, results[0])

