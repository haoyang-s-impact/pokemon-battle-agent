import asyncio
import importlib
import sys
import types
import unittest
from collections import defaultdict


class RecordingGraph:
    def __init__(self):
        self.active_by_session = defaultdict(int)
        self.max_by_session = defaultdict(int)
        self.total_active = 0
        self.max_total_active = 0

    async def ainvoke(self, payload, config):
        session_id = config["configurable"]["thread_id"]
        self.active_by_session[session_id] += 1
        self.max_by_session[session_id] = max(
            self.max_by_session[session_id],
            self.active_by_session[session_id],
        )
        self.total_active += 1
        self.max_total_active = max(self.max_total_active, self.total_active)
        try:
            await asyncio.sleep(0.01)
            return {"answer": payload["query"]}
        finally:
            self.total_active -= 1
            self.active_by_session[session_id] -= 1


class RecordingEngine:
    def __init__(self, session_id, shared_activity=None):
        self.session_id = session_id
        self.shared_activity = shared_activity
        self.active_by_session = defaultdict(int)
        self.max_by_session = defaultdict(int)
        self.total_active = 0
        self.max_total_active = 0

    async def achat(self, query):
        self.active_by_session[self.session_id] += 1
        self.max_by_session[self.session_id] = max(
            self.max_by_session[self.session_id],
            self.active_by_session[self.session_id],
        )
        self.total_active += 1
        self.max_total_active = max(self.max_total_active, self.total_active)
        if self.shared_activity is not None:
            self.shared_activity["active"] += 1
            self.shared_activity["max_active"] = max(
                self.shared_activity["max_active"],
                self.shared_activity["active"],
            )
        try:
            await asyncio.sleep(0.01)
            return query
        finally:
            if self.shared_activity is not None:
                self.shared_activity["active"] -= 1
            self.total_active -= 1
            self.active_by_session[self.session_id] -= 1


def _install_shared_indexer_stub(original_modules):
    indexer = types.ModuleType("src.doc_assistant.shared.indexer")
    indexer.retrieve = lambda query, top_k=5: [query]
    indexer.retrieve_abilities = lambda query, top_k=5: [query]
    indexer.get_index = lambda: None
    indexer.get_abilities_index = lambda: None
    original_modules["src.doc_assistant.shared.indexer"] = sys.modules.get(
        "src.doc_assistant.shared.indexer"
    )
    sys.modules["src.doc_assistant.shared.indexer"] = indexer


def _install_langchain_openai_stub(original_modules):
    module = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            return types.SimpleNamespace(content="ok")

    module.ChatOpenAI = ChatOpenAI
    original_modules["langchain_openai"] = sys.modules.get("langchain_openai")
    sys.modules["langchain_openai"] = module


def _install_llamaindex_stubs(original_modules):
    module_names = [
        "llama_index",
        "llama_index.core",
        "llama_index.core.memory",
        "llama_index.core.query_engine",
        "llama_index.core.selectors",
        "llama_index.core.tools",
        "llama_index.llms",
        "llama_index.llms.openai",
    ]
    for name in module_names:
        original_modules[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        if name in {"llama_index", "llama_index.core", "llama_index.llms"}:
            module.__path__ = []
        sys.modules[name] = module

    class ChatMemoryBuffer:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class RouterQueryEngine:
        def __init__(self, *args, **kwargs):
            pass

        async def aquery(self, query):
            return query

    class LLMSingleSelector:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    class QueryEngineTool:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    sys.modules["llama_index.core.memory"].ChatMemoryBuffer = ChatMemoryBuffer
    sys.modules["llama_index.llms.openai"].OpenAI = OpenAI
    sys.modules["llama_index.core.query_engine"].RouterQueryEngine = RouterQueryEngine
    sys.modules["llama_index.core.selectors"].LLMSingleSelector = LLMSingleSelector
    sys.modules["llama_index.core.tools"].QueryEngineTool = QueryEngineTool


def _restore_modules(original_modules):
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class SessionSerializationTests(unittest.TestCase):
    def tearDown(self):
        for name in [
            "src.doc_assistant.langgraph_assistant",
            "src.doc_assistant.llamaindex_assistant",
        ]:
            sys.modules.pop(name, None)

    def test_langgraph_serializes_same_session_but_not_different_sessions(self):
        original_modules = {}
        _install_shared_indexer_stub(original_modules)
        _install_langchain_openai_stub(original_modules)
        try:
            module = importlib.import_module("src.doc_assistant.langgraph_assistant")
            graph = RecordingGraph()
            module.get_graph = lambda: graph

            async def run_same_session():
                return await asyncio.gather(
                    module.ask("first", session_id="same"),
                    module.ask("second", session_id="same"),
                )

            self.assertEqual(asyncio.run(run_same_session()), ["first", "second"])
            self.assertEqual(graph.max_by_session["same"], 1)

            graph = RecordingGraph()
            module.get_graph = lambda: graph

            async def run_different_sessions():
                return await asyncio.gather(
                    module.ask("first", session_id="one"),
                    module.ask("second", session_id="two"),
                )

            self.assertEqual(asyncio.run(run_different_sessions()), ["first", "second"])
            self.assertGreaterEqual(graph.max_total_active, 2)
        finally:
            _restore_modules(original_modules)

    def test_llamaindex_serializes_same_session_but_not_different_sessions(self):
        original_modules = {}
        _install_shared_indexer_stub(original_modules)
        _install_llamaindex_stubs(original_modules)
        try:
            module = importlib.import_module("src.doc_assistant.llamaindex_assistant")
            engines = {}
            shared_activity = {"active": 0, "max_active": 0}

            def get_engine(session_id):
                engines.setdefault(
                    session_id,
                    RecordingEngine(session_id, shared_activity=shared_activity),
                )
                return engines[session_id]

            module._get_engine = get_engine

            async def run_same_session():
                return await asyncio.gather(
                    module.ask("first", session_id="same"),
                    module.ask("second", session_id="same"),
                )

            self.assertEqual(asyncio.run(run_same_session()), ["first", "second"])
            self.assertEqual(engines["same"].max_by_session["same"], 1)

            engines = {}
            module._get_engine = get_engine

            async def run_different_sessions():
                return await asyncio.gather(
                    module.ask("first", session_id="one"),
                    module.ask("second", session_id="two"),
                )

            self.assertEqual(asyncio.run(run_different_sessions()), ["first", "second"])
            self.assertGreaterEqual(shared_activity["max_active"], 2)
        finally:
            _restore_modules(original_modules)


if __name__ == "__main__":
    unittest.main()
