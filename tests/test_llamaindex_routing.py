import asyncio
import importlib
import sys
import types
import unittest


def _install_dependency_stubs():
    memory_mod = types.ModuleType("llama_index.core.memory")

    class _ChatMemoryBuffer:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    memory_mod.ChatMemoryBuffer = _ChatMemoryBuffer

    llms_openai_mod = types.ModuleType("llama_index.llms.openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    llms_openai_mod.OpenAI = _OpenAI

    query_engine_mod = types.ModuleType("llama_index.core.query_engine")

    class _RouterQueryEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    query_engine_mod.RouterQueryEngine = _RouterQueryEngine

    selectors_mod = types.ModuleType("llama_index.core.selectors")

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    selectors_mod.LLMSingleSelector = _LLMSingleSelector

    tools_mod = types.ModuleType("llama_index.core.tools")

    class _QueryEngineTool:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            return cls()

    tools_mod.QueryEngineTool = _QueryEngineTool

    llama_index_mod = types.ModuleType("llama_index")
    core_mod = types.ModuleType("llama_index.core")
    llms_mod = types.ModuleType("llama_index.llms")

    core_mod.memory = memory_mod
    core_mod.query_engine = query_engine_mod
    core_mod.selectors = selectors_mod
    core_mod.tools = tools_mod
    llms_mod.openai = llms_openai_mod
    llama_index_mod.core = core_mod
    llama_index_mod.llms = llms_mod

    sys.modules["llama_index"] = llama_index_mod
    sys.modules["llama_index.core"] = core_mod
    sys.modules["llama_index.core.memory"] = memory_mod
    sys.modules["llama_index.core.query_engine"] = query_engine_mod
    sys.modules["llama_index.core.selectors"] = selectors_mod
    sys.modules["llama_index.core.tools"] = tools_mod
    sys.modules["llama_index.llms"] = llms_mod
    sys.modules["llama_index.llms.openai"] = llms_openai_mod

    indexer_mod = types.ModuleType("src.doc_assistant.shared.indexer")
    indexer_mod.get_index = lambda: None
    indexer_mod.get_abilities_index = lambda: None
    sys.modules["src.doc_assistant.shared.indexer"] = indexer_mod


class _AsyncQueryEngine:
    def __init__(self, name, response, calls):
        self.name = name
        self.response = response
        self.calls = calls

    async def aquery(self, query):
        self.calls.append((self.name, query))
        return self.response


class LlamaIndexRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_dependency_stubs()
        sys.modules.pop("src.doc_assistant.llamaindex_assistant", None)
        cls.module = importlib.import_module("src.doc_assistant.llamaindex_assistant")

    def test_ability_queries_use_main_and_abilities_context(self):
        calls = []
        main_engine = _AsyncQueryEngine(
            "main",
            "Earthquake is a Ground-type physical move with 100 base power.",
            calls,
        )
        abilities_engine = _AsyncQueryEngine(
            "abilities",
            "Levitate makes the Pokemon immune to Ground-type attacks.",
            calls,
        )

        self.module._get_ability_engines = lambda: (main_engine, abilities_engine)

        def fail_router():
            raise AssertionError("single-selector router should not handle ability queries")

        self.module._get_router = fail_router

        query = "How does Levitate interact with Earthquake?"
        answer = asyncio.run(self.module.ask_with_routing(query))

        self.assertEqual(calls, [("main", query), ("abilities", query)])
        self.assertIn("Earthquake is a Ground-type physical move", answer)
        self.assertIn("Levitate makes the Pokemon immune", answer)

    def test_non_ability_queries_still_use_router(self):
        calls = []
        router = _AsyncQueryEngine("router", "Water is weak to Electric and Grass.", calls)

        self.module._get_router = lambda: router

        def fail_ability_engines():
            raise AssertionError("ability engines should not handle non-ability queries")

        self.module._get_ability_engines = fail_ability_engines

        query = "What types are super effective against Water?"
        answer = asyncio.run(self.module.ask_with_routing(query))

        self.assertEqual(calls, [("router", query)])
        self.assertEqual(answer, "Water is weak to Electric and Grass.")

    def test_combined_responses_remove_exact_duplicates(self):
        self.assertEqual(
            self.module._combine_responses(["same answer", "same answer", " other answer "]),
            "same answer\n\nother answer",
        )


if __name__ == "__main__":
    unittest.main()
