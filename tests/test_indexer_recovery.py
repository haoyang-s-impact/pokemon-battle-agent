import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_llama_index_stubs():
    llama_index = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    node_parser = types.ModuleType("llama_index.core.node_parser")
    embeddings = types.ModuleType("llama_index.embeddings")
    huggingface = types.ModuleType("llama_index.embeddings.huggingface")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class _Settings:
        embed_model = None
        llm = None

    core.SimpleDirectoryReader = _Dummy
    core.StorageContext = _Dummy
    core.VectorStoreIndex = _Dummy
    core.load_index_from_storage = lambda storage_context: _Dummy()
    core.Settings = _Settings
    node_parser.SentenceSplitter = _Dummy
    huggingface.HuggingFaceEmbedding = _Dummy

    sys.modules.setdefault("llama_index", llama_index)
    sys.modules.setdefault("llama_index.core", core)
    sys.modules.setdefault("llama_index.core.node_parser", node_parser)
    sys.modules.setdefault("llama_index.embeddings", embeddings)
    sys.modules.setdefault("llama_index.embeddings.huggingface", huggingface)


_install_llama_index_stubs()
indexer = importlib.import_module("src.doc_assistant.shared.indexer")


class IndexerRecoveryTest(unittest.TestCase):
    def setUp(self):
        indexer._index = None
        indexer._abilities_index = None

    def tearDown(self):
        indexer._index = None
        indexer._abilities_index = None

    def test_get_index_rebuilds_after_persisted_load_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Path(temp_dir) / "storage"
            storage.mkdir()
            (storage / "docstore.json").write_text("{not valid json")
            stale_file = storage / "stale-vector-store.json"
            stale_file.write_text("partial write")
            rebuilt_index = object()
            calls = []

            def fail_load(load_storage):
                calls.append(("load", load_storage))
                raise ValueError("corrupt persisted store")

            def rebuild():
                calls.append(("build", storage))
                (storage / "docstore.json").write_text("{}")
                return rebuilt_index

            with (
                mock.patch.object(indexer, "_STORAGE_DIR", storage),
                mock.patch.object(indexer, "_configure_indexing"),
                mock.patch.object(indexer, "_load_persisted_index", side_effect=fail_load),
                mock.patch.object(indexer, "_build_index", side_effect=rebuild),
            ):
                self.assertIs(indexer.get_index(), rebuilt_index)

            self.assertEqual(calls, [("load", storage), ("build", storage)])
            self.assertFalse(stale_file.exists())
            self.assertTrue((storage / ".index.lock").exists())

    def test_get_abilities_index_rebuilds_after_persisted_load_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Path(temp_dir) / "storage_abilities"
            storage.mkdir()
            (storage / "docstore.json").write_text("{not valid json")
            rebuilt_index = object()

            with (
                mock.patch.object(indexer, "_ABILITIES_STORAGE_DIR", storage),
                mock.patch.object(indexer, "_configure_indexing"),
                mock.patch.object(indexer, "_load_persisted_index", side_effect=ValueError("bad store")),
                mock.patch.object(indexer, "_build_abilities_index", return_value=rebuilt_index),
            ):
                self.assertIs(indexer.get_abilities_index(), rebuilt_index)

            self.assertTrue((storage / ".index.lock").exists())

    def test_configuration_failures_do_not_clear_existing_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Path(temp_dir) / "storage"
            storage.mkdir()
            docstore = storage / "docstore.json"
            docstore.write_text("{}")
            marker = storage / "keep-me.json"
            marker.write_text("valid existing store")

            with (
                mock.patch.object(indexer, "_STORAGE_DIR", storage),
                mock.patch.object(indexer, "_configure_indexing", side_effect=RuntimeError("model unavailable")),
                mock.patch.object(indexer, "_load_persisted_index") as load_index,
                mock.patch.object(indexer, "_build_index") as build_index,
            ):
                with self.assertRaises(RuntimeError):
                    indexer.get_index()

            load_index.assert_not_called()
            build_index.assert_not_called()
            self.assertTrue(docstore.exists())
            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
