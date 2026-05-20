import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


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


class IndexManifestTest(unittest.TestCase):
    def test_storage_is_current_requires_matching_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage = root / "storage"
            source = root / "data.md"
            storage.mkdir()
            (storage / "docstore.json").write_text("{}")
            source.write_text("old facts")

            indexer._write_manifest(storage, indexer._current_manifest([source]))
            self.assertTrue(indexer._storage_is_current(storage, [source]))

            source.write_text("new facts")
            self.assertFalse(indexer._storage_is_current(storage, [source]))

    def test_storage_is_current_requires_matching_index_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage = root / "storage"
            source = root / "data.md"
            storage.mkdir()
            (storage / "docstore.json").write_text("{}")
            source.write_text("facts")

            original_model = indexer._EMBED_MODEL_NAME
            try:
                indexer._EMBED_MODEL_NAME = "old-model"
                indexer._write_manifest(storage, indexer._current_manifest([source]))
                self.assertTrue(indexer._storage_is_current(storage, [source]))

                indexer._EMBED_MODEL_NAME = "new-model"
                self.assertFalse(indexer._storage_is_current(storage, [source]))
            finally:
                indexer._EMBED_MODEL_NAME = original_model

    def test_storage_without_manifest_is_not_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage = root / "storage"
            source = root / "data.md"
            storage.mkdir()
            (storage / "docstore.json").write_text("{}")
            source.write_text("facts")

            self.assertFalse(indexer._storage_is_current(storage, [source]))


if __name__ == "__main__":
    unittest.main()
