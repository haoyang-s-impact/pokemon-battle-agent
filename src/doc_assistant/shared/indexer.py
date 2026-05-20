"""
Shared vector index for the Pokemon document assistant.

Builds a single VectorStoreIndex from data/*.md, persists to storage/.
Exposes two interfaces so both assistants use identical retrieval:
  - get_index()  -> VectorStoreIndex   (native LlamaIndex object)
  - retrieve()   -> list[str]          (plain callable, framework-agnostic)
"""

import hashlib
import json
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_STORAGE_DIR = _PROJECT_ROOT / "storage"

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CHUNK_SIZE = 256
_CHUNK_OVERLAP = 32
_DEFAULT_TOP_K = 5
_MANIFEST_FILE = "index_manifest.json"
_MANIFEST_VERSION = 1

_index: VectorStoreIndex | None = None
_abilities_index: VectorStoreIndex | None = None

_ABILITIES_STORAGE_DIR = _PROJECT_ROOT / "storage_abilities"


def _source_file_entry(path: Path) -> dict:
    """Return a deterministic manifest entry for one source document."""
    content = path.read_bytes()
    try:
        manifest_path = path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        manifest_path = path.as_posix()

    return {
        "path": manifest_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _current_manifest(source_files: list[Path]) -> dict:
    """Describe the source data and index settings used to build storage."""
    return {
        "version": _MANIFEST_VERSION,
        "embed_model": _EMBED_MODEL_NAME,
        "chunk_size": _CHUNK_SIZE,
        "chunk_overlap": _CHUNK_OVERLAP,
        "source_files": [
            _source_file_entry(path)
            for path in sorted(source_files, key=lambda p: p.as_posix())
        ],
    }


def _manifest_path(storage_dir: Path) -> Path:
    return storage_dir / _MANIFEST_FILE


def _read_manifest(storage_dir: Path) -> dict | None:
    path = _manifest_path(storage_dir)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(storage_dir: Path, manifest: dict) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(storage_dir)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _storage_is_current(storage_dir: Path, source_files: list[Path]) -> bool:
    if not (storage_dir / "docstore.json").exists():
        return False
    return _read_manifest(storage_dir) == _current_manifest(source_files)


def _main_source_files() -> list[Path]:
    return list(_DATA_DIR.glob("*.md"))


def _build_index() -> VectorStoreIndex:
    Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    Settings.llm = None  # indexing does not need an LLM

    source_files = _main_source_files()
    documents = SimpleDirectoryReader(str(_DATA_DIR)).load_data()
    splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(_STORAGE_DIR))
    _write_manifest(_STORAGE_DIR, _current_manifest(source_files))
    return index


def _load_index() -> VectorStoreIndex:
    Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    Settings.llm = None

    storage_context = StorageContext.from_defaults(persist_dir=str(_STORAGE_DIR))
    return load_index_from_storage(storage_context)


def get_index() -> VectorStoreIndex:
    """Return the shared VectorStoreIndex, building or loading as needed."""
    global _index
    if _index is not None:
        return _index

    if _storage_is_current(_STORAGE_DIR, _main_source_files()):
        _index = _load_index()
    else:
        _index = _build_index()
    return _index


def retrieve(query: str, top_k: int = _DEFAULT_TOP_K) -> list[str]:
    """Retrieve chunk texts for a query. Framework-agnostic plain function."""
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]


# ---------------------------------------------------------------------------
# Extension 2: separate abilities index
# ---------------------------------------------------------------------------

def _build_abilities_index() -> VectorStoreIndex:
    Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    Settings.llm = None

    abilities_file = _DATA_DIR / "abilities.md"
    documents = SimpleDirectoryReader(input_files=[str(abilities_file)]).load_data()
    splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(_ABILITIES_STORAGE_DIR))
    _write_manifest(_ABILITIES_STORAGE_DIR, _current_manifest([abilities_file]))
    return index


def get_abilities_index() -> VectorStoreIndex:
    """Return the abilities VectorStoreIndex, building or loading as needed."""
    global _abilities_index
    if _abilities_index is not None:
        return _abilities_index

    abilities_file = _DATA_DIR / "abilities.md"
    if _storage_is_current(_ABILITIES_STORAGE_DIR, [abilities_file]):
        Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
        Settings.llm = None
        sc = StorageContext.from_defaults(persist_dir=str(_ABILITIES_STORAGE_DIR))
        _abilities_index = load_index_from_storage(sc)
    else:
        _abilities_index = _build_abilities_index()
    return _abilities_index


def retrieve_abilities(query: str, top_k: int = _DEFAULT_TOP_K) -> list[str]:
    """Retrieve from the abilities index. Framework-agnostic plain function."""
    index = get_abilities_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]
