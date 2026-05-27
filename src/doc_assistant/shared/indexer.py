"""
Shared vector index for the Pokemon document assistant.

Builds a single VectorStoreIndex from data/*.md, persists to storage/.
Exposes two interfaces so both assistants use identical retrieval:
  - get_index()  -> VectorStoreIndex   (native LlamaIndex object)
  - retrieve()   -> list[str]          (plain callable, framework-agnostic)
"""

from contextlib import contextmanager
import fcntl
import logging
from pathlib import Path
import shutil
from typing import Callable, Iterator

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_STORAGE_DIR = _PROJECT_ROOT / "storage"

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CHUNK_SIZE = 256
_CHUNK_OVERLAP = 32
_DEFAULT_TOP_K = 5

_index: VectorStoreIndex | None = None
_abilities_index: VectorStoreIndex | None = None

_ABILITIES_STORAGE_DIR = _PROJECT_ROOT / "storage_abilities"


def _configure_indexing() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    Settings.llm = None


@contextmanager
def _storage_lock(storage_dir: Path) -> Iterator[None]:
    """Serialize load/build so concurrent cold starts do not interleave writes."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    lock_path = storage_dir / ".index.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _clear_storage_dir(storage_dir: Path) -> None:
    """Remove persisted index files while preserving the lock file."""
    if not storage_dir.exists():
        return

    for child in storage_dir.iterdir():
        if child.name == ".index.lock":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _load_persisted_index(storage_dir: Path) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    return load_index_from_storage(storage_context)


def _load_or_build(storage_dir: Path, build_index: Callable[[], VectorStoreIndex]) -> VectorStoreIndex:
    with _storage_lock(storage_dir):
        if not (storage_dir / "docstore.json").exists():
            return build_index()

        _configure_indexing()
        try:
            return _load_persisted_index(storage_dir)
        except Exception:
            logger.warning(
                "Failed to load persisted index at %s; rebuilding it from source data.",
                storage_dir,
                exc_info=True,
            )
            _clear_storage_dir(storage_dir)
            return build_index()


def _build_index() -> VectorStoreIndex:
    _configure_indexing()

    documents = SimpleDirectoryReader(str(_DATA_DIR)).load_data()
    splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(_STORAGE_DIR))
    return index


def _load_index() -> VectorStoreIndex:
    _configure_indexing()
    return _load_persisted_index(_STORAGE_DIR)


def get_index() -> VectorStoreIndex:
    """Return the shared VectorStoreIndex, building or loading as needed."""
    global _index
    if _index is not None:
        return _index

    _index = _load_or_build(_STORAGE_DIR, _build_index)
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
    _configure_indexing()

    abilities_file = _DATA_DIR / "abilities.md"
    documents = SimpleDirectoryReader(input_files=[str(abilities_file)]).load_data()
    splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(_ABILITIES_STORAGE_DIR))
    return index


def get_abilities_index() -> VectorStoreIndex:
    """Return the abilities VectorStoreIndex, building or loading as needed."""
    global _abilities_index
    if _abilities_index is not None:
        return _abilities_index

    _abilities_index = _load_or_build(_ABILITIES_STORAGE_DIR, _build_abilities_index)
    return _abilities_index


def retrieve_abilities(query: str, top_k: int = _DEFAULT_TOP_K) -> list[str]:
    """Retrieve from the abilities index. Framework-agnostic plain function."""
    index = get_abilities_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]
