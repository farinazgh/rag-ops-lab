from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Optional: direct FAISS index creation like your notebook.
try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# =========================
# CONFIG (edit these)
# =========================

@dataclass(frozen=True)
class IndexConfig:
    url: str
    output_dir: Path
    embedding_model: str = "text-embedding-3-small"

    # Splitting strategy
    chunk_size: int = 1000
    chunk_overlap: int = 120
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ")

    # Small debug previews
    preview_count: int = 2
    preview_chars: int = 200

    # Demo query
    demo_query: str = "Who won the 2023 Cricket World Cup?"


CONFIG = IndexConfig(
    url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup",
    output_dir=Path("./faiss_store"),
)


# =========================
# HELPERS
# =========================

def require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set.\n"
            "Set it like:\n"
            "  export OPENAI_API_KEY='...'\n"
            "Or in Python before running (less ideal):\n"
            "  os.environ['OPENAI_API_KEY']='...'\n"
        )


def preview(label: str, items: List, count: int, chars: int) -> None:
    print(f"\n[{label}] Preview ({min(count, len(items))} of {len(items)})")
    for i in range(min(count, len(items))):
        text = items[i].page_content.replace("\n", " ").strip()
        print(f"  - {label.lower()}[{i}]: {text[:chars]}{'...' if len(text) > chars else ''}")


# =========================
# PIPELINE STEPS
# =========================

def load_page_text(url: str):
    """
    Connector: fetch page content using WebBaseLoader.
    This typically returns cleaner main-page text than raw HTML loaders.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No data loaded from URL: {url}")

    # Light cleanup (helps avoid empty lines + some weird spacing)
    for d in documents:
        d.page_content = "\n".join(line.strip() for line in d.page_content.splitlines() if line.strip())

    return documents


def chunk_documents_recursively(
    documents,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List:
    """
    Splitter: break long documents into manageable chunks for embedding.
    """
    chunker = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunker.split_documents(documents)


def create_embeddings(model_name: str) -> OpenAIEmbeddings:
    require_openai_api_key()
    return OpenAIEmbeddings(model=model_name)


# =========================
# FAISS BUILD/LOAD
# =========================

def build_faiss_store_via_add_documents(chunk_documents, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Notebook-parity approach:
      - Create an index
      - Create FAISS vector store with InMemoryDocstore
      - add_documents(...)
    """
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed (or failed to import). Install it with:\n"
            "  pip install faiss-cpu==1.10.0\n"
        )

    dim = len(embeddings.embed_query("dimension check"))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(documents=chunk_documents)
    return vector_store


def save_store(vector_store: FAISS, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))


def load_store(output_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    if not output_dir.exists():
        raise RuntimeError(f"FAISS store directory does not exist: {output_dir.resolve()}")
    return FAISS.load_local(
        str(output_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def demo_similarity_search(vector_store: FAISS, query: str) -> None:
    docs = vector_store.similarity_search(query)
    print(f"\n[QUERY] {query}")
    print(f"[RESULTS] Returned: {len(docs)} docs")
    if docs:
        print("\nTop hit:\n")
        print(docs[0].page_content)


# =========================
# ORCHESTRATION
# =========================

def run_indexing_pipeline(config: IndexConfig, *, rebuild: bool = True) -> None:
    print("\n=== Indexing + FAISS Pipeline (WebBaseLoader) ===")
    print(f"Source URL: {config.url}")
    print(f"Output dir: {config.output_dir.resolve()}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Rebuild: {rebuild}")

    embeddings = create_embeddings(config.embedding_model)
    print("[CONVERTER] Embeddings ready")

    if rebuild:
        # 1) Source -> Connector (WebBaseLoader)
        docs = load_page_text(config.url)
        print(f"\n[CONNECTOR] Loaded documents: {len(docs)}")
        print(f"[CONNECTOR] First doc chars: {len(docs[0].page_content)}")
        preview("DOC", docs, config.preview_count, config.preview_chars)

        # 2) Splitter
        chunk_documents = chunk_documents_recursively(
            docs,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=list(config.separators),
        )
        print(f"[SPLITTER] Chunks: {len(chunk_documents)}")
        preview("CHUNK", chunk_documents, config.preview_count, config.preview_chars)

        # 3) Knowledge base
        vector_store = build_faiss_store_via_add_documents(chunk_documents, embeddings)
        print(f"\n[KNOWLEDGE BASE] Indexed vectors (index.ntotal): {vector_store.index.ntotal}")

        # 4) Save
        save_store(vector_store, config.output_dir)
        print(f"[SAVE] FAISS store saved to: {config.output_dir.resolve()}")

    else:
        vector_store = load_store(config.output_dir, embeddings)
        print(f"[LOAD] Loaded FAISS store. Indexed vectors (index.ntotal): {vector_store.index.ntotal}")

    # 5) Demo query
    demo_similarity_search(vector_store, config.demo_query)
    print("\n=== Done ===\n")


if __name__ == "__main__":
    run_indexing_pipeline(CONFIG, rebuild=True)
