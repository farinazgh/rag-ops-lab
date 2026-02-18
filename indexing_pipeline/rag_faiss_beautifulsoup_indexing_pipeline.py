from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# CONFIG
# =========================

@dataclass(frozen=True)
class IndexConfig:
    url: str
    output_dir: Path
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1000
    chunk_overlap: int = 120
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ")

    preview_count: int = 2
    preview_chars: int = 240

    demo_query: str = "Who won the 2023 Cricket World Cup?"


CONFIG = IndexConfig(
    url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup",
    output_dir=Path("./faiss_store_clean"),
)


# =========================
# UTILS
# =========================

def require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='...'\n"
        )


def preview(label: str, items: List, count: int, chars: int) -> None:
    print(f"\n[{label}] Preview ({min(count, len(items))} of {len(items)})")
    for i in range(min(count, len(items))):
        text = items[i].page_content.replace("\n", " ").strip()
        print(f"  - {label.lower()}[{i}]: {text[:chars]}{'...' if len(text) > chars else ''}")


# =========================
# PIPELINE
# =========================

def load_web_page(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No data loaded from URL: {url}")
    return docs


def clean_with_bs_transformer(docs):
    """
    For Wikipedia specifically, keep only the main article content.
    This removes most menus, headers, footers, and language lists.
    """
    transformer = BeautifulSoupTransformer()

    # Extract only the main content area
    cleaned = transformer.transform_documents(
        docs,
        tags_to_extract=["div"],
        attrs_to_extract={"id": "mw-content-text"},
    )

    # Fallback: if extraction failed and content is empty, return original docs
    if cleaned and cleaned[0].page_content.strip():
        return cleaned
    return docs


def split_into_chunks(docs, chunk_size: int, chunk_overlap: int, separators: List[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)


def create_embeddings(model_name: str) -> OpenAIEmbeddings:
    require_openai_api_key()
    return OpenAIEmbeddings(model=model_name)


def build_and_save_faiss(chunks, embeddings: OpenAIEmbeddings, output_dir: Path) -> FAISS:
    store = FAISS.from_documents(chunks, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(output_dir))
    return store


def demo_query(store: FAISS, query: str) -> None:
    docs = store.similarity_search(query, k=4)
    print(f"\n[QUERY] {query}")
    print(f"[RESULTS] Returned: {len(docs)} docs")
    if docs:
        print("\nTop hit:\n")
        print(docs[0].page_content)


def run(config: IndexConfig) -> None:
    print("\n=== Indexing + FAISS Pipeline (WebBaseLoader + Clean) ===")
    print(f"Source URL: {config.url}")
    print(f"Output dir: {config.output_dir.resolve()}")
    print(f"Embedding model: {config.embedding_model}")

    embeddings = create_embeddings(config.embedding_model)
    print("[CONVERTER] Embeddings ready")

    docs = load_web_page(config.url)
    print(f"\n[CONNECTOR] Loaded documents: {len(docs)}")
    print(f"[CONNECTOR] First doc chars (raw): {len(docs[0].page_content)}")

    cleaned_docs = clean_with_bs_transformer(docs)
    print(f"[CLEAN] First doc chars (cleaned): {len(cleaned_docs[0].page_content)}")
    preview("DOC", cleaned_docs, 1, config.preview_chars)

    chunks = split_into_chunks(
        cleaned_docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
    )
    print(f"\n[SPLITTER] Chunks: {len(chunks)}")
    preview("CHUNK", chunks, config.preview_count, config.preview_chars)

    store = build_and_save_faiss(chunks, embeddings, config.output_dir)
    print(f"\n[KNOWLEDGE BASE] Indexed vectors: {store.index.ntotal}")
    print(f"[SAVE] Saved to: {config.output_dir.resolve()}")

    demo_query(store, config.demo_query)
    print("\n=== Done ===\n")


if __name__ == "__main__":
    run(CONFIG)
