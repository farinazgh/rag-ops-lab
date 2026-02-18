"""
RAG Indexing Pipeline (Chunking + OpenAI Embeddings)

What it does:
- Loads a URL using AsyncHtmlLoader
- Splits HTML into sections by tags (h1, h2, table, p)
- Chunks those sections with RecursiveCharacterTextSplitter
- Optionally compares with CharacterTextSplitter on raw HTML
- Optionally embeds chunks with OpenAIEmbeddings
- Optionally saves embeddings + chunks + metadata to disk

Prereq:
  export OPENAI_API_KEY="..."

Run:
  python rag_indexing_pipeline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

# -------------------------
# HARD-CODED CONFIG
# -------------------------
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

# Recursive chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS: Optional[List[str]] = ["\n\n", "\n", "."]

# Preview which chunk
PREVIEW_INDEX = 4

# Compare baseline
COMPARE_SIMPLE = True
SIMPLE_CHUNK_SIZE = 1000
SIMPLE_CHUNK_OVERLAP = 200
SIMPLE_SEPARATOR = "\n"

# For HTML section splitting
SECTIONS_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("table", "Table"),
    ("p", "Paragraph"),
]

# Embedding controls
DO_EMBED = True
EMBED_SOURCE = "final"  # "final" or "simple"
EMBED_MODEL = "text-embedding-3-small"

# Saving controls (set to "" to disable saving)
SAVE_OUT_DIR = "./out"  # e.g. "./out" or ""

# -------------------------
# PRINT / DEBUG CONTROLS
# -------------------------
VERBOSE = True

# Show preview of chunk text before embedding
PRINT_TEXT_PREVIEW = True
TEXT_PREVIEW_ROWS = 2
TEXT_PREVIEW_CHARS = 140

# Show embedding previews + stats
PRINT_EMBEDDING_PREVIEW = True
EMBED_PREVIEW_ROWS = 2  # how many chunk embeddings to preview
EMBED_PREVIEW_DIMS = 20  # how many dimensions from each embedding

PRINT_EMBEDDING_STATS = True
PRINT_SIMILARITY_CHECK = True  # cosine similarity between emb[0] and emb[1], if possible


def load_html(url: str):
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No data loaded from URL: {url}")
    return docs  # list[Document]


def split_html_into_sections(html_doc, sections_to_split_on=None):
    """
    Split HTML text into sections using tag-based rules.
    Returns list[Document] where each doc is a section.
    """
    if sections_to_split_on is None:
        sections_to_split_on = SECTIONS_TO_SPLIT_ON

    splitter = HTMLSectionSplitter(sections_to_split_on)
    section_docs = splitter.split_text(html_doc.page_content)
    return section_docs


def chunk_sections_recursively(
        section_docs,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
):
    """Chunk section Documents into smaller Documents using RecursiveCharacterTextSplitter."""
    if separators is None:
        separators = ["\n\n", "\n", "."]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(section_docs)
    return chunks


def simple_character_chunks(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
):
    """Simple character splitter working on a single text string."""
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.create_documents([text])
    return chunks


def preview_chunks(chunks, idx: int):
    """Print a little preview like your notebook example."""
    if not chunks:
        print("No chunks to preview.")
        return

    if idx < 0 or idx >= len(chunks):
        print(f"Index {idx} is out of range (0..{len(chunks) - 1}).")
        return

    content = chunks[idx].page_content
    print(f"\n--- Chunk[{idx}] preview ---")
    print(f" --> Length: {len(content)} chars")
    print("\n --> First 200 chars:\n", content[:200])
    print("\n --> Last 200 chars:\n", content[-200:])
    print("------------ end preview ------------\n")


# -------------------------
# EMBEDDING + SAVING
# -------------------------

def embed_chunks_openai(
        chunks,
        model: str,
        *,
        verbose: bool = True,
        print_text_preview: bool = True,
        text_preview_rows: int = 2,
        text_preview_chars: int = 140,
        print_embedding_preview: bool = True,
        embed_preview_rows: int = 2,
        embed_preview_dims: int = 20,
        print_stats: bool = True,
        print_similarity_check: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Create OpenAI embeddings for chunk texts.

    Returns:
      matrix: shape (N, D) float32
      dimension: embedding dimension D
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Example:\n  export OPENAI_API_KEY='...'")

    texts = [c.page_content for c in chunks]

    if len(texts) == 0:
        if verbose:
            print("[embed_chunks_openai] No chunks provided → returning empty matrix.")
        return np.zeros((0, 0), dtype=np.float32), 0

    if verbose:
        print("[embed_chunks_openai] Starting embeddings")
        print(f"  Model: {model}")
        print(f"  Number of texts: {len(texts)}")

    if verbose and print_text_preview:
        print("[embed_chunks_openai] Text preview (before embedding):")
        for i in range(min(text_preview_rows, len(texts))):
            snippet = texts[i].replace("\n", " ")
            if len(snippet) > text_preview_chars:
                snippet = snippet[:text_preview_chars] + "..."
            print(f"  - text[{i}] ({len(texts[i])} chars): {snippet}")

    embedder = OpenAIEmbeddings(model=model)

    vectors = embedder.embed_documents(texts)  # list[list[float]]
    # Turn slow Python lists into a compact, fast numerical matrix suitable for large - scale vector math.
    # Python lists → VERY slow loops
    # NumPy arrays → fast C/compiled math
    # Embeddings from OpenAI are already effectively 32-bit precision.

    matrix = np.array(vectors, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise RuntimeError(f"Unexpected embeddings shape: {matrix.shape}")

    dimension = int(matrix.shape[1])

    if verbose:
        print("[embed_chunks_openai] Embeddings created")
        print(f"  Matrix shape: {matrix.shape}  (N={matrix.shape[0]}, D={matrix.shape[1]})")
        print(f"  dtype: {matrix.dtype}")

        # Make numpy printing nicer
        np.set_printoptions(precision=4, suppress=True)

        if print_embedding_preview:
            rows_to_show = min(embed_preview_rows, matrix.shape[0])
            dims_to_show = min(embed_preview_dims, matrix.shape[1])

            print("[embed_chunks_openai] Embedding preview (first rows / first dims):")
            for i in range(rows_to_show):
                print(f"  - emb[{i}][: {dims_to_show}]: {matrix[i, :dims_to_show]}")

        if print_stats:
            print("[embed_chunks_openai] Embedding stats:")
            print(f"  min:  {matrix.min():.6f}")
            print(f"  max:  {matrix.max():.6f}")
            print(f"  mean: {matrix.mean():.6f}")
            print(f"  std:  {matrix.std():.6f}")

        if print_similarity_check and matrix.shape[0] >= 2:
            v0, v1 = matrix[0], matrix[1]
            denom = (np.linalg.norm(v0) * np.linalg.norm(v1))
            cos_sim = float(np.dot(v0, v1) / denom) if denom != 0 else float("nan")
            print("[embed_chunks_openai] Cosine similarity sanity check:")
            print(f"  cos_sim(emb[0], emb[1]) = {cos_sim:.6f}")

    return matrix, dimension


def save_embeddings(out_dir: Path, embeddings: np.ndarray, chunks, meta: dict):
    """
    Save:
      - embeddings.npy
      - chunks.json
      - meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings)

    chunk_records = []
    for i, c in enumerate(chunks):
        chunk_records.append(
            {
                "i": i,
                "text": c.page_content,
                "metadata": getattr(c, "metadata", {}) or {},
            }
        )

    (out_dir / "chunks.json").write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"Saved embeddings + metadata to: {out_dir.resolve()}")


def main():
    html_docs = load_html(URL)
    html_doc = html_docs[0]

    print(f"Loaded HTML from: {URL}")
    print(f"Raw page_content length: {len(html_doc.page_content)} chars")

    # 2) Split HTML into sections
    section_docs = split_html_into_sections(html_doc, SECTIONS_TO_SPLIT_ON)
    print(f"\nNumber of HTML sections created: {len(section_docs)}")
    print("*" * 60)

    # 3) Chunk sections recursively
    final_chunks = chunk_sections_recursively(
        section_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    print(f"Number of final chunks created (recursive): {len(final_chunks)}")
    print("*" * 60)

    # Preview final chunks
    preview_chunks(final_chunks, PREVIEW_INDEX)

    # 4) Optional: simple character splitting on raw HTML page text
    simple_chunks = None
    if COMPARE_SIMPLE:
        simple_chunks = simple_character_chunks(
            html_doc.page_content,
            chunk_size=SIMPLE_CHUNK_SIZE,
            chunk_overlap=SIMPLE_CHUNK_OVERLAP,
            separator=SIMPLE_SEPARATOR,
        )
        print(f"\n[Compare] Number of simple CharacterTextSplitter chunks: {len(simple_chunks)}")
        print("*" * 60)
        preview_chunks(simple_chunks, PREVIEW_INDEX)

    # 5) Optional: embeddings
    if DO_EMBED:
        if EMBED_SOURCE == "simple":
            if simple_chunks is None:
                simple_chunks = simple_character_chunks(
                    html_doc.page_content,
                    chunk_size=SIMPLE_CHUNK_SIZE,
                    chunk_overlap=SIMPLE_CHUNK_OVERLAP,
                    separator=SIMPLE_SEPARATOR,
                )
            chunks_to_embed = simple_chunks
        else:
            chunks_to_embed = final_chunks

        print(f"\nEmbedding {len(chunks_to_embed)} chunks with '{EMBED_MODEL}' ...")

        emb_mat, dimension = embed_chunks_openai(
            chunks_to_embed,
            model=EMBED_MODEL,
            verbose=VERBOSE,
            print_text_preview=PRINT_TEXT_PREVIEW,
            text_preview_rows=TEXT_PREVIEW_ROWS,
            text_preview_chars=TEXT_PREVIEW_CHARS,
            print_embedding_preview=PRINT_EMBEDDING_PREVIEW,
            embed_preview_rows=EMBED_PREVIEW_ROWS,
            embed_preview_dims=EMBED_PREVIEW_DIMS,
            print_stats=PRINT_EMBEDDING_STATS,
            print_similarity_check=PRINT_SIMILARITY_CHECK,
        )

        print(f"Embedding dimension: {dimension}")
        print("*" * 60)

        # 6) Optional: save results
        if SAVE_OUT_DIR:
            meta = {
                "url": URL,
                "embed_model": EMBED_MODEL,
                "embed_dim": dimension,
                "embed_source": EMBED_SOURCE,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "num_sections": len(section_docs),
                "num_final_chunks": len(final_chunks),
                "num_simple_chunks": len(simple_chunks) if simple_chunks else 0,
            }
            save_embeddings(Path(SAVE_OUT_DIR), emb_mat, chunks_to_embed, meta)
        else:
            print("SAVE_OUT_DIR is empty; not saving embeddings to disk.")


if __name__ == "__main__":
    main()
