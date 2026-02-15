"""
RAG Indexing Pipeline (Chunking-focused)

What it does:
- Loads a URL (Wikipedia or any HTML page) using AsyncHtmlLoader
- Splits HTML into semantic-ish sections by tags (h1, h2, table, p) using HTMLSectionSplitter
- Chunks those sections with RecursiveCharacterTextSplitter
- Optionally runs CharacterTextSplitter on the raw page content for comparison

Run:
    pip install langchain
    pip install langchain-community
    pip install langchain-text-splitters

    python rag_indexing_pipeline.py
"""

from __future__ import annotations

from typing import List, Optional

# LangChain loaders + splitters
# downloads the webpage and returns LangChain Document objects
from langchain_community.document_loaders import AsyncHtmlLoader

# breaks HTML into “sections” based on tags like h1, h2, etc.
from langchain_text_splitters import HTMLSectionSplitter

# the simple baseline chunker (fixed separator approach).
from langchain.text_splitter import CharacterTextSplitter

# makes better chunks by splitting using a priority of separators.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# HARD-CODED CONFIG
# -------------------------
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

# Recursive chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# split by blank lines first, then newlines, then periods.
SEPARATORS: Optional[List[str]] = ["\n\n", "\n", "."]

# Preview which chunk
PREVIEW_INDEX = 4

# If True, also run a simple CharacterTextSplitter on raw HTML content
COMPARE_SIMPLE = True

# For HTML section splitting
SECTIONS_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("table", "Table"),
    ("p", "Paragraph"),
]


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
    print("********* chunk_sections_recursively *********")

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


def main():
    html_docs = load_html(URL)
    # one URL; one document
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

    # Preview
    preview_chunks(final_chunks, PREVIEW_INDEX)

    # 4) Optional: simple character splitting on raw HTML page text
    if COMPARE_SIMPLE:
        simple_chunks = simple_character_chunks(
            html_doc.page_content,
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
        )
        print(f"\n[Compare] Number of simple CharacterTextSplitter chunks: {len(simple_chunks)}")
        print("*" * 60)

        preview_chunks(simple_chunks, PREVIEW_INDEX)


if __name__ == "__main__":
    main()
