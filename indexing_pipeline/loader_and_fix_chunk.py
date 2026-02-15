#!/usr/bin/env python3
"""
loader_html2text_and_chunk.py

Install deps:
    pip install langchain-community html2text==2024.2.26 aiohttp lxml

Optional (depending on your environment):
    pip install langchain==0.3.19
"""

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter


def main() -> None:
    # URL of Wikipedia 2023 Cricket World Cup page
    url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

    # -----------------------------
    # 1) Load raw HTML as Documents
    # -----------------------------
    loader = AsyncHtmlLoader([url])  # must be a list
    docs = loader.load()

    if not docs:
        raise RuntimeError("No documents were loaded. Check the URL or your network access.")

    print(f"Loaded {len(docs)} document(s)\n")
    print("=== Metadata ===")
    print(docs[0].metadata)

    print("\n=== First 500 characters of RAW HTML ===\n")
    print(docs[0].page_content[:500])

    # ---------------------------------
    # 2) Transform HTML -> Cleaned text
    # ---------------------------------
    html2text = Html2TextTransformer()
    html_data_transformed = html2text.transform_documents(docs)

    if not html_data_transformed:
        raise RuntimeError("HTML was loaded but transformation produced no output.")

    clean_text = html_data_transformed[0].page_content

    print("\n=== First 1000 characters of CLEAN TEXT ===\n")
    print(clean_text[:1000])

    # -----------------------------
    # 3) Split clean text into chunks
    # -----------------------------
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.create_documents([clean_text])

    # -----------------------------
    # 4) Print chunking results
    # -----------------------------
    print(f"\nThe number of chunks created: {len(chunks)}")

    if chunks:
        print("********************************************************")
        print("\n=== First chunk preview ===\n")
        print(chunks[0].page_content[:400])

    if len(chunks) > 1:
        print("********************************************************")
        print("\n=== Second chunk preview ===\n")
        print(chunks[1].page_content[:400])

    if len(chunks) > 2:
        print("********************************************************")
        print("\n=== Third chunk preview ===\n")
        print(chunks[2].page_content[:400])

    print("********************************************************")


if __name__ == "__main__":
    main()
