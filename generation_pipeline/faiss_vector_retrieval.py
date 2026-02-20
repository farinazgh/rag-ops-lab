"""
Load an existing FAISS vector store from disk and retrieve top-k chunks.

This is the "vector retrieval" part of RAG, without generation yet.

pip install -U langchain langchain-community langchain-openai openai tiktoken faiss-cpu python-dotenv

export OPENAI_API_KEY="..."

FAISS.load_local uses pickle for some metadata. Only load indexes you trust.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FAISS_FOLDER_PATH = "/home/ubuntu/faiss_store"
FAISS_INDEX_NAME = "index"
QUERY = "Who won the 2023 Cricket World Cup?"
TOP_K = 2


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Do: export OPENAI_API_KEY='...'"
        )

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    vector_store = FAISS.load_local(
        folder_path=FAISS_FOLDER_PATH,
        index_name=FAISS_INDEX_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # only load stores you trust
    )

    docs = vector_store.similarity_search(QUERY, k=TOP_K)

    print("=== FAISS Vector Retrieval Demo ===")
    print(f"Query: {QUERY!r}")
    print(f"Top-k: {TOP_K}\n")

    for i, doc in enumerate(docs, start=1):
        print(f"--- Result #{i} ---")
        print(doc.page_content)
        # metadata often includes source, URL, chunk ids, etc.
        if doc.metadata:
            print(f"\nmetadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    main()
