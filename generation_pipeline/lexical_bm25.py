"""
A minimal, hard-coded BM25 lexical retriever demo using LangChain.

- Lexical retrieval baseline
- BM25 retriever in langchain-community

pip install -U langchain langchain-community rank_bm25
"""

from langchain_community.retrievers import BM25Retriever


TEXTS = [
    "Australia won the Cricket World Cup 2023",
    "India and Australia played in the finals",
    "Australia won the sixth time having last won in 2015",
]
QUERY = "Who won the 2023 Cricket World Cup?"


def main() -> None:
    retriever = BM25Retriever.from_texts(TEXTS)
    results = retriever.invoke(QUERY)

    print("=== BM25 Retriever Demo ===")
    print(f"Query: {QUERY!r}\n")
    print("Top results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content}")


if __name__ == "__main__":
    main()
