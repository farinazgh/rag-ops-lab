"""
A minimal, hard-coded TF‑IDF lexical retriever demo using LangChain.

- Lexical retrieval (no embeddings, no vector database)
- TF‑IDF retriever in langchain-community

pip install -U langchain langchain-community scikit-learn
"""

from langchain_community.retrievers import TFIDFRetriever


TEXTS = [
    "Australia won the Cricket World Cup 2023",
    "India and Australia played in the finals",
    "Australia won the sixth time having last won in 2015",
]
QUERY = "won"


def main() -> None:
    retriever = TFIDFRetriever.from_texts(TEXTS)
    results = retriever.invoke(QUERY)

    print("=== TF-IDF Retriever Demo ===")
    print(f"Query: {QUERY!r}\n")
    print("Top results:")
    for i, doc in enumerate(results, start=1):
        # doc is a LangChain Document
        print(f"{i}. {doc.page_content}")


if __name__ == "__main__":
    main()
