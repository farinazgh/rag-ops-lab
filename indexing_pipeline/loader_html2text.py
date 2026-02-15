"""
loader_html2text.py

Install deps (PyCharm Terminal):
    pip install langchain-community html2text==2024.2.26 aiohttp lxml

Optional:
    pip install langchain==0.3.19
"""

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


def main() -> None:
    # URL of Wikipedia 2023 Cricket World Cup page
    url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

    loader = AsyncHtmlLoader([url])  # must be a list
    docs = loader.load()

    print(f"Loaded {len(docs)} document(s)\n")
    print("=== Metadata ===")
    print(docs[0].metadata)

    print("\n=== First 500 characters of RAW HTML ===\n")
    print(docs[0].page_content[:500])

    html2text = Html2TextTransformer()
    transformed_docs = html2text.transform_documents(docs)

    print("\n=== First 1000 characters of CLEAN TEXT ===\n")
    print(transformed_docs[0].page_content[:1000])


if __name__ == "__main__":
    main()

