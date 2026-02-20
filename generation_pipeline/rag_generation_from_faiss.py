"""
Minimal, hard-coded "retrieve + augment prompt + generate" RAG demo:
- Load an existing FAISS vector store from disk
- Retrieve top-k chunks
- Build an augmented prompt with the retrieved context
- Generate an answer using ChatOpenAI (gpt-4o-mini)

pip install -U langchain langchain-community langchain-openai openai tiktoken faiss-cpu python-dotenv

export OPENAI_API_KEY="..."

FAISS.load_local uses pickle for some metadata. Only load indexes you trust.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
FAISS_FOLDER_PATH = "/home/ubuntu/faiss_store"
FAISS_INDEX_NAME = "index"

QUERY = "Who won the 2023 Cricket World Cup?"
TOP_K = 2

PROMPT_TEMPLATE = """
Given the context below, answer the question.

Question: {question}

Context:
{context}

Rules:
- Answer only based on the context provided.
- If the question cannot be answered from the context, say: I don't know.
"""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Do: export OPENAI_API_KEY='...'"
        )

    # 1) Load embeddings + vector store
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    vector_store = FAISS.load_local(
        folder_path=FAISS_FOLDER_PATH,
        index_name=FAISS_INDEX_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # only load stores you trust
    )

    # 2) Retrieve
    docs = vector_store.similarity_search(QUERY, k=TOP_K)

    #  we keep "first chunk only for now.
    # (Later, we can join multiple chunks and/or add citations.)
    retrieved_context = docs[0].page_content if docs else ""

    # 3) Augment prompt
    augmented_prompt = PROMPT_TEMPLATE.format(
        question=QUERY,
        context=retrieved_context,
    )

    # 4) Generate
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        max_retries=2,
        timeout=None,
    )

    response = llm.invoke([("human", augmented_prompt)])
    answer = response.content

    # 5) Print
    print("=== RAG Generation Demo (FAISS + ChatOpenAI) ===")
    print(f"Question: {QUERY}\n")
    print("Answer:")
    print(answer)
    print("\n--- Retrieved context (first chunk) ---")
    print(retrieved_context)


if __name__ == "__main__":
    main()
