# rag.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- CONFIG ---
FAISS_PATH = "./faiss_store"
INDEX_NAME = "cwc_index"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 2

QUERY = "Who won the 2023 Cricket World Cup?"

# Load FAISS
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    index_name=INDEX_NAME,
    allow_dangerous_deserialization=True
)

# Retrieve
docs = vector_store.similarity_search(QUERY, k=TOP_K)
contexts = [d.page_content for d in docs]

# Prompt
prompt = f"""
Answer the question based only on the context below.

Question: {QUERY}

Context:
{contexts}

If the answer is not in the context, say: I don't know.
"""

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
response = llm.invoke([("human", prompt)])

print("\n=== ANSWER ===\n")
print(response.content)