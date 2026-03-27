# index.py

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIG ---
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
FAISS_PATH = "./faiss_store"
INDEX_NAME = "cwc_index"
EMBED_MODEL = "text-embedding-3-small"

print("Loading documents...")
loader = WebBaseLoader(URL)
documents = loader.load()

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)

print("Creating embeddings...")
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

print("Building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)

print("Saving FAISS index...")
vector_store.save_local(FAISS_PATH, INDEX_NAME)

print("Done.")