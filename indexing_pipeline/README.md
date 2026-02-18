# rag-ops-lab

A minimal setup for experimenting with RAG (Retrieval-Augmented Generation) using LangChain and OpenAI.

---

## Install dependencies

### Option 1 — Install step by step

```bash
pip install numpy
pip install faiss-cpu==1.10.0

pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-openai

pip install openai
pip install tiktoken
pip install pydantic

pip install beautifulsoup4
pip install lxml
pip install aiohttp

pip install python-dotenv   # optional


export OPENAI_API_KEY=""

```

---

## Embedding Example

### Generate embeddings with OpenAI

```python
embedder = OpenAIEmbeddings(model=model)

vectors = embedder.embed_documents(texts)  # list[list[float]]
mat = np.array(vectors, dtype=np.float32)

if mat.ndim != 2 or mat.shape[0] != len(texts):
    raise RuntimeError(f"Unexpected embeddings shape: {mat.shape}")

dim = int(mat.shape[1])

return mat, dim
```

Embedding 280 chunks with 'text-embedding-3-small' ...
[embed_chunks_openai] Starting embeddings
  Model: text-embedding-3-small
  Number of texts: 280

[embed_chunks_openai] Text preview (before embedding):
  - text[0] (656 chars): Jump to content ... Main page ... Contents ...
  - text[1] (983 chars): Contents ... Background ... Host ...

[embed_chunks_openai] Embeddings created
  Matrix shape: (280, 1536)  (N=280, D=1536)
  dtype: float32

[embed_chunks_openai] Embedding preview (first rows / first dims):
  - emb[0][:20]: [ 0.0252  0.0175 -0.012  -0.0023 -0.0422  0.0183 -0.0366 -0.0075 -0.0193
                   0.038   0.0328  0.0033  0.0012  0.0359  0.0637  0.0543 -0.0132  0.0615
                  -0.0264  0.0345]
  - emb[1][:20]: [-0.0125  0.0108  0.0461 -0.0197 -0.0472  0.017   0.0011 -0.0161  0.0128
                   0.0417  0.0318 -0.0019 -0.0104 -0.0198  0.0517  0.01   -0.0113  0.0495
                  -0.0068  0.0601]

[embed_chunks_openai] Embedding stats:
  min:  -0.158769
  max:   0.156426
  mean: -0.000051
  std:   0.025515

[embed_chunks_openai] Cosine similarity sanity check:
  cos_sim(emb[0], emb[1]) = 0.510017

Embedding dimension: 1536
************************************************************
Saved embeddings + metadata to: /home/ubuntu/out


### Turn slow Python lists into a compact, fast numerical matrix suitable for large-scale vector math.

```python

    matrix = np.array(vectors, dtype=np.float32)
```

Python lists are slow for math.
cosine similarity on thousands of embeddings:

Python lists → VERY slow loops

NumPy arrays → fast C/compiled math

| Type        | How math runs                   |
| ----------- | ------------------------------- |
| Python list | element-by-element in Python    |
| NumPy array | optimized linear algebra engine |

### Embeddings from OpenAI are already effectively 32-bit precision.

| dtype   | Approx size |
| ------- | ----------- |
| float64 | ~1.2 GB     |
| float32 | ~600 MB     |


Running the `rag_faiss_indexing_pipeline.py` pipeline creates a local FAISS store:

- `index.faiss` — binary vector index (embeddings only)
- `index.pkl` — document store and metadata mapping

Both files are required to reload the vector database and perform similarity search.
