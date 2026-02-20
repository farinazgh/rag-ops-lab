## FAISS Retrieval + RAG Notes

This document explains how FAISS-based vector retrieval works in a RAG (Retrieval-Augmented Generation) pipeline and clarifies common misunderstandings.

### What a FAISS Vector Store Contains

A FAISS vector store contains:

* Dense vectors (numerical embeddings)
* A mapping from vector index → stored Document
* Metadata (often pickled)

### What it does NOT contain:

* The embedding model itself
* The ability to embed new queries


When you later call:

```python

docs = vector_store.similarity_search(QUERY)
```

### What happens internally?

Step 1:

Your query string:

"Who won the 2023 Cricket World Cup?"

must be converted into a vector.

FAISS cannot do that.

So LangChain uses the embedding object you passed to:

```python
embeddings.embed_query(QUERY)
```

THAT is why you pass it in.

It’s not for loading existing vectors.

It’s for embedding future queries.


### What if the stored embeddings were created with a different model?


That means:

* Different dimension size (could error out)
* Even if dimensions match, semantic geometry is different
* Similarity search becomes meaningless

It’s like:

`Comparing GPS coordinates from Earth with coordinates from Mars.`


### Embeddings are not universal.

Each model defines its own:

* Dimensionality
* Semantic structure
* Distance geometry

`Query embeddings and stored embeddings come from the same model.
`
save this in prodcution when saving index
`{
  "embedding_model": "text-embedding-3-small",
  "dimension": 1536
}`
And validate on load.
### What if model differs?

* If dimension differs → runtime error.
* If dimension matches but model differs → silent incorrect retrieval.

### temperature=0

* Deterministic
* Less creative
* Better for factual RAG


### max_retries=2

If API fails, retry twice.

Production stability.


* The LLM does not search.
* FAISS does the search.

The LLM just:

* Reads the retrieved text
* Synthesizes an answer
* Follows instructions

Search ≠ generation.