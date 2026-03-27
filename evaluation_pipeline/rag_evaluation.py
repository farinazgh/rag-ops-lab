# evaluate.py

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    AnswerCorrectness,
    ResponseRelevancy,
    FactualCorrectness
)
from datasets import Dataset

# --- CONFIG ---
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
TEST_SIZE = 5
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

print("Loading document...")
documents = WebBaseLoader(URL).load()

print("Splitting...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print("Generating testset...")
generator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL))
generator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model=EMBED_MODEL)
)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

testset = generator.generate_with_langchain_docs(
    chunks,
    test_size=TEST_SIZE
)

df = testset.to_pandas()

# Convert to evaluation dataset
dataset = Dataset.from_pandas(df)

print("Evaluating...")
result = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        AnswerCorrectness(),
        ResponseRelevancy(),
        FactualCorrectness()
    ],
    llm=generator_llm
)

print("\n=== RAGAS RESULT ===\n")
print(result)