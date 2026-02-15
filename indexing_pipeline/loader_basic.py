from langchain_community.document_loaders import AsyncHtmlLoader

# URL of Wikipedia 2023 Cricket World Cup page
url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

loader = AsyncHtmlLoader([url])  # must be a list

docs = loader.load()

print(f"Loaded {len(docs)} document(s)\n")

print("=== Metadata ===")
print(docs[0].metadata)

print("\n=== First 1000 characters of content ===\n")
print(docs[0].page_content[:1000])
