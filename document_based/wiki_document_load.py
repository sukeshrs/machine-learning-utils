from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

from util import chunk_docs, create_vector_store

# Specify the wiki page title
loader = WikipediaLoader("Machu Picchu")
documents = loader.load()

print(documents)

chunked_docs = chunk_docs(documents,2000)
v_store = create_vector_store(chunked_docs)
#print(v_store)

# Query the vector store
query = "Where is Machu pichu?"
results = v_store.similarity_search(query, k=3)

# Display results
for i, result in enumerate(results):
    print(f"Result {i+1}:\n{result.page_content}\n")
