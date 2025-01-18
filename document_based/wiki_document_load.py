from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

from util import chunk_docs, create_vector_store

# Specify the Wikipedia page title
loader = WikipediaLoader("Machu Picchu")
documents = loader.load()

print(documents)

chunked_docs = chunk_docs(documents,2000)
v_store = create_vector_store(chunked_docs)
print(v_store)

