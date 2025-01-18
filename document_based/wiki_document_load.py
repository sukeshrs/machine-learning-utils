from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os


# Specify the Wikipedia page title
loader = WikipediaLoader("Machu Picchu")
documents = loader.load()

print(documents)
