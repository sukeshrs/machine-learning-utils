from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def chunk_docs(documents, size=1000, overlap=200):
    # Retrieve relevant documents
    # chunk_size : Number of characters per chunk.
    # chunk_iverlap : Number of overlapping characters between consecutive chunks to maintain context.
    text_splitter = CharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

'''
Chunks the document using RecursiveCharacterTextSplitter.
It tries to split at smaller sections like paragraphs, sentences, or words
'''
def chunk_docs_rec_text_splitter(documents, size=1000, overlap=200):

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

'''
Embeds the documents using sentance transformers. Creates a vector store using FAISS library
FAISS(Facebook AI Similarity Search) is an open-source library developed by Meta for efficient 
similarity search and clustering of dense vectors.
'''
def create_vector_store(chunked_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store
