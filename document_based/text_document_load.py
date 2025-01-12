from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

loader = TextLoader("../files/soccer_history.txt")

# Set your API key for OpenAI
os.environ["OPENAI_API_KEY"] = ""

'''
The method chunks the set of input documents in to chunks
'''
def chunk_docs(documents):
    # Retrieve relevant documents
    # chunk_size : Number of characters per chunk.
    # chunk_iverlap : Number of overlapping characters between consecutive chunks to maintain context.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


def embed_chunks(documents):

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store


def content_search(vector_store, query):
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)
    return retriever,retrieved_docs


def retrieve_with_llm(retriever, query):
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    # Create a Retrieval QA Chain
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    # Ask a question
    response = qa_chain.run(query)
    return response


# Load the document into a list of LangChain Documents
documents = loader.load()
# A langchain document mainly has two sections. 1.metadata and 2.page_content
# page_content contains the actual text and the metadata is used to store additional meta data about the content
print(documents)
query = "What is the main idea of the document?"
chunked_docs = chunk_docs(documents)
embedded = embed_chunks(chunked_docs)
retriever, docs = content_search(embedded, query)
for doc in docs:
    print(doc.page_content)

# Use LLM to answer the question
result = retrieve_with_llm(retriever, query)
print(result)


