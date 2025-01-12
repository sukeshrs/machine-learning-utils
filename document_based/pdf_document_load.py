from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the PDF file
pdf_path = "../files/Metiorids_Impacting_moon.pdf"
loader = PyPDFLoader(pdf_path)

# Load the PDF into LangChain documents
documents = loader.load()

# Check the number of pages loaded
print(f"Number of pages: {len(documents)}")


def chunk_docs(documents):
    # Retrieve relevant documents
    # chunk_size : Number of characters per chunk.
    # chunk_iverlap : Number of overlapping characters between consecutive chunks to maintain context.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


chunked_docs = chunk_docs(documents)

print(f"Number of chunks: {len(chunked_docs)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)


# Create a retriever
retriever = vector_store.as_retriever()

# Search for a specific query
query = "What does the document say about angular momentum?"
retrieved_docs = retriever.get_relevant_documents(query)

# Display the retrieved chunks
for idx, doc in enumerate(retrieved_docs):
    print(f"Chunk {idx + 1}:")
    print(doc.page_content)
    print("-" * 80)
