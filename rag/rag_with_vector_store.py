import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load a sentence transformer model . all-MiniLM-L6-v2 is a sentence-transformers model: It maps sentences & paragraphs 
# to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the content from the excel file
text_embeddings = embedding_model.encode(texts, convert_to_tensor=True)


# Initialize FAISS index
dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Convert embeddings to numpy array and add to FAISS
faiss_embeddings = np.array([embedding.cpu().numpy()
                            for embedding in text_embeddings])
index.add(faiss_embeddings)

# Keep track of metadata (e.g., mapping indices to original texts)
doc_map = {i: text for i, text in enumerate(texts)}


'''
Use the query embedding to find the nearest matches in the vector store.
'''
def retrieve_docs_with_embeddings(query, embedding_model, index, doc_map, k=5):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(
        [query], convert_to_tensor=True).cpu().numpy()

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k)

    # Retrieve matching documents
    results = [doc_map[i] for i in indices[0]]
    return results

'''
This method combines the retrieved documents into a single context for the LLM, 
ensuring the total token count remains manageable.
'''
def format_context(docs, max_chars=2000):
    """Format retrieved documents as context for the LLM."""
    context = "\n".join(docs)
    return context[:max_chars]  # Truncate to fit within model limits


def answer_query_with_optimized_rag(query, embedding_model, index, doc_map, llm):
    # Retrieve relevant documents
    retrieved_docs = retrieve_docs_with_embeddings(
        query, embedding_model, index, doc_map)

    if not retrieved_docs:
        return "No relevant documents found."

    # Format context for LLM
    context = format_context(retrieved_docs)

    # Generate response using LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt)

    return response

# Test the system


query = "Great Books to read"
response = answer_query_with_optimized_rag(
    query, embedding_model, index, doc_map, llm)

print(response)
