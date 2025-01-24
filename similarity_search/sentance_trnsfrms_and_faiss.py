from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Example data
documents = [
    "I love programming in Python.",
    "The cat sat on the mat.",
    "Artificial Intelligence is fascinating.",
    "Machine learning is a subset of AI.",
    "I enjoy hiking in the mountains."
]


# Query
query1 = "Tell me about machine learning and AI."

query2 = "Which are the best mountains"


# Step 1: Load a SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Encode the documents and query into embeddings
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query1, query2])

print(doc_embeddings.shape)

dimension = doc_embeddings.shape[1]  # Embedding dimension
# Creates an empty FAISS index.Vectors with dimension size can be added to this index for similarity search.
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(doc_embeddings)

# Step 3: Perform a similarity search
k = 3  # Number of results to retrieve
# reshape(1, -1): Reshapes the query vector into a 2D array because FAISS expects input in the form of[n_queries, dimension]
distances, indices = index.search(query_embedding.reshape(2, -1), k)


# Print results
print("Query1:", query1)
for i, idx in enumerate(indices[0]):
    print(f"Result {i + 1} (Score: {distances[0][i]:.5f}): {documents[idx]}")

print("Query2:", query2)
for i, idx in enumerate(indices[1]):
    print(f"Result {i + 1} (Score: {distances[1][i]:.5f}): {documents[idx]}")
