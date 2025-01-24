from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2")
# Load Pre-trained model
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Helper function to compute embeddings
def compute_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**inputs)
    return model_output.pooler_output[0].numpy()


# Example data
documents = [
    "I love programming in Python.",
    "Data science is a fascinating field.",
    "Machine learning enables computers to learn from data.",
    "Python is great for building machine learning models.",
    "I enjoy outdoor activities like hiking and cycling."
]
query = "Tell me about Python and machine learning."


# Step 1: Generate embeddings
doc_embeddings = [compute_embedding(doc) for doc in documents]
query_embedding = compute_embedding(query)


# Step 2: Compute cosine similarity
similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()


# Step 3: Retrieve top results
top_indices = similarities.argsort()[-3:][::-1]  # Top 3 results
print("Query:", query)
for idx in top_indices:
    print(f"Result (Score: {similarities[idx]:.4f}): {documents[idx]}")
