from elasticsearch import Elasticsearch

# Initialize ElasticSearch client
es = Elasticsearch("http://localhost:9200")

# Example data
documents = [
    {"text": "I love programming in Python."},
    {"text": "Data science is a fascinating field."},
    {"text": "Machine learning enables computers to learn from data."},
    {"text": "Python is great for building machine learning models."},
    {"text": "I enjoy outdoor activities like hiking and cycling."}
]
query = "Tell me about Python and machine learning."

# Step 1: Index documents with embeddings
for i, doc in enumerate(documents):
    # Assume `model.encode()` for embeddings
    embedding = model.encode(doc["text"]).tolist()
    doc["embedding"] = embedding
    es.index(index="embeddings", id=i, body=doc)

# Step 2: Query using embedding similarity
query_embedding = model.encode(query).tolist()
script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
            "params": {"query_vector": query_embedding}
        }
    }
}

response = es.search(index="embeddings", body={
                     "query": script_query, "size": 3})

# Print results
print("Query:", query)
for hit in response["hits"]["hits"]:
    print(f"Result (Score: {hit['_score']:.4f}): {hit['_source']['text']}")
