from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import pandas as pd

# Load Excel data
file_path = "your_data.xlsx"
df = pd.read_excel(file_path)

# Extract text/content column
texts = df["Content"].tolist()


def retrieve_docs(query, texts):
    """Retrieve matching documents from the dataset based on text."""
    results = [text for text in texts if query.lower() in text.lower()]
    return results


# Load Mistral Instruct model
mistral_pipeline = pipeline("text-generation", model="mistral-instruct")
llm = HuggingFacePipeline(pipeline=mistral_pipeline)


def answer_query_with_rag(query, texts):
    # Retrieve relevant documents
    retrieved_docs = retrieve_docs(query, texts)

    if not retrieved_docs:
        return "No relevant documents found."

    # Combine retrieved documents
    context = "\n".join(retrieved_docs)

    # Generate response using LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt)

    return response


query = "Great books to read"
response = answer_query_with_rag(query, texts)

print(response)
