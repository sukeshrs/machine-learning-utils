import autogen
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define assistant agents
llm_config = {"config_list": [{"model": "gpt-4", "temperature": 0}]}
researcher = AssistantAgent(name="Researcher", llm_config=llm_config)
summarizer = AssistantAgent(name="Summarizer", llm_config=llm_config)
critic = AssistantAgent(name="Critic", llm_config=llm_config)
planner = AssistantAgent(name="Planner", llm_config=llm_config)
executor = AssistantAgent(name="Executor", llm_config=llm_config)

# Define a user proxy agent
user_proxy = UserProxyAgent(name="User", human_input_mode="TERMINATE")

# Define research tasks


def research_task(query):
    return researcher.generate_reply(f"Find recent research on: {query}")


def summarize_task(info):
    return summarizer.generate_reply(f"Summarize this research:\n{info}")


def critic_task(summary):
    return critic.generate_reply(f"Critique this summary for accuracy and bias:\n{summary}")

# Define tool functions


def search_tool(query):
    return f"Search results for: {query}"


def stock_price_tool(symbol):
    url = f"https://api.example.com/stocks/{symbol}"  # Replace with a real API
    response = requests.get(url)
    return response.json()


# Load & split documents for RAG
loader = TextLoader("research_papers.txt")  # Assume research papers exist
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector database
embedding_function = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embedding_function)


def retrieve_knowledge(query):
    return vector_store.similarity_search(query)[0].page_content

# Self-critique & refinement


def self_critique(response):
    return critic.generate_reply(f"Review this response for clarity and accuracy:\n{response}")


def refine_answer(response, critique):
    return executor.generate_reply(f"Refine the response based on this critique:\n{critique}\nOriginal: {response}")

# Planning & execution


def execute_plan(goal):
    chat = GroupChat(agents=[planner, executor], messages=[])
    manager = GroupChatManager(groupchat=chat)
    planner.initiate_chat(manager, message=goal)
    return "Execution Complete"


# Example Execution
if __name__ == "__main__":
    # Multi-Agent Collaboration
    research_output = research_task("Recent advancements in AI Agents")
    summary = summarize_task(research_output)
    critique = critic_task(summary)
    print("Summary:", summary)
    print("Critique:", critique)

    # RAG
    print("Relevant Document:", retrieve_knowledge(
        "Latest trends in AI governance"))

    # Self-Improvement
    initial_response = user_proxy.generate_reply(
        "Explain quantum computing in simple terms.")
    critique = self_critique(initial_response)
    final_response = refine_answer(initial_response, critique)
    print("Final Improved Response:", final_response)

    # Planning
    goal = "Find the top 5 AI companies, summarize their recent advancements, and critique their impact on society."
    print("Planned Execution:", execute_plan(goal))
