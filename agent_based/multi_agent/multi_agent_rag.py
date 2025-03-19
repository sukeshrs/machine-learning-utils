from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.plan_and_execute import load_agent_executor, load_chat_planner
import requests

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Define LLM models
llm = ChatOpenAI(model="gpt-4", temperature=0)
researcher = ChatOpenAI(model="gpt-4")
summarizer = ChatOpenAI(model="gpt-4")
critic = ChatOpenAI(model="gpt-4")

# Define research agents


def research_task(query):
    return researcher.predict(f"Find recent research on: {query}")


def summarize_task(info):
    return summarizer.predict(f"Summarize this research:\n{info}")


def critic_task(summary):
    return critic.predict(f"Critique this summary for accuracy and bias:\n{summary}")

# Define tool usage


def search_tool(query):
    return f"Search results for: {query}"


def stock_price_tool(symbol):
    url = f"https://api.example.com/stocks/{symbol}"  # Replace with a real API
    response = requests.get(url)
    return response.json()


tools = [
    Tool(name="Web Search", func=search_tool,
         description="Search the web for info"),
    Tool(name="Stock Price Fetcher", func=stock_price_tool,
         description="Fetches live stock prices")
]

# Initialize agent with memory
agent = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory, verbose=True
)

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
    critique_prompt = f"Review this response for clarity and accuracy:\n{response}"
    return llm.predict(critique_prompt)


def refine_answer(response, critique):
    refine_prompt = f"Refine the response based on this critique:\n{critique}\nOriginal: {response}"
    return llm.predict(refine_prompt)


# Planning & execution
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools=tools, verbose=True)


def execute_plan(goal):
    plan = planner.plan(goal)
    return executor.execute(plan)


# Example Execution
if __name__ == "__main__":
    # Memory
    agent.run("Who discovered gravity?")
    print(agent.run("What else did he contribute to science?"))

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
    initial_response = agent.run("Explain quantum computing in simple terms.")
    critique = self_critique(initial_response)
    final_response = refine_answer(initial_response, critique)
    print("Final Improved Response:", final_response)

    # Planning
    goal = "Find the top 5 AI companies, summarize their recent advancements, and critique their impact on society."
    print("Planned Execution:", execute_plan(goal))
