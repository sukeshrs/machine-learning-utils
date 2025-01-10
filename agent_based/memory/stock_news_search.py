from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain_experimental.tools import BingSearchTool

# Initialize the LLM
llm = OpenAI(temperature=0)

# Step 1: Define the Search Tool
search_tool = Tool(
    name="Search Engine",
    func=BingSearchTool(api_key="your_bing_search_api_key").run,
    description="Search for the latest news or stock prices on the internet."
)

# Step 2: Define Memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

# Step 3: Define a Prompt Template
prompt = PromptTemplate(
    input_variables=["chat_history", "query"],
    template="""
You are a financial assistant with access to live search capabilities. 
You can find the latest news and stock prices for the user.

Conversation History:
{chat_history}

User Query:
{query}

Answer the user's query with relevant, accurate information.
"""
)

# Step 4: Initialize the Agent
tools = [search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)

# Step 5: User Query
query = "What is the latest news about Tesla's stock price?"

# Run the Agent
response = agent.run(prompt.format(
    query=query, chat_history=memory.load_memory_variables()))
print(response)
