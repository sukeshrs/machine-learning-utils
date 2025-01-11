from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool

# Initialize LLM
llm = OpenAI(temperature=0, model="text-davinci-003")

# Define Tools
calculator_tool = Tool(
    name="Calculator",
    func=PythonREPLTool().run,
    description="Perform financial calculations."
)

tools = [calculator_tool]

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# Create Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory,
)

# Example Interaction
print("Personal Finance Assistant: Hi, how can I help you with your finances today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Personal Finance Assistant: Goodbye!")
        break
    response = agent.run(user_input)
    print(f"Personal Finance Assistant: {response}")
