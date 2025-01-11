from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool

# Define a Python REPL Tool for calculations
python_tool = PythonREPLTool()

# Wrap it as a Tool
tools = [
    Tool(
        name="Calculator",
        func=python_tool.run,
        description="Use this for performing calculations or executing Python code."
    )
]

# Initialize the LLM
llm = OpenAI(temperature=0)

# Initialize the Agent
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)

# Simulate the agent answering a user query
response = agent.run(
    "If I invest $10,000 in TSLA and its price increases by 10%, what will my return be?")
print(response)
