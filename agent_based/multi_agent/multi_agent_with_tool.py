import os
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    CodeExecutorAgent,
    config_list_from_json
)
from autogen.agentchat.contrib.tools import FunctionTool

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load LLM configuration
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",  # or provide path to JSON
    filter_dict={"model": ["gpt-4", "gpt-3.5-turbo"]}
)

# Define Tool: Simple calculator


def multiply_numbers(a: float, b: float) -> float:
    return a * b


calculator_tool = FunctionTool.from_defaults(fn=multiply_numbers)

# Define Assistant Agent (brain of the operation)
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0,
    },
    tools=[calculator_tool]
)

# Define Code Executor Agent
executor = CodeExecutorAgent(
    name="code_executor",
    code_execution_config={"use_docker": False}  # safer in Docker
)

# Define User Proxy Agent (simulates a user prompt)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Change to ALWAYS if you want interactive mode
    code_execution_config={"use_docker": False}
)

# Register other agents with assistant
assistant.register_agent(executor)
assistant.register_agent(user_proxy)

# Register assistant with user proxy (bi-directional)
user_proxy.register_agent(assistant)

# Run a multi-step task
user_proxy.initiate_chat(
    assistant,
    message="""
I'd like to:
1. Calculate the product of 7.5 and 8 using a tool.
2. Generate and plot the first 15 Fibonacci numbers using Python.
3. Save the plot as a file called fibonacci_plot.png.
4. Return the result of the multiplication and show the plot.
"""
)
