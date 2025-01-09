import yfinance as yf
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool
import os

# Set your API key for OpenAI
os.environ["OPENAI_API_KEY"] = ""

# Step 1: Define custom tool for stock data


def get_stock_price(symbol: str) -> str:
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"The current price of {symbol.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Error retrieving stock data: {str(e)}"


stock_tool = Tool(
    name="StockPriceFetcher",
    func=lambda query: get_stock_price(query),
    description="Fetches the current stock price for a given ticker symbol (e.g., AAPL, GGOG, META)."
)

# Step 2: Add a calculator tool for financial computations
calculator_tool = PythonREPLTool()

tools = [stock_tool, calculator_tool]

# Step 3: Set up the LLM
llm = OpenAI(temperature=0)

# Step 4: Initialize the agent
finance_agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Step 5: Use the agent for financial queries
queries = [
    "What is the current price of META stock?",
    "If I invest $1000 in META and its price increases by 10%, what will my return be?",
]

for query in queries:
    response = finance_agent.run(query)
    print(f"Query: {query}\nResponse: {response}\n")
