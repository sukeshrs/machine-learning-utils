# pip install google-adk openai

import os
from google.adk.agents import LlmAgent, SequentialAgent, AgentTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Setup OPENAI_API_KEY
# export OPENAI_API_KEY="sk-..."

# ------------------ Tools ------------------


def kb_search(query: str, top_k: int = 5) -> dict:
    """Mock KB search tool."""
    return {"results": [f"hit for: {query}"]}


def create_ticket(title: str, body: str) -> dict:
    """Mock ticket creation tool."""
    return {"ticket_id": "CASE-12345"}


# ------------------ Sub-agents ------------------
faq_agent = LlmAgent(
    name="faq_agent",
    model="gpt-4.1-mini",
    description="Answers short policy/FAQ queries without side-effects.",
    instruction="Answer simple FAQs concisely. If unsure, escalate to search."
)

search_agent = LlmAgent(
    name="search_agent",
    model="gpt-4.1-mini",
    description="Searches KB and synthesizes answers.",
    instruction="Use the `kb_search` tool to find info and synthesize an answer.",
    tools=[kb_search],
)

action_agent = LlmAgent(
    name="action_agent",
    model="gpt-4.1-mini",
    description="Executes actions like creating tickets.",
    instruction="Only perform side-effects when explicitly asked. Use `create_ticket`.",
    tools=[create_ticket],
)

# Optional: sequential flow (confirm before acting)
confirm_then_act = SequentialAgent(
    name="confirm_then_act",
    sub_agents=[
        LlmAgent(
            name="confirm_intent",
            model="gpt-4.1-mini",
            instruction="Confirm the exact action and fields. Write to state['action']",
            output_key="action"
        ),
        action_agent
    ],
)

# ------------------ Supervisor ------------------
supervisor = LlmAgent(
    name="supervisor_router",
    model="gpt-4.1-mini",
    description="Routes requests to the correct sub-agent.",
    instruction=(
        "You are the supervisor. Analyze the user's request and delegate:\n"
        " - Use 'faq_agent' for simple FAQs.\n"
        " - Use 'search_agent' for retrieval questions.\n"
        " - Use 'confirm_then_act' for action requests (tickets, refunds).\n"
        "If uncertain, prefer 'search_agent'."
    ),
    sub_agents=[faq_agent, search_agent, confirm_then_act],
)

# ------------------ Run ------------------
runner = Runner(session_service=InMemorySessionService())
session = runner.create_session(supervisor)

# Try different queries
queries = [
    "What’s the capital of France?",
    "Find dispute policy for credit card chargebacks.",
    "Open a ticket: my card was double charged yesterday."
]

for q in queries:
    resp = runner.run(session, q)
    print(f"User: {q}")
    print(f"Agent: {resp.text}\n")
