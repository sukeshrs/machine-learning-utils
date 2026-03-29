import os
import requests
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()

MODEL = "gpt-4.1-mini"   # You can swap to gemini-2.0-flash later

# ---------------- Tools ----------------


def kb_search(query: str, top_k: int = 5) -> dict:
    return {"results": [f"KB hit for: {query}"]}


def create_ticket(title: str, body: str) -> dict:
    return {"ticket_id": "CASE-12345"}


def memory_search_tool(query: str) -> dict:
    resp = requests.get("http://localhost:8001/memory/search",
                        params={"query": query, "limit": 3})
    return resp.json()


# ---------------- Sub-agents ----------------
faq_agent = LlmAgent(
    name="faq_agent",
    model=MODEL,
    description="Answers FAQs.",
    instruction="Answer simple FAQs concisely."
)

search_agent = LlmAgent(
    name="search_agent",
    model=MODEL,
    description="Knowledge search agent.",
    instruction="Search KB for information.",
    tools=[kb_search],
)

action_agent = LlmAgent(
    name="action_agent",
    model=MODEL,
    description="Action agent.",
    instruction="Use create_ticket tool when asked to open a ticket.",
    tools=[create_ticket],
)

confirm_then_act = SequentialAgent(
    name="confirm_then_act",
    sub_agents=[
        LlmAgent(
            name="confirm_intent",
            model=MODEL,
            instruction="Confirm the action and fill state['action']",
            output_key="action"
        ),
        action_agent
    ],
)

# ---------------- Summarizer ----------------
summarizer = LlmAgent(
    name="summarizer",
    model=MODEL,
    instruction="Summarize the conversation in 2–3 sentences, capturing facts & decisions."
)


def save_summary_to_memory(session, runner):
    transcript = "\n".join([f"{m.role}: {m.text}" for m in session.history])
    resp = runner.run(session, transcript, agent=summarizer)
    summary_text = resp.text.strip()

    payload = {"session_id": session.session_id, "summary": summary_text}
    r = requests.post("http://localhost:8001/memory/save", json=payload)
    if r.status_code == 200:
        print(f"✅ Saved summary: {summary_text}")
    else:
        print(f"❌ Failed to save summary: {r.text}")


# ---------------- Supervisor ----------------
supervisor = LlmAgent(
    name="supervisor_router",
    model=MODEL,
    description="Supervisor that routes queries and recalls from memory.",
    instruction=(
        "Route queries:\n"
        " - Use 'faq_agent' for FAQs\n"
        " - Use 'search_agent' for KB lookups\n"
        " - Use 'confirm_then_act' for tickets/actions\n"
        "If the query references past sessions, call 'memory_search_tool'."
    ),
    sub_agents=[faq_agent, search_agent, confirm_then_act],
    tools=[memory_search_tool]
)

# ---------------- Main ----------------
if __name__ == "__main__":
    runner = Runner(session_service=InMemorySessionService())
    session = runner.create_session(supervisor)

    # Conversation
    queries = [
        "What’s the capital of France?",
        "Also note we should open a chargeback dispute process."
    ]
    for q in queries:
        resp = runner.run(session, q)
        print(f"User: {q}")
        print(f"Agent: {resp.text}\n")

    # Save summary to Postgres memory API
    save_summary_to_memory(session, runner)

    # Later — new session, recalling from memory
    new_session = runner.create_session(supervisor)
    followup = "What did we decide about chargeback disputes?"
    resp2 = runner.run(new_session, followup)
    print(f"User: {followup}")
    print(f"Agent: {resp2.text}\n")
