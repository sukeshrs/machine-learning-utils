import autogen
from autogen import AssistantAgent, UserProxyAgent
from constants import *

print(OPENAI_API_KEY)

# Define the model
config_list = [
    {
        "model": "gpt-4-turbo",
        "api_key": OPENAI_API_KEY,
    }
]

# Create Agents
researcher = AssistantAgent(
    "Researcher", system_message="Find and retrieve research papers related to a given topic.", config_list=config_list)
summarizer = AssistantAgent(
    "Summarizer", system_message="Extract key insights and summarize papers.", config_list=config_list)
analyst = AssistantAgent(
    "Analyst", system_message="Analyze trends and limitations from multiple papers.", config_list=config_list)
writer = AssistantAgent(
    "Writer", system_message="Compile findings into a structured research report.", config_list=config_list)


# User proxy to control the workflow
# user = UserProxyAgent("User", code_execution_config={"work_dir": "code"})

# Agentic workflow
def run_research_pipeline(topic):
    print(f"🔍 Researching: {topic}")

    # Researcher fetches papers
    researcher_message = f"Find research papers related to '{topic}' and provide summaries."
    researcher_response = researcher.generate_reply(researcher_message)

    # Summarizer extracts insights
    summarizer_message = f"Summarize the key insights from these papers:\n\n{researcher_response}"
    summary = summarizer.generate_reply(summarizer_message)

    # Analyst identifies trends
    analyst_message = f"Analyze the following summaries for trends, limitations, and future directions:\n\n{summary}"
    trends = analyst.generate_reply(analyst_message)

    # Writer generates the report
    writer_message = f"Generate a structured research report on '{topic}' based on this analysis:\n\n{trends}"
    final_report = writer.generate_reply(writer_message)

    return final_report

# Example usage
topic = "AI in Algorithmic Trading"
report = run_research_pipeline(topic)
print("\n📄 Final Research Report:\n", report)
