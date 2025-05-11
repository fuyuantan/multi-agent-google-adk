from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools import google_search

import os

# Add your google api key
os.environ["GOOGLE_API_KEY"] = "XXX"

# tools
web_searcher = LlmAgent(
    model="gemini-2.0-flash",
    name="WebSearch",
    description="Performs web searches on the web for facts.",
    tools=[google_search]
)

summarizer = LlmAgent(
    model="gemini-2.0-flash",
    name="Summarizer",
    description="Summarizes text obtained from web searches.")

# search assistant
search_assistant = LlmAgent(
    name="SearchAssistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="Finds and summarizes information on a topic, Use the Websearch tool to \
    search the web and the Summarizer tool to summarize text obtained from web searches.",
    tools=[agent_tool.AgentTool(agent=web_searcher), agent_tool.AgentTool(agent=summarizer)]
)

# search planning
search_planner = LlmAgent(
    name="SearchPlanner",
    model="gemini-2.5-pro-exp-03-25",
    instruction="You are a helpful assistant. You will plan a search strategy.",
    description="Plans a search strategy. Use the SearchAssistant to find and summarize information."
)

# report writer
report_writer = LlmAgent(
    name="ReportWriter",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. You will write a report on topic X.\
    Use the SearchPlanner to plan and refine the search strategy. Use the \
    SearchAssistant to find and summarize information based on the search strategy.\
    Output *only* the final report.",
    description="Writes a report based on information gathered from the SearchPlanner.",
    tools=[agent_tool.AgentTool(agent=search_planner), agent_tool.AgentTool(agent=search_assistant)]
)

root_agent = report_writer

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai import types


async def call_agent_async(query: str, runner_instance, user_id_str, session_id_str):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner_instance.run_async(user_id=user_id_str, session_id=session_id_str, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break
    print(f"<<< Agent Response: {final_response_text}")


if __name__ == "__main__":
    print("ADK Agent setup complete. API Key configured.")
    print(f"Root agent: {root_agent.name}")

    session_service = InMemorySessionService()
    APP_NAME = "report_writer_app"
    USER_ID = "Admin_01"
    SESSION_ID = "session_001"  # Using a fixed ID for simplicity

    # Create the specific session where the conversation will happen
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=root_agent,  # The agent we want to run
        app_name=APP_NAME,  # Associates runs with our app
        session_service=session_service  # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")


    # 定义异步对话函数
    async def run_report_conversation():
        report_topic_1 = "The ethics of gene editing technologies."
        await call_agent_async(f"Write a report on: {report_topic_1}",
                               runner_instance=runner,  # 传递实例化的 runner
                               user_id_str=USER_ID,
                               session_id_str=SESSION_ID)

        # report_topic_2 = "Recent breakthroughs in fusion energy research."
        # await call_agent_async(f"Generate a comprehensive report about: {report_topic_2}",
        #                                    runner_instance=runner,
        #                                    user_id_str=USER_ID,
        #                                    session_id_str=SESSION_ID)


    # --- 执行异步对话 ---
    print("\nStarting asynchronous conversation with the ReportWriter agent...")
    try:
        asyncio.run(run_report_conversation())
    except Exception as e:
        print(f"An error occurred during the async conversation: {e}")
        import traceback

        traceback.print_exc()

    print("\nProcess finished.")
