import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper
from agents.tool import function_tool
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


@function_tool
def add(a:int, b:int)->int:
    return a+b

agent = Agent(
    name="triage_agent",
    instructions="you are a health agent and you help the patient and where you are confuse you can handoff the agent and say them that theycan give me answer and you can use tool",
    tools=[add]
)

result=Runner.run_sync(
    agent,"add 2+2",run_config=run_config
)


