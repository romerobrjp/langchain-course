# uv add python-dotenv black isort langchain langchain-core langchain-ollama langchainhub langchain-community
from dotenv import load_dotenv

load_dotenv()

import os

# The agents are now in langchain_classic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# In LangChain 1.0.4, AgentExecutor and create_react_agent have been removed/deprecated. You need to install the langchain-community package which contains the legacy agents
# from langsmith import hub # throws error, can't find hub
from langsmith import Client

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOllama(model="mistral")
structured_llm = llm.with_structured_output(AgentResponse) # it will create a new instance of the model capable to produce outputs in the desired structured format
client = Client()
# react_prompt = hub.pull('hwchase17/react') # throws error, can't find 'pull' funcition in hub
react_prompt = client.pull_prompt("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools_names"],
    partial_variables={
        "format_instructions": ""
    },
)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

extract_output = RunnableLambda(lambda x: x["output"])

chain = agent_executor | extract_output | structured_llm

def main():
    print("Hello Tavily")
    print(os.getenv("TAVILY_API_KEY"))
    result = chain.invoke(
        input={
            "input": "search for the latest news about Tavily and summarize them in a concise manner."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
