# uv add python-dotenv black isort langchain langchain-core langchain-ollama langchainhub langchain-community
from dotenv import load_dotenv
load_dotenv()

import os

# In LangChain 1.0.4, AgentExecutor and create_react_agent have been removed/deprecated. You need to install the langchain-community package which contains the legacy agents
# from langsmith import hub # throws error, can't find hub
from langsmith import Client
# The agents are now in langchain_classic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch 

tools = [TavilySearch()]
llm = ChatOllama(model="mistral")
client = Client()
# react_prompt = hub.pull('hwchase17/react') # throws error, can't find 'pull' funcition in hub
react_prompt = client.pull_prompt("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    print("Hello Tavily")
    print(os.getenv("TAVILY_API_KEY"))
    result = chain.invoke(input={'input': 'search for the latest news about Tavily and summarize them in a concise manner.'})

if __name__ == "__main__":
    main()
