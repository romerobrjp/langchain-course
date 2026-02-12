from langchain.tools import tool
from langchain_core.tools import render_text_description
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from typing import Sequence, Union
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

if __name__ == "__main__":
    print("Hello Tool Calling")

    tools = [TavilySearchResults(), multiply]
    llm = ChatOllama(model="qwen2.5", temperature=0, stop=["\nObservation", "Observation", "Observation:"])
    llm = ChatOllama(model="mistral", temperature=0)

    # Create agent using LangGraph
    agent = create_react_agent(llm, tools)

    res = agent.invoke(
        {
            "messages": [("user", "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious")],
        }
    )

    print(res)
