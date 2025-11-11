from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

def main():
    # Define tools
    tools = [TavilySearch()]
    
    # Initialize model (using qwen2.5 for better tool calling support)
    llm = ChatOllama(model="qwen2.5")
    
    # Create agent with structured output using ToolStrategy
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can search for information using Tavily. After using tools to gather information, you MUST provide your final response using the structured format tool with an answer field and a sources list containing the URLs you found.",
        response_format=ToolStrategy(AgentResponse)
    )
    
    # Invoke the agent
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "search for the latest news about Tavily and summarize them in a concise manner."
        }]
    })
    
    # Access the structured response
    structured_response = result.get("structured_response")
    
    if structured_response:
        print("\n=== Answer ===")
        print(structured_response.answer)
        
        print(f"\n=== Sources ({len(structured_response.sources)}) ===")
        for i, source in enumerate(structured_response.sources, 1):
            print(f"{i}. {source.url}")
    else:
        print("\n=== Debug: No structured response ===")
        print(f"Result keys: {list(result.keys())}")
        print(f"Number of messages: {len(result.get('messages', []))}")
        for i, msg in enumerate(result.get("messages", [])):
            print(f"\n  Message {i} ({type(msg).__name__}):")
            if hasattr(msg, 'content') and msg.content:
                print(f"    Content: {msg.content[:150]}...")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"    Tool calls: {[tc.get('name', tc) for tc in msg.tool_calls]}")


if __name__ == "__main__":
    main()
