from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def get_length_of_string(s: str) -> int:
    """Returns the length of the input string."""
    print(f">> get_text_length enter with {s=}")
    s = s.strip("'\n").strip('"')
    return len(s)

if __name__ == "__main__":
    tools = [get_length_of_string]
    print(f">> Available tools: {[tool.name for tool in tools]}")

    # Use bind_tools to enable native tool calling
    llm = ChatOllama(model="qwen2.5", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [HumanMessage(content="What is the length of the word 'lion'?")]
    
    # Agent loop
    while True:
        print(f"\n>> Invoking LLM with {len(messages)} messages...")
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        print(f">> AI Message: {ai_message}")

        # Check if the model wants to call tools
        if not ai_message.tool_calls:
            print(f"\n>> Final answer: {ai_message.content}")
            break

        # Execute tool calls
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n>> Calling tool: {tool_name} with args: {tool_args}")
            
            selected_tool = {tool.name: tool for tool in tools}[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            print(f">> Tool output: {tool_output}")
            
            # Add tool response to messages
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
