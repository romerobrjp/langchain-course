from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class SimpleResponse(BaseModel):
    answer: str

# llm = ChatOllama(model="mistral")
llm = ChatOllama(model="qwen2.5")

agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(SimpleResponse)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})

print("Result keys:", result.keys())
print("\nStructured response:", result.get("structured_response"))
print("\nAll messages:")
for msg in result.get("messages", []):
    print(f"  - {type(msg).__name__}: {msg.content[:100] if hasattr(msg, 'content') and msg.content else 'No content'}")
