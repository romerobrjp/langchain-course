# ReACT Agents vs Tool Calling: Principais Diferenças

## 1. **Prompts vs Capacidade Nativa**

**ReACT**: Usa um prompt texto extenso com formato Thought/Action/Observation para "ensinar" o modelo a usar ferramentas

```python
template = """
Answer the following questions as best as you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template=template)
```

**Tool Calling**: O modelo tem suporte nativo para ferramentas via `.bind_tools()` - não precisa de prompt especial

```python
llm = ChatOllama(model="qwen2.5", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

---

## 2. **Parsing de Respostas**

**ReACT**: Precisa de um parser customizado (`ReActSingleInputOutputParser`) para extrair actions/thoughts da resposta em texto

```python
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

agent = prompt | llm | ReActSingleInputOutputParser()
agent_step = agent.invoke({"input": "...", "agent_scratchpad": "..."})

# Retorna AgentAction ou AgentFinish parseado da string
if isinstance(agent_step, AgentAction):
    tool_name = agent_step.tool
    tool_input = agent_step.tool_input
```

**Tool Calling**: O modelo retorna `tool_calls` estruturados diretamente no `AIMessage` - sem parsing manual

```python
ai_message = llm_with_tools.invoke(messages)

# tool_calls já vem estruturado!
for tool_call in ai_message.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
```

---

## 3. **Formato de Comunicação**

**ReACT**: Usa dicionários com strings

```python
agent.invoke({
    "input": "What is the length of the word 'lion'?",
    "agent_scratchpad": intermediate_steps
})
```

**Tool Calling**: Usa mensagens tipadas

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

messages: list[BaseMessage] = [
    HumanMessage(content="What is the length of the word 'lion'?")
]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

messages.append(ToolMessage(
    content=str(tool_output),
    tool_call_id=tool_call["id"]
))
```

---

## 4. **Histórico de Execução**

**ReACT**: Concatena tudo em string no `agent_scratchpad`

```python
intermediate_steps = []

intermediate_steps.append(
    f"\n\nThought: {agent_step.log}\n"
    f"Action: {tool_name}\n"
    f"Action Input: {tool_input}\n"
    f"Observation: {observation}\n"
)

# Passa como string no próximo invoke
agent.invoke({
    "input": "...",
    "agent_scratchpad": intermediate_steps  # String concatenada
})
```

**Tool Calling**: Mantém histórico como lista de mensagens - mais limpo e estruturado

```python
messages: list[BaseMessage] = []

# Adiciona mensagens de forma estruturada
messages.append(HumanMessage(content="..."))
messages.append(AIMessage(content="...", tool_calls=[...]))
messages.append(ToolMessage(content="...", tool_call_id="..."))

# Passa lista de mensagens
llm_with_tools.invoke(messages)
```

---

## 5. **Detecção de Conclusão**

**ReACT**: Verifica se `agent_step` é `AgentFinish`

```python
from langchain_core.agents import AgentAction, AgentFinish

agent_step = None

while not isinstance(agent_step, AgentFinish):
    agent_step = agent.invoke(...)
    
    if isinstance(agent_step, AgentAction):
        # Executa ferramenta
        pass

if isinstance(agent_step, AgentFinish):
    print(f"Final answer: {agent_step.return_values}")
```

**Tool Calling**: Verifica se `ai_message.tool_calls` está vazio

```python
while True:
    ai_message = llm_with_tools.invoke(messages)
    
    if not ai_message.tool_calls:
        # Sem tool calls = resposta final
        print(f"Final answer: {ai_message.content}")
        break
    
    # Executa tool calls
    for tool_call in ai_message.tool_calls:
        # ...
```

---

## 6. **Confiabilidade**

**ReACT**: Depende do modelo seguir o formato do prompt corretamente (pode falhar se o modelo não gerar o formato exato)

```python
# Se o modelo gerar:
# "Thought: Let me think... Oh wait, the Final Answer is 4"
# ❌ Parser pode falhar ao extrair Action/Action Input
```

**Tool Calling**: API estruturada - mais robusta e previsível, com suporte nativo do modelo

```python
# O modelo sempre retorna tool_calls no formato correto:
# {
#   "name": "get_length_of_string",
#   "args": {"s": "lion"},
#   "id": "call_123"
# }
# ✅ Estrutura garantida pela API
```

---

## Resumo

### Tool Calling (Recomendado)
- ✅ Mais simples (menos código)
- ✅ Mais confiável (API estruturada)
- ✅ Mais limpo (mensagens tipadas)
- ✅ Melhor manutenibilidade
- ✅ Suporte nativo do modelo

### ReACT (Legado)
Ainda é útil para:
- Modelos sem suporte nativo a tool calling
- Casos onde você quer controle total sobre o formato de raciocínio
- Estudos acadêmicos sobre prompt engineering
- Debugging detalhado do processo de pensamento do modelo

---

## Exemplo Completo: Tool Calling

```python
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama

@tool
def get_length_of_string(s: str) -> int:
    """Returns the length of the input string."""
    return len(s)

# Setup
tools = [get_length_of_string]
llm = ChatOllama(model="qwen2.5", temperature=0)
llm_with_tools = llm.bind_tools(tools)

messages: list[BaseMessage] = [
    HumanMessage(content="What is the length of the word 'lion'?")
]

# Agent loop
while True:
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    
    if not ai_message.tool_calls:
        print(f"Final answer: {ai_message.content}")
        break
    
    for tool_call in ai_message.tool_calls:
        selected_tool = {tool.name: tool for tool in tools}[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        
        messages.append(ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"]
        ))
```
