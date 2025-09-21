"""
聊天机器人（Chatbot）

目标：

1. 使用不同类型的消息（HumanMessage, AIMessage）
2. 使用这两种消息维护完整对话历史
3. 使用 GPT-4o 模型（ChatOpenAI）
4. 创建一个更复杂的对话循环

主要目标：为 Agent 创建记忆机制
"""

import os
from typing import TypedDict, List, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")

llm = ChatOpenAI(model=model,base_url=base_url,api_key=api_key)

class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("conversation_history.txt","w") as file:
    file.write("Your Conversation Log:\n")

    for conversation in conversation_history:
        if isinstance(conversation, HumanMessage):
            file.write(f"You: {conversation.content}\n")
        elif isinstance(conversation, AIMessage):
            file.write(f"AI: {conversation.content}\n")
    file.write("End of Conversation")

print("Conversation saved to conversation_history.txt")