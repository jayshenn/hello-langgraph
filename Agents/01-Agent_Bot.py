"""
简单机器人（Simple Bot）

目标：

1. 定义包含 HumanMessage 列表的状态结构
2. 使用 LangChain 的 ChatOpenAI 初始化 GPT-4o 模型
3. 发送和处理不同类型的消息
4. 构建并编译 Agent 图

主要目标：学习如何将 LLM 集成到 LangGraph 中
"""

from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")

llm = ChatOpenAI(model=model,base_url=base_url,api_key=api_key)

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
