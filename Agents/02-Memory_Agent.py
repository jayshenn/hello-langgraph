"""
聊天机器人（Chatbot）

目标：

1. 使用不同类型的消息（HumanMessage, AIMessage）
2. 使用这两种消息维护完整对话历史
3. 使用 GPT-4o 模型（ChatOpenAI）
4. 创建一个更复杂的对话循环

主要目标：为 Agent 创建记忆机制

数据流向对比：
01-Agent_Bot.py：  用户输入 → [单条HumanMessage] → LLM → 打印 → 丢弃
02-Memory_Agent.py： 用户输入 → [完整历史] → LLM → 添加AI回复 → 更新历史
"""

# === 与 01-Agent_Bot.py 的关键区别总结 ===
# ✅ 支持多种消息类型：HumanMessage + AIMessage
# ✅ 实现状态累积更新（记忆机制）
# ✅ 维护完整对话历史
# ✅ 对话持久化功能
# ✅ LLM 能看到上下文，实现连贯对话

# === 系统和环境相关导入 ===
import os  # 系统操作接口
from dotenv import load_dotenv  # 加载 .env 环境变量

# === 类型注解相关导入 ===
# 相比 01-Agent_Bot.py，新增了 Union 类型
from typing import TypedDict, List, Union  # Union: 允许一个变量可以是多种类型之一

# === LangChain 消息相关导入 ===
# 相比 01-Agent_Bot.py，新增了 AIMessage
from langchain_core.messages import HumanMessage, AIMessage  # AIMessage: AI 回复消息类，用于存储 AI 的响应
from langchain_openai import ChatOpenAI  # OpenAI 聊天模型接口

# === LangGraph 相关导入 ===
from langgraph.graph import StateGraph, START, END  # LangGraph 状态图构建器

# === 加载环境配置 ===
load_dotenv()  # 从 .env 文件加载 API 密钥等敏感信息

# === 读取 OPENAI 配置 ===
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

# === 初始化 LLM 实例 ===
llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

# === 定义升级版状态结构 ===
# 相比 01-Agent_Bot.py 的关键升级：支持多种消息类型
class AgentState(TypedDict):
    # 原版：messages: List[HumanMessage]  ← 只支持用户消息
    # 新版：同时支持用户消息和 AI 消息，实现完整对话历史
    messages: List[Union[HumanMessage, AIMessage]]  # Union 表示可以是 HumanMessage 或 AIMessage 中的任意一种

# === 定义升级版处理节点函数 ===
# 相比 01-Agent_Bot.py 的核心改进：实现状态的累积更新（记忆机制）
def process(state: AgentState) -> AgentState:
    """
    升级版处理函数 - 实现对话记忆的关键节点

    与 01-Agent_Bot.py 的区别：
    - 原版：不修改状态，直接返回原状态
    - 新版：将 AI 的回复添加到状态中，实现记忆累积

    参数:
        state: 包含完整对话历史的状态

    返回:
        更新后的状态（包含新的 AI 回复）

    数据流向: 历史状态 -> LLM -> AI回复 -> 更新状态
    """
    # 步骤 1：将完整对话历史发送给 LLM（LLM 能看到上下文）
    response = llm.invoke(state["messages"])  # 这里的 invoke 是 LangChain 的方法

    # 步骤 2：关键改进！将 AI 的回复添加到状态中（实现记忆）
    state["messages"].append(AIMessage(content=response.content))
    # 原版没有这一步，所以不会保存 AI 的回复

    # 步骤 3：打印 AI 的回复给用户查看
    print(f"\nAI: {response.content}")

    # 步骤 4：调试输出 - 显示当前状态中的所有消息
    print("CURRENT STATE: ", state["messages"])

    # 步骤 5：返回更新后的状态（包含了新的 AI 回复）
    return state

# === 构建状态图（与 01-Agent_Bot.py 相同） ===
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# 执行流程仍然是：START → process → END

# === 对话历史管理（核心新增功能） ===
# 相比 01-Agent_Bot.py，新增了在程序外部维护对话历史
conversation_history = []  # 存储完整对话历史的列表

# === 升级版交互循环（实现记忆机制） ===
user_input = input("Enter: ")  # 获取用户输入

while user_input != "exit":
    # 步骤 1：将用户新消息添加到对话历史
    conversation_history.append(HumanMessage(content=user_input))

    # 步骤 2：调用 agent.invoke() 并传入完整历史
    # 注意：这里传入的是完整的 conversation_history，而不是单条消息
    result = agent.invoke({"messages": conversation_history})  # 这里的 invoke 是 LangGraph 的方法

    # 步骤 3：更新本地对话历史（包含了 process 函数添加的 AI 回复）
    conversation_history = result["messages"]  # 获取更新后的状态

    # 步骤 4：等待下一次用户输入
    user_input = input("Enter: ")

# === 对话持久化功能（全新功能） ===
# 01-Agent_Bot.py 没有这个功能，这是 Memory Agent 的重要特性
with open("conversation_history.txt", "w", encoding="utf-8") as file:
    # 写入文件头部
    file.write("对话历史记录：\n")
    file.write("=" * 50 + "\n\n")

    # 遍历所有消息并根据类型格式化输出
    for conversation in conversation_history:
        # 使用 isinstance 检查消息类型
        if isinstance(conversation, HumanMessage):
            file.write(f"🗨️ 用户: {conversation.content}\n")
        elif isinstance(conversation, AIMessage):
            file.write(f"🤖 AI: {conversation.content}\n")
        file.write("-" * 30 + "\n")  # 添加分隔线

    # 文件结尾
    file.write("\n对话结束")

# 提示用户文件保存成功
print("\n💾 对话历史已保存到 conversation_history.txt")