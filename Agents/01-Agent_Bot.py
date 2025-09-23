"""
简单机器人（Simple Bot）

目标：

1. 定义包含 HumanMessage 列表的状态结构
2. 使用 LangChain 的 ChatOpenAI 初始化 GPT-4o 模型
3. 发送和处理不同类型的消息
4. 构建并编译 Agent 图

主要目标：学习如何将 LLM 集成到 LangGraph 中
"""

# === 类型注解相关导入 ===
from typing import TypedDict, List  # TypedDict: 定义带类型的字典结构, List: 列表类型注解

# === LangChain 相关导入 ===
from langchain_core.messages import HumanMessage  # 人类消息类，用于封装用户输入
from langchain_openai import ChatOpenAI  # OpenAI 聊天模型接口，用于调用 LLM

# === LangGraph 相关导入 ===
from langgraph.graph import StateGraph, START, END  # StateGraph: 状态图构建器, START/END: 预定义的起始/结束节点

# === 环境变量和配置相关导入 ===
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量
import os  # 操作系统接口，用于读取环境变量

# === 加载环境配置 ===
load_dotenv()  # 从 .env 文件中加载环境变量（API 密钥等敏感信息）

# === 读取 OPENAI 配置 ===
# OPENAI 是一个提供统一 API 访问多种 LLM 的服务
api_key = os.getenv("OPENAI_API_KEY")    # API 密钥，用于身份认证
base_url = os.getenv("OPENAI_BASE_URL")  # API 基础 URL，指向 OPENAI 服务
model = os.getenv("OPENAI_MODEL")        # 指定使用的具体模型（如 GPT-4, Claude 等）

# === 初始化 LLM 实例 ===
# 创建一个可以调用大语言模型的对象
llm = ChatOpenAI(
    model=model,        # 使用的模型名称
    base_url=base_url,  # API 地址
    api_key=api_key     # 身份认证密钥
)

# === 定义状态结构 ===
# AgentState 是整个图执行过程中共享的数据结构
# 类似于一个全局变量，所有节点都可以读取和修改
class AgentState(TypedDict):
    messages: List[HumanMessage]  # 存储对话消息的列表，这里只包含用户消息

# === 定义处理节点函数 ===
# 这是图中的一个节点，负责处理用户消息并获取 AI 回复
def process(state: AgentState) -> AgentState:
    """
    处理函数 - LangGraph 图中的核心节点

    参数:
        state: 当前的状态，包含用户消息列表

    返回:
        更新后的状态（这里没有修改，直接返回原状态）

    数据流向: 状态 -> LLM -> 打印回复 -> 返回状态
    """
    # 调用 LLM 处理消息列表，获取 AI 的回复
    response = llm.invoke(state["messages"])  # 这里的 invoke 是 LangChain 的方法

    # 打印 AI 的回复内容给用户查看
    print(f"\nAI: {response.content}")

    # 返回状态（注意：这里没有修改状态，所以不会保存对话历史）
    return state

# === 构建和编译状态图 ===
# 这是 LangGraph 的核心部分：定义工作流的结构

# 步骤 1：创建状态图实例
graph = StateGraph(AgentState)  # 使用 AgentState 作为状态类型的图

# 步骤 2：添加处理节点
graph.add_node("process", process)  # 注册 process 函数为一个名为 "process" 的节点

# 步骤 3：定义节点间的连接关系（数据流向）
graph.add_edge(START, "process")     # 从起始节点连接到 process 节点
graph.add_edge("process", END)       # 从 process 节点连接到结束节点

# 步骤 4：编译图为可执行的 Agent
# 编译后就可以通过 invoke 方法执行整个流程
agent = graph.compile()

# 执行流程图：START → process → END

# === 主程序：用户交互循环 ===
# 这是程序的入口点，实现与用户的持续对话

# 获取用户的第一次输入
user_input = input("Enter: ")  # 等待用户输入文本

# 开始交互循环，直到用户输入 "exit" 退出
while user_input != "exit":
    # 步骤 1：将用户输入封装为符合 AgentState 结构的状态
    initial_state = {
        "messages": [HumanMessage(content=user_input)]  # 创建包含用户消息的状态
    }

    # 步骤 2：调用 agent.invoke() 启动整个图的执行
    # 执行流程：START → process节点（调用LLM并打印回复）→ END
    agent.invoke(initial_state)  # 这里的 invoke 是 LangGraph 的方法

    # 步骤 3：等待用户下一次输入
    user_input = input("Enter: ")  # 再次等待用户输入

# 程序结束提示
print("\n程序已退出，再见！")
