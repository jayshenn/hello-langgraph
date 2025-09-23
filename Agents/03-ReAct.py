"""
ReAct Agent（推理 + 行动代理）

目标：

1. 学习如何在 LangGraph 中创建工具（Tools）
2. 构建一个 ReAct 图
3. 使用不同类型的消息（如 ToolMessages）
4. 测试图的鲁棒性

主要目标：构建一个稳健的 ReAct Agent

ReAct 是一种将推理（Reasoning）和行动（Acting）相结合的代理架构。
Agent 会先思考要做什么，然后执行相应的工具，根据工具返回的结果继续推理。
"""

# 导入必要的类型注解和库
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage  # LangGraph 中所有消息类型的基础类
from langchain_core.messages import ToolMessage  # 工具调用后返回数据给 LLM 的消息类型（包含内容和 tool_call_id）
from langchain_core.messages import SystemMessage  # 用于向 LLM 提供系统指令的消息类型
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool  # 装饰器，用于定义工具函数
from langgraph.graph.message import add_messages  # 消息累加器，用于管理对话历史
from langgraph.graph import StateGraph, END  # StateGraph: 状态图类，END: 图的结束节点
from langgraph.prebuilt import ToolNode  # 预构建的工具节点，自动处理工具调用
import os


# 加载环境变量（从 .env 文件中读取配置）
load_dotenv()

# 获取 OpenRouter API 配置
# OpenRouter 是一个统一的 API 网关，可以访问多种 LLM 模型
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model = os.getenv("OPENROUTER_MODEL")

# 初始化语言模型
llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

# 定义 Agent 的状态结构
# AgentState 用于在图的不同节点之间传递和维护状态
class AgentState(TypedDict):
    """Agent 的状态定义，包含消息历史"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Annotated 用于添加元数据
    # add_messages 是一个特殊的累加器函数，自动管理消息的追加和更新

# 定义工具函数
# @tool 装饰器将普通函数转换为 LangChain 工具
# 工具函数允许 LLM 执行具体的操作

@tool
def add(a: int, b: int):
    """这是一个将两个数字相加的加法函数"""
    # 函数的 docstring 会作为工具的描述，帮助 LLM 理解何时使用此工具
    return a + b

@tool
def subtract(a: int, b: int):
    """减法函数：计算 a - b 的结果"""
    return a - b

@tool
def multiply(a: int, b: int):
    """乘法函数：计算 a × b 的结果"""
    return a * b

# 将所有工具收集到一个列表中
tools = [add, subtract, multiply]

# 将工具绑定到 LLM
# bind_tools 让 LLM 知道可以调用哪些工具
llm = llm.bind_tools(tools)

# 定义模型调用节点
# 这是 Agent 的核心节点，负责调用 LLM 并处理响应
def model_call(state: AgentState) -> AgentState:
    """
    调用 LLM 处理当前状态中的消息

    参数:
        state: 当前的 Agent 状态，包含消息历史

    返回:
        更新后的状态，包含 LLM 的响应
    """
    # 创建系统提示，定义 AI 助手的角色和行为
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")

    # 调用 LLM，传入系统提示和历史消息
    # LLM 会分析消息并决定是直接回答还是调用工具
    response = llm.invoke([system_prompt] + state["messages"])

    # 返回更新后的状态，将 LLM 的响应添加到消息历史中
    return {"messages": [response]}

# 定义路由函数
# 决定执行流程的下一步：继续调用工具还是结束
def should_continue(state: AgentState) -> str:
    """
    根据最后一条消息判断是否需要继续执行

    参数:
        state: 当前的 Agent 状态

    返回:
        "continue": 如果需要调用工具
        "end": 如果可以结束执行
    """
    # 获取消息历史
    messages = state["messages"]
    # 获取最后一条消息（LLM 的最新响应）
    last_message = messages[-1]

    # 检查最后一条消息是否包含工具调用请求
    if not last_message.tool_calls:
        # 没有工具调用，说明 LLM 已经得出最终答案，结束执行
        return "end"
    else:
        # 有工具调用请求，需要执行工具并继续
        return "continue"

# 构建状态图
# StateGraph 是 LangGraph 的核心组件，用于定义执行流程
graph = StateGraph(AgentState)

# 添加节点到图中
# "agent" 节点：调用 LLM 进行推理
graph.add_node("agent", model_call)

# 创建工具节点
# ToolNode 会自动处理工具的执行和结果返回
tool_node = ToolNode(tools=tools)
# 添加 "tools" 节点：执行工具调用
graph.add_node("tools", tool_node)

# 设置图的入口点
# 执行总是从 "agent" 节点开始
graph.set_entry_point("agent")

# 添加条件边
# 根据 should_continue 函数的返回值决定下一步
graph.add_conditional_edges(
    "agent",  # 从 agent 节点出发
    should_continue,  # 使用这个函数判断路径
    {
        "continue": "tools",  # 如果返回 "continue"，去执行工具
        "end": END,  # 如果返回 "end"，结束执行
    }
)

# 添加普通边
# 工具执行完成后，总是返回到 agent 节点
# 这形成了一个循环：Agent 思考 → 执行工具 → Agent 再思考
graph.add_edge("tools", "agent")

# 编译图
# 将图编译成可执行的应用
app = graph.compile()

# 定义打印流输出的辅助函数
def print_stream(stream):
    """
    打印流式输出的消息

    参数:
        stream: 从图执行返回的流对象
    """
    for s in stream:
        # 获取每个步骤中的最新消息
        message = s["messages"][-1]
        if isinstance(message, tuple):
            # 如果是元组格式（用户输入），直接打印
            print(message)
        else:
            # 使用 pretty_print 方法美化输出
            # 这会自动格式化不同类型的消息（AI响应、工具调用、工具结果等）
            message.pretty_print()

# 准备输入
# 这个复杂的查询要求 Agent：
# 1. 计算 40 + 12（需要调用 add 工具）
# 2. 将结果乘以 6（需要调用 multiply 工具）
# 3. 还要讲一个笑话（测试 Agent 处理混合任务的能力）
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}

# 执行图并打印流式输出
# stream_mode="values" 表示返回每个节点执行后的完整状态值
print("=" * 50)
print("开始执行 ReAct Agent...")
print("=" * 50)
print_stream(app.stream(inputs, stream_mode="values"))
print("=" * 50)
print("执行完成！")
print("=" * 50)