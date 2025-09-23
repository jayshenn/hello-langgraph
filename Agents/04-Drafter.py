"""
起草助手（DRAFTER）

**任务（Boss’s Orders）：**

我们的公司效率低下！起草文档、邮件花了太多时间。

你的任务：

* 构建一个 **AI 文档起草系统**
* 支持 **人机协作（Human-AI Collaboration）**：人类可持续提供反馈，AI 根据反馈修改草稿
* 当人类确认满意时，AI 停止
* 系统需快速且能保存草稿
"""

# 导入必要的库
from typing import Annotated, Sequence, TypedDict  # Python 类型注解
from dotenv import load_dotenv  # 加载环境变量
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage  # LangChain 消息类型
from langchain_openai import ChatOpenAI  # OpenAI 模型接口
from langchain_core.tools import tool  # 工具装饰器
from langgraph.graph.message import add_messages  # 消息累加器
from langgraph.graph import StateGraph, END  # 图结构和结束节点
from langgraph.prebuilt import ToolNode  # 预构建的工具节点
import os

# 加载 .env 文件中的环境变量（如 API 密钥）
load_dotenv()

# 获取 OPENAI API 配置
# OPENAI 是一个统一的 API 网关，可以访问多种 LLM 模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

# 初始化语言模型
llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

# 全局变量：存储文档内容
# 这个变量会在整个程序运行期间保存和更新文档的内容
document_content = "Unable_to_Attend_Meeting_Email.txt"


class AgentState(TypedDict):
    """Agent 的状态定义

    这是一个 TypedDict，用于定义 Agent 在整个对话过程中的状态。
    在 LangGraph 中，状态会在不同的节点之间传递和更新。
    """
    # messages 字段存储所有的对话消息
    # Annotated 和 add_messages 确保新消息会被添加到列表中而不是替换整个列表
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool  # 装饰器将这个函数转换为 LangChain 工具
def update(content: str) -> str:
    """更新文档内容的工具函数

    这个工具允许 AI Agent 更新存储在内存中的文档内容。

    Args:
        content: 要更新的新文档内容

    Returns:
        更新成功的消息和当前文档内容
    """
    global document_content  # 使用全局变量
    document_content = content  # 更新文档内容
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool  # 装饰器将这个函数转换为 LangChain 工具
def save(filename: str) -> str:
    """保存文档到文件的工具函数

    将当前的文档内容保存到指定的文本文件中，并结束对话流程。

    Args:
        filename: 要保存的文件名（会自动添加 .txt 扩展名）

    Returns:
        保存成功或失败的消息
    """
    global document_content  # 访问全局文档内容

    # 确保文件名以 .txt 结尾
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        # 将文档内容写入文件
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        # 处理可能的文件写入错误
        return f"Error saving document: {str(e)}"


# 定义 Agent 可以使用的工具列表
tools = [update, save]

# 初始化 LLM 模型并绑定工具
# bind_tools 让模型知道可以调用哪些工具
model = llm.bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    """Agent 节点函数

    这是图中的主要节点，负责：
    1. 接收用户输入
    2. 调用 LLM 生成响应
    3. 决定是否调用工具

    Args:
        state: 当前的 Agent 状态

    Returns:
        更新后的状态（包含新的消息）
    """
    # 创建系统提示词，定义 Agent 的行为和职责
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{document_content}
    """)

    # 处理用户输入
    if not state["messages"]:
        # 如果是对话的开始（没有消息历史），提供欢迎消息
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        # 如果对话已经开始，从用户获取输入
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")  # 显示用户输入
        user_message = HumanMessage(content=user_input)

    # 构建完整的消息列表：系统提示 + 历史消息 + 当前用户消息
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # 调用 LLM 生成响应（可能包含工具调用）
    response = model.invoke(all_messages)

    # 打印 AI 的响应内容
    print(f"\n🤖 AI: {response.content}")
    # 如果 AI 决定使用工具，显示使用了哪些工具
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # 返回更新后的状态，包含用户消息和 AI 响应
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """条件函数：决定对话是否继续

    这个函数检查最近的消息，判断是否应该结束对话。
    当文档被成功保存后，对话就会结束。

    Args:
        state: 当前的 Agent 状态

    Returns:
        "continue" 继续对话，或 "end" 结束对话
    """
    messages = state["messages"]

    # 如果还没有消息，继续对话
    if not messages:
        return "continue"

    # 从最新的消息开始，向前查找工具消息
    for message in reversed(messages):
        # 检查是否有保存文档的工具消息
        # 如果找到包含 "saved" 和 "document" 的工具消息，说明文档已保存
        if (isinstance(message, ToolMessage) and
                "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end"  # 结束对话流程

    return "continue"  # 默认继续对话


def print_messages(messages):
    """辅助函数：以更易读的格式打印消息

    只打印最近的工具消息结果，避免输出过于冗长。

    Args:
        messages: 消息列表
    """
    if not messages:
        return

    # 只查看最近的3条消息
    for message in messages[-3:]:
        # 如果是工具消息，打印工具执行结果
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# ===== 构建 LangGraph 工作流 =====

# 创建状态图，定义 Agent 的工作流程
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent", our_agent)  # Agent 节点：处理用户输入和生成响应
graph.add_node("tools", ToolNode(tools))  # 工具节点：执行工具调用

# 设置入口点：对话从 agent 节点开始
graph.set_entry_point("agent")

# 添加边：从 agent 到 tools（当 Agent 决定使用工具时）
graph.add_edge("agent", "tools")

# 添加条件边：工具执行后的路由
# - 如果 should_continue 返回 "continue"，回到 agent 节点
# - 如果返回 "end"，结束整个流程
graph.add_conditional_edges(
    "tools",  # 从工具节点出发
    should_continue,  # 使用这个函数决定下一步
    {
        "continue": "agent",  # 继续对话
        "end": END,  # 结束流程
    },
)

# 编译图，生成可执行的应用
app = graph.compile()


def run_document_agent():
    """主运行函数

    启动文档编辑 Agent，处理用户交互直到文档被保存。
    """
    print("\n ===== DRAFTER =====")

    # 初始化空状态（没有消息历史）
    state = {"messages": []}

    # 流式运行应用，处理每一步的输出
    # stream_mode="values" 表示返回每步后的完整状态
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            # 打印工具执行结果
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


# 程序入口点
# 当直接运行这个脚本时（而不是被导入），启动 Agent
if __name__ == "__main__":
    run_document_agent()
