"""
RAG（检索增强生成）Agent 实现
=================================

这个脚本实现了一个完整的 RAG (Retrieval-Augmented Generation) 系统：
1. 从 PDF 文档中提取文本并进行向量化存储
2. 根据用户问题检索相关文档片段
3. 将检索到的信息传递给 LLM 生成准确回答

RAG 的核心思想：让 AI 能够基于特定的文档资料回答问题，而不是仅依赖预训练知识
"""

# === 第一部分：导入必要的库 ===
from dotenv import load_dotenv  # 用于加载环境变量文件 (.env)
import os  # 操作系统接口，用于文件路径操作和环境变量获取
from langgraph.graph import StateGraph, END  # LangGraph 的图结构和结束节点
from typing import TypedDict, Annotated, Sequence  # 类型注解工具
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage  # 不同类型的消息对象
from operator import add as add_messages  # 用于消息列表的累加操作
from langchain_openai import ChatOpenAI  # OpenAI 兼容的聊天模型接口
from langchain_openai import OpenAIEmbeddings  # OpenAI 的文本嵌入模型
from langchain_community.document_loaders import PyPDFLoader  # PDF 文档加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 递归文本分割器
from langchain_chroma import Chroma  # ChromaDB 向量数据库
from langchain_core.tools import tool  # 工具装饰器，用于创建 LangChain 工具

# 加载 .env 文件中的环境变量（如 API 密钥等敏感信息）
load_dotenv()

# === 第二部分：配置 LLM 和嵌入模型 ===
# 获取 OPENAI API 配置
# OPENAI 是一个统一的 API 网关，允许我们通过一个接口访问多种 LLM 模型
# 这样可以轻松切换不同的模型（如 GPT-4、Claude 等）而无需修改代码
api_key = os.getenv("OPENAI_API_KEY")  # API 密钥
base_url = os.getenv("OPENAI_BASE_URL")  # API 基础 URL
model = os.getenv("OPENAI_MODEL")  # 要使用的具体模型名称

# 初始化语言模型
# temperature=0 让模型输出更加确定性，减少随机性和"幻觉"
# 对于需要准确信息的 RAG 应用，这很重要
llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)

# 初始化嵌入模型
# 嵌入模型将文本转换为向量，用于相似性搜索
# text-embedding-3-small 是 OpenAI 的高效嵌入模型
# 注意：嵌入模型和 LLM 模型要保持兼容性
# 这里为嵌入模型配置 API 参数，使其能够正常工作
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# === 第三部分：PDF 文档处理和向量化 ===
# PDF 文件路径：使用相对于当前脚本的路径
# 由于 PDF 在 Agents 目录中，且脚本也在 Agents 目录中，所以使用当前目录的相对路径
pdf_path = "Stock_Market_Performance_2024.pdf"

# 安全检查：确保 PDF 文件存在
# 这是一个良好的编程习惯，可以提前发现文件路径问题
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# 创建 PDF 加载器
# PyPDFLoader 可以读取 PDF 文件并提取其中的文本内容
pdf_loader = PyPDFLoader(pdf_path)

# 加载 PDF 并进行错误处理
# 这里使用 try-except 块来优雅地处理可能的加载错误
try:
    pages = pdf_loader.load()  # 加载 PDF，返回页面列表
    print(f"PDF 已成功加载，共 {len(pages)} 页")
except Exception as e:
    print(f"PDF 加载错误: {e}")
    raise

# 文本分块处理
# 为什么需要分块？
# 1. LLM 有输入长度限制
# 2. 较小的文本块能提供更精确的检索结果
# 3. 向量相似性搜索在较小文本块上效果更好
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文本块的最大字符数
    chunk_overlap=200  # 块之间的重叠字符数，确保重要信息不会被截断
)

# 将 PDF 页面分割成小的文本块
pages_split = text_splitter.split_documents(pages)

# === 第四部分：创建向量数据库 ===
# 指定向量数据库的持久化目录
# 使用相对路径在当前目录创建 chroma_db 文件夹
persist_directory = "./chroma_db"
collection_name = "stock_market"  # 向量集合的名称

# 确保存储目录存在
# 如果目录不存在，自动创建它
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # 创建 ChromaDB 向量存储
    # 这一步会：
    # 1. 将文本块转换为向量（使用嵌入模型）
    # 2. 将向量存储到 ChromaDB 中
    # 3. 建立索引以支持快速相似性搜索
    vectorstore = Chroma.from_documents(
        documents=pages_split,  # 要存储的文档块
        embedding=embeddings,  # 嵌入模型
        persist_directory=persist_directory,  # 持久化目录
        collection_name=collection_name  # 集合名称
    )
    print(f"ChromaDB 向量存储创建成功！")

except Exception as e:
    print(f"创建 ChromaDB 时出错: {str(e)}")
    raise

# 创建检索器
# 检索器负责根据查询找到最相关的文档块
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 使用相似性搜索
    search_kwargs={"k": 5}  # 返回最相似的 5 个文档块
)


# === 第五部分：定义检索工具 ===
@tool
def retriever_tool(query: str) -> str:
    """
    检索工具：根据用户查询在股票市场表现 2024 文档中搜索相关信息

    这个工具是 RAG 系统的核心组件：
    1. 接收用户的自然语言查询
    2. 将查询转换为向量
    3. 在向量数据库中搜索最相似的文档块
    4. 返回格式化的搜索结果
    """
    # 使用检索器搜索相关文档
    # retriever.invoke() 会自动处理向量化和相似性搜索
    docs = retriever.invoke(query)

    # 如果没有找到相关文档，返回友好的提示
    if not docs:
        return "在股票市场表现 2024 文档中未找到相关信息。"

    # 格式化搜索结果
    # 将多个文档块组合成易读的格式
    results = []
    for i, doc in enumerate(docs):
        results.append(f"文档片段 {i + 1}:\n{doc.page_content}")

    # 用双换行符连接所有结果，提高可读性
    return "\n\n".join(results)


# 将工具放入列表中
tools = [retriever_tool]

# 将工具绑定到 LLM
# 这样 LLM 就知道可以调用这个检索工具来获取信息
llm = llm.bind_tools(tools)


# === 第六部分：定义 Agent 状态和控制逻辑 ===
class AgentState(TypedDict):
    """
    Agent 状态定义

    这里使用 TypedDict 来定义 Agent 的状态结构：
    - messages: 存储对话历史中的所有消息
    - Annotated[Sequence[BaseMessage], add_messages]:
      * Sequence[BaseMessage] 表示消息序列类型
      * add_messages 是累加函数，用于合并新消息到现有消息列表中
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """
    判断是否需要继续执行工具调用

    这个函数检查 LLM 的最后一条回复是否包含工具调用：
    - 如果包含工具调用，返回 True，流程将转到工具执行节点
    - 如果不包含工具调用，返回 False，流程将结束
    """
    result = state['messages'][-1]  # 获取最后一条消息
    # 检查消息是否有 tool_calls 属性且不为空
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# 系统提示词定义
# 这是 RAG Agent 的"人格"和行为指南
system_prompt = """
你是一个智能 AI 助手，专门回答关于 2024 年股票市场表现的问题。你的知识来源于已加载到知识库中的 PDF 文档。

使用指南：
1. 使用可用的检索工具来查找股票市场表现数据
2. 如果需要查找更多信息，可以进行多次工具调用
3. 如果需要在回答前查找补充信息，你被允许这样做
4. 请始终在回答中引用你使用的具体文档片段
5. 基于检索到的信息提供准确、有用的回答
"""

# 创建工具字典，方便通过名称查找工具
tools_dict = {our_tool.name: our_tool for our_tool in tools}


# === 第七部分：定义 Agent 节点函数 ===

def call_llm(state: AgentState) -> AgentState:
    """
    LLM 调用节点

    这个函数负责：
    1. 准备消息历史（包括系统提示词）
    2. 调用 LLM 生成回复
    3. 返回更新后的状态

    工作流程：
    - 获取当前消息历史
    - 在消息前添加系统提示词
    - 调用 LLM 生成回复
    - 将 LLM 的回复添加到状态中
    """
    messages = list(state['messages'])  # 获取当前消息列表
    # 在消息前添加系统提示词，让 LLM 了解自己的角色和任务
    messages = [SystemMessage(content=system_prompt)] + messages
    # 调用 LLM 生成回复
    message = llm.invoke(messages)
    # 返回包含新消息的状态更新
    return {'messages': [message]}


def take_action(state: AgentState) -> AgentState:
    """
    工具执行节点

    当 LLM 决定调用工具时，这个函数负责：
    1. 提取 LLM 回复中的工具调用信息
    2. 执行相应的工具
    3. 将工具执行结果包装成 ToolMessage
    4. 返回更新后的状态

    错误处理：
    - 如果工具名称不存在，返回错误信息
    - 打印调试信息帮助开发者了解执行过程
    """
    # 从最后一条消息中提取工具调用信息
    tool_calls = state['messages'][-1].tool_calls
    results = []  # 存储工具执行结果

    # 遍历所有工具调用
    for t in tool_calls:
        query = t['args'].get('query', '无查询内容')
        print(f"调用工具: {t['name']}，查询内容: {query}")

        # 检查工具是否存在
        if t['name'] not in tools_dict:
            print(f"\n错误：工具 {t['name']} 不存在")
            result = "工具名称错误，请重试并从可用工具列表中选择。"
        else:
            # 执行工具并获取结果
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"检索结果长度: {len(str(result))} 字符")

        # 将工具执行结果包装成 ToolMessage
        # ToolMessage 是 LangChain 中专门用于表示工具执行结果的消息类型
        results.append(ToolMessage(
            tool_call_id=t['id'],  # 工具调用的唯一标识符
            name=t['name'],  # 工具名称
            content=str(result)  # 工具执行结果
        ))

    print("工具执行完成，返回到模型！")
    return {'messages': results}


# === 第八部分：构建 LangGraph 工作流 ===

# 创建状态图
# StateGraph 是 LangGraph 的核心组件，用于定义 Agent 的工作流程
graph = StateGraph(AgentState)

# 添加节点
# 每个节点代表工作流中的一个步骤
graph.add_node("llm", call_llm)  # LLM 调用节点
graph.add_node("retriever_agent", take_action)  # 工具执行节点

# 添加条件边
# 条件边根据特定条件决定下一步的流向
graph.add_conditional_edges(
    "llm",  # 起始节点
    should_continue,  # 判断函数
    {
        True: "retriever_agent",  # 如果需要调用工具，转到工具执行节点
        False: END  # 如果不需要调用工具，结束流程
    }
)

# 添加普通边
# 工具执行完成后，返回到 LLM 节点让 LLM 基于工具结果生成最终回答
graph.add_edge("retriever_agent", "llm")

# 设置入口点
# 工作流从 LLM 节点开始
graph.set_entry_point("llm")

# 编译图为可执行的 Agent
# compile() 方法将图结构转换为可以直接调用的 Agent 对象
rag_agent = graph.compile()


# === 第九部分：运行 Agent 的主函数 ===

def running_agent():
    """
    RAG Agent 的主运行循环

    这个函数提供了一个简单的命令行界面，让用户可以：
    1. 输入关于股票市场的问题
    2. 获得基于文档的准确回答
    3. 通过输入 'exit' 或 'quit' 退出程序

    工作流程：
    - 接收用户输入
    - 将输入包装成 HumanMessage
    - 调用 RAG Agent 处理
    - 显示 Agent 的回答
    """
    print("\n=== RAG 智能助手 ===")
    print("我可以回答关于 2024 年股票市场表现的问题")
    print("输入 'exit' 或 'quit' 退出程序\n")

    while True:
        # 获取用户输入
        user_input = input("请输入你的问题: ")

        # 检查是否要退出
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("再见！")
            break

        # 将用户输入转换为 LangChain 消息格式
        # HumanMessage 表示来自人类用户的消息
        messages = [HumanMessage(content=user_input)]

        # 调用 RAG Agent 处理用户问题
        # invoke() 方法会执行整个工作流程：
        # 1. LLM 分析问题
        # 2. 如果需要，调用检索工具搜索文档
        # 3. LLM 基于检索结果生成最终回答
        result = rag_agent.invoke({"messages": messages})

        # 显示 Agent 的回答
        print("\n=== 回答 ===")
        print(result['messages'][-1].content)
        print()  # 添加空行以提高可读性


# 启动 RAG Agent
# 这行代码会开始运行交互式问答循环
# 注意：确保 PDF 文件存在且环境变量已正确配置
if __name__ == "__main__":
    running_agent()