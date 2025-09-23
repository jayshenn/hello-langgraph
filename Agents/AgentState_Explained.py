"""
AgentState 类详解示例

本文件展示 AgentState 类中每个组件的详细用法和作用：

1. TypedDict - 类型化字典基础
2. Sequence vs List 的区别
3. BaseMessage 及其子类
4. Annotated 深入理解 - 核心概念
5. add_messages reducer 的作用
6. LangGraph 中的实际应用
7. 类型注解的好处

核心要点：
- Annotated 为类型添加元数据，指定状态更新的处理方式
- 解决并行更新冲突，防止数据丢失
- 支持自定义 reducer 函数，实现复杂的状态合并逻辑
"""

from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# 1. TypedDict - 类型化字典基础示例
print("=" * 50)
print("1. TypedDict 基础示例")
print("=" * 50)

class BasicState(TypedDict):
    count: int
    name: str

# 创建状态实例（实际上是字典）
state = BasicState(count=1, name="测试")
print(f"BasicState 实例: {state}")
print(f"类型: {type(state)}")  # 实际上是 dict
print(f"访问字段: count={state['count']}, name={state['name']}")

# 2. Sequence vs List 的区别
print("\n" + "=" * 50)
print("2. Sequence vs List 的区别")
print("=" * 50)

# Sequence 可以接受任何有序可迭代对象
def process_sequence(items: Sequence[str]) -> None:
    print(f"处理序列: {list(items)}")

# 这些都可以传入 Sequence
process_sequence(["a", "b", "c"])  # list
process_sequence(("a", "b", "c"))  # tuple
process_sequence("abc")            # string

# List 只能接受 list
def process_list(items: List[str]) -> None:
    print(f"处理列表: {items}")

process_list(["a", "b", "c"])  # OK
# process_list(("a", "b", "c"))  # 类型检查会报错（运行时可以）

# 3. BaseMessage 及其子类
print("\n" + "=" * 50)
print("3. BaseMessage 及其子类")
print("=" * 50)

# 创建不同类型的消息
human_msg = HumanMessage(content="你好，我是用户")
ai_msg = AIMessage(content="你好！我是AI助手")
system_msg = SystemMessage(content="你是一个helpful assistant")

# 展示消息属性
messages = [human_msg, ai_msg, system_msg]
for msg in messages:
    print(f"类型: {type(msg).__name__}")
    print(f"内容: {msg.content}")
    print(f"是否为BaseMessage: {isinstance(msg, BaseMessage)}")
    print("-" * 30)

# 4. Annotated 深入理解 - 核心概念
print("\n" + "=" * 50)
print("4. Annotated 深入理解 - 核心概念")
print("=" * 50)

"""
Annotated 的本质：为类型添加元数据（metadata）

语法：Annotated[类型, 元数据1, 元数据2, ...]

基础语法对比：
- 普通类型提示：messages: List[BaseMessage]
- 使用 Annotated：messages: Annotated[List[BaseMessage], add_messages]
                          └─类型部分─┘    └──元数据（reducer）──┘

在 LangGraph 中的作用：
1. 没有 Annotated：新值直接覆盖旧值
2. 有 Annotated：使用 reducer 函数智能合并新旧值

这是解决并行更新冲突的关键机制！
"""

# 实际示例
from typing import Dict, Set
from operator import add as operator_add

# 定义不同的 reducer 函数
def max_reducer(existing: float, new: float) -> float:
    """保留最大值"""
    return max(existing or 0, new or 0)

def set_union_reducer(existing: Set[str], new: Set[str]) -> Set[str]:
    """合并集合"""
    return (existing or set()).union(new or set())

def dict_merge_reducer(existing: Dict, new: Dict) -> Dict:
    """深度合并字典"""
    result = (existing or {}).copy()
    result.update(new or {})
    return result

# 展示不同 reducer 的效果
class MultiReducerState(TypedDict):
    count: Annotated[int, operator_add]  # 累加数字
    max_score: Annotated[float, max_reducer]  # 保留最大值
    tags: Annotated[Set[str], set_union_reducer]  # 合并集合
    metadata: Annotated[Dict, dict_merge_reducer]  # 合并字典

print("不同 Reducer 示例：")

# 模拟节点更新
def simulate_node_updates():
    # 节点 A 的更新
    update_a = {
        "count": 5,
        "max_score": 0.7,
        "tags": {"urgent", "analysis"},
        "metadata": {"source": "node_a", "timestamp": 100}
    }

    # 节点 B 的更新
    update_b = {
        "count": 3,
        "max_score": 0.9,
        "tags": {"processing", "analysis"},
        "metadata": {"priority": "high", "timestamp": 200}
    }

    print("节点 A 更新:", update_a)
    print("节点 B 更新:", update_b)
    print()

    # 模拟 reducer 的合并结果
    print("各 reducer 的合并结果：")
    print(f"count (累加): {operator_add(update_a['count'], update_b['count'])}")
    print(f"max_score (最大值): {max_reducer(update_a['max_score'], update_b['max_score'])}")
    print(f"tags (集合合并): {set_union_reducer(update_a['tags'], update_b['tags'])}")
    print(f"metadata (字典合并): {dict_merge_reducer(update_a['metadata'], update_b['metadata'])}")

simulate_node_updates()

print("\n" + "=" * 50)
print("并行更新冲突处理示例")
print("=" * 50)

"""
并行更新冲突场景：
想象两个节点同时更新同一个状态字段时会发生什么？

情况1：没有 Annotated
- 节点A返回: {'count': 10}
- 节点B返回: {'count': 20}
- 结果: count = 20 (B覆盖了A，丢失了A的更新)

情况2：使用 Annotated[int, add]
- 节点A返回: {'count': 10}
- 节点B返回: {'count': 20}
- 结果: count = 30 (10 + 20，两个更新都保留)

这就是 Annotated 在 LangGraph 中的关键作用：
解决并行更新的冲突，确保信息不丢失！
"""

# 实际演示冲突处理
print("演示并行更新冲突处理：")
print("情况1 - 没有 Annotated (覆盖): 10 -> 20 = 20")
print("情况2 - 使用 Annotated (累加): 10 + 20 = 30")

# 5. add_messages reducer 的作用
print("\n" + "=" * 50)
print("5. add_messages reducer 详细演示")
print("=" * 50)

# 不使用 add_messages 的普通状态
class NormalState(TypedDict):
    messages: List[BaseMessage]

# 使用 add_messages 的状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 模拟状态更新
print("普通状态更新（会覆盖）:")
normal_state = NormalState(messages=[HumanMessage(content="第一条消息")])
print(f"初始状态: {[msg.content for msg in normal_state['messages']]}")

# 普通更新会覆盖
normal_state = NormalState(messages=[HumanMessage(content="第二条消息")])
print(f"更新后: {[msg.content for msg in normal_state['messages']]}")

print("\nAgentState 更新（会累加）:")
# 在实际 LangGraph 中，add_messages 会自动累加消息
# 这里模拟其行为
def simulate_add_messages(existing_msgs, new_msgs):
    """模拟 add_messages 的行为"""
    return existing_msgs + list(new_msgs)

existing_messages = [HumanMessage(content="第一条消息")]
new_messages = [AIMessage(content="第二条消息")]
result = simulate_add_messages(existing_messages, new_messages)

print(f"累加结果: {[msg.content for msg in result]}")

# 6. 在 LangGraph 中的实际应用
print("\n" + "=" * 50)
print("6. LangGraph 中的实际应用")
print("=" * 50)

# 设置 LLM
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model = os.getenv("OPENROUTER_MODEL")

if api_key and base_url and model:
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

    # 定义节点函数
    def chatbot_node(state: AgentState) -> AgentState:
        """聊天机器人节点 - 处理用户消息并生成回复"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}  # 返回新消息，add_messages 会自动累加

    def logger_node(state: AgentState) -> AgentState:
        """日志节点 - 记录当前对话状态"""
        print(f"\n当前对话包含 {len(state['messages'])} 条消息:")
        for i, msg in enumerate(state["messages"]):
            print(f"{i+1}. {type(msg).__name__}: {msg.content}")
        return {}  # 不添加新消息

    # 构建图
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("logger", logger_node)

    # 添加边
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", "logger")
    workflow.add_edge("logger", END)

    # 编译图
    app = workflow.compile()

    # 运行示例
    print("\n运行 LangGraph 示例:")
    initial_state = AgentState(messages=[HumanMessage(content="请用中文简单介绍一下 LangGraph")])

    try:
        final_state = app.invoke(initial_state)
        print(f"\n最终状态包含 {len(final_state['messages'])} 条消息")
    except Exception as e:
        print(f"运行出错（可能是API配置问题）: {e}")
else:
    print("未配置 API 信息，跳过 LangGraph 示例")

# 7. 类型注解的好处
print("\n" + "=" * 50)
print("7. 类型注解的好处")
print("=" * 50)

def process_agent_state(state: AgentState) -> None:
    """演示类型提示的好处"""
    # IDE 可以提供自动补全
    messages = state["messages"]  # IDE 知道这是 Sequence[BaseMessage]

    # 类型检查可以发现错误
    # state["invalid_key"]  # 这会被类型检查器标记为错误

    print(f"状态包含 {len(messages)} 条消息")
    for msg in messages:
        # IDE 知道 msg 是 BaseMessage，可以提供 content 属性的补全
        print(f"- {msg.content[:50]}...")

# 创建测试状态
test_state = AgentState(messages=[
    HumanMessage(content="这是一条测试消息"),
    AIMessage(content="这是AI的回复消息")
])

process_agent_state(test_state)

print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print("""
AgentState 类的核心组成：

1. TypedDict: 提供类型安全的字典结构
2. Annotated: 为类型添加元数据，指定状态更新的处理方式
   - 这是 LangGraph 状态管理的核心机制
   - 解决并行更新冲突，防止数据丢失
   - 支持自定义 reducer 函数，实现复杂的状态合并逻辑
3. Sequence: 比 List 更通用的有序集合类型
4. BaseMessage: LangChain 中所有消息的基类
5. add_messages: LangGraph 的消息累加器，确保消息历史得以保留

Annotated 的关键作用：
- 没有 Annotated：状态更新会相互覆盖
- 有 Annotated：状态更新会智能合并
- 这对于多节点、并行执行的复杂图结构至关重要

这种设计让 LangGraph 能够：
- 自动管理对话历史
- 提供类型安全
- 支持复杂的多轮对话
- 处理并行节点的状态冲突
- 便于调试和维护
""")