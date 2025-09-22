# LangGraph API 速查表

> 🚀 **快速参考**：常用 LangGraph API 的签名、参数和基础用法

## 📊 目录

- [图构建 APIs](#图构建-apis)
- [状态管理 APIs](#状态管理-apis)
- [执行 APIs](#执行-apis)
- [工具与集成 APIs](#工具与集成-apis)
- [配置 APIs](#配置-apis)
- [流式处理 APIs](#流式处理-apis)
- [平台相关 APIs](#平台相关-apis)

---

## 🏗️ 图构建 APIs

### StateGraph

创建状态图的核心类。

```python
from langgraph.graph import StateGraph

# 签名
StateGraph(schema: Type[TypedDict])

# 基础用法
class MyState(TypedDict):
    message: str

graph = StateGraph(MyState)
```

### add_node

向图中添加节点。

```python
# 签名
graph.add_node(name: str, func: Callable[[State], State])

# 基础用法
def my_node(state: MyState) -> MyState:
    return {"message": "处理完成"}

graph.add_node("process", my_node)
```

### add_edge

添加无条件边。

```python
# 签名
graph.add_edge(from_node: str, to_node: str)
# 或
graph.add_edge(from_nodes: List[str], to_node: str)

# 基础用法
graph.add_edge("start", "process")
graph.add_edge(["node1", "node2"], "merge")
```

### add_conditional_edges

添加条件边。

```python
# 签名
graph.add_conditional_edges(
    source: str,
    condition: Callable[[State], str],
    mapping: Dict[str, str]
)

# 基础用法
def router(state: MyState) -> str:
    return "success" if state["message"] else "error"

graph.add_conditional_edges(
    "process",
    router,
    {"success": "success_node", "error": "error_node"}
)
```

### set_entry_point & set_finish_point

设置图的入口和出口。

```python
# 签名
graph.set_entry_point(node_name: str)
graph.set_finish_point(node_name: str)

# 基础用法
graph.set_entry_point("start")
graph.set_finish_point("end")

# 或使用预定义常量
from langgraph.graph import START, END
graph.add_edge(START, "first_node")
graph.add_edge("last_node", END)
```

### compile

编译图为可执行的应用。

```python
# 签名
graph.compile(
    checkpointer: Optional[BaseCheckpointer] = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    debug: bool = False
) -> CompiledGraph

# 基础用法
app = graph.compile()

# 带检查点
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

---

## 🗂️ 状态管理 APIs

### TypedDict

定义状态结构的基础类。

```python
from typing import TypedDict, List, Optional

# 基础状态
class AgentState(TypedDict):
    message: str
    count: int

# 复杂状态
class ComplexState(TypedDict):
    user_input: str
    search_results: List[Dict[str, Any]]
    metadata: Optional[Dict[str, str]]
```

### Annotated & add_messages

用于消息类型状态的特殊注解。

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MessageState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

### RemoveMessage

用于从消息历史中删除特定消息。

```python
from langgraph.graph.message import RemoveMessage

def trim_messages(state: MessageState) -> MessageState:
    # 删除第一条消息
    return {"messages": [RemoveMessage(id=state["messages"][0].id)]}
```

---

## ▶️ 执行 APIs

### invoke

同步执行图。

```python
# 签名
app.invoke(
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]

# 基础用法
result = app.invoke({"message": "Hello"})

# 带配置
config = {"configurable": {"thread_id": "123"}}
result = app.invoke({"message": "Hello"}, config=config)
```

### ainvoke

异步执行图。

```python
# 签名
await app.ainvoke(
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]

# 基础用法
result = await app.ainvoke({"message": "Hello"})
```

### stream

同步流式执行。

```python
# 签名
app.stream(
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    stream_mode: str = "values"
) -> Iterator[Dict[str, Any]]

# 基础用法
for chunk in app.stream({"message": "Hello"}):
    print(chunk)

# 不同流模式
for update in app.stream({"message": "Hello"}, stream_mode="updates"):
    print(update)
```

### astream

异步流式执行。

```python
# 签名
app.astream(
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    stream_mode: str = "values"
) -> AsyncIterator[Dict[str, Any]]

# 基础用法
async for chunk in app.astream({"message": "Hello"}):
    print(chunk)
```

### batch

批量执行。

```python
# 签名
app.batch(
    inputs: List[Dict[str, Any]],
    config: Optional[RunnableConfig] = None
) -> List[Dict[str, Any]]

# 基础用法
inputs = [{"message": "Hello"}, {"message": "World"}]
results = app.batch(inputs)
```

---

## 🛠️ 工具与集成 APIs

### ToolNode

创建工具节点。

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

# 创建工具节点
tools = [search_tool]
tool_node = ToolNode(tools)

# 添加到图
graph.add_node("tools", tool_node)
```

### interrupt

暂停执行以获取人工输入。

```python
from langgraph.prebuilt import interrupt

def human_review_node(state: MyState) -> MyState:
    if needs_review(state):
        human_input = interrupt("请审核内容")
        return {**state, "review": human_input}
    return state
```

### create_react_agent

创建 ReAct 风格的智能体。

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
tools = [search_tool]

agent = create_react_agent(model, tools)
```

---

## ⚙️ 配置 APIs

### RunnableConfig

执行配置对象。

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={
        "thread_id": "user_123",
        "user_id": "user_456"
    },
    tags=["production"],
    metadata={"version": "1.0"}
)
```

### MemorySaver

内存检查点保存器。

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

### SqliteSaver

SQLite 检查点保存器。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)
```

---

## 🌊 流式处理 APIs

### 流模式参数

不同的流式输出模式。

```python
# values: 输出每个节点后的完整状态
for chunk in app.stream(input, stream_mode="values"):
    pass

# updates: 输出每个节点的状态更新
for chunk in app.stream(input, stream_mode="updates"):
    pass

# debug: 输出调试信息
for chunk in app.stream(input, stream_mode="debug"):
    pass

# custom: 输出自定义数据
for chunk in app.stream(input, stream_mode="custom"):
    pass
```

### astream_events

异步事件流。

```python
async for event in app.astream_events(input, version="v1"):
    if event["event"] == "on_llm_stream":
        print(event["data"]["chunk"])
```

---

## 🏢 平台相关 APIs

### RemoteGraph

连接远程 LangGraph 服务。

```python
from langgraph.graph import RemoteGraph

# 连接到远程图
remote_graph = RemoteGraph(
    graph_id="my-graph",
    url="https://api.langgraph.dev"
)

# 像本地图一样使用
result = await remote_graph.ainvoke({"message": "Hello"})
```

### LangGraphClient

LangGraph Platform 客户端。

```python
from langgraph_sdk import LangGraphClient

client = LangGraphClient(
    url="https://your-deployment.langraph.dev",
    api_key="your-api-key"
)

# 创建助手
assistant = await client.assistants.create(
    graph_id="agent",
    config={"configurable": {"model": "gpt-4"}}
)

# 创建线程
thread = await client.threads.create()

# 运行对话
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant["assistant_id"],
    input={"messages": [{"role": "user", "content": "Hello!"}]}
):
    print(chunk)
```

---

## 📚 快速示例

### 最简图

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str

def process(state: State) -> State:
    return {"message": f"处理: {state['message']}"}

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"message": "Hello"})
```

### 带条件的图

```python
def router(state: State) -> str:
    return "good" if "good" in state["message"] else "bad"

graph.add_conditional_edges(
    "process",
    router,
    {"good": "good_path", "bad": "bad_path"}
)
```

### 带工具的智能体

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

model = ChatOpenAI(model="gpt-4")
agent = create_react_agent(model, [search])

result = agent.invoke({"messages": [("user", "搜索Python教程")]})
```

---

## 🔗 相关链接

- 📖 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- 🐛 [错误码对照](./错误码对照.md)
- 💡 [代码片段库](./代码片段.md)
- ❓ [常见问题](./FAQ.md)

---

*API 速查表会持续更新，建议收藏备用！* ⭐