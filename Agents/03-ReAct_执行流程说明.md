# ReAct Agent 执行流程详解

## 什么是 ReAct？

ReAct 是 "Reasoning and Acting"（推理和行动）的缩写，这是一种让 AI Agent 能够：
1. **推理（Reasoning）**：思考该做什么
2. **行动（Acting）**：执行具体的工具或动作
3. **观察（Observation）**：获取工具执行的结果
4. **再推理**：基于结果继续思考下一步

## 核心组件说明

### 1. 状态（State）
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```
- 存储整个对话历史
- 包含用户输入、AI 响应、工具调用和工具结果
- `add_messages` 自动处理消息的累加

### 2. 工具（Tools）
```python
@tool
def add(a: int, b: int):
    """加法函数"""
    return a + b
```
- 使用 `@tool` 装饰器定义
- LLM 可以选择调用这些工具来完成任务
- 工具的 docstring 帮助 LLM 理解工具用途

### 3. 节点（Nodes）

#### Agent 节点
- 调用 LLM 进行推理
- 决定是回答用户还是调用工具
- 是整个流程的"大脑"

#### Tools 节点
- 执行工具调用
- 返回执行结果
- 使用 `ToolNode` 自动处理

### 4. 边（Edges）

#### 条件边
```python
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
```
- 根据条件决定下一步
- 如果需要工具，去 tools 节点
- 如果完成任务，结束执行

#### 普通边
```python
graph.add_edge("tools", "agent")
```
- 工具执行完总是返回 agent 节点
- 形成循环：思考 → 执行 → 再思考

## 执行流程示例

以 "计算 40+12 然后乘以 6" 为例：

```
1. 用户输入 → Agent 节点
   消息: "Add 40 + 12 and then multiply the result by 6"

2. Agent 推理
   LLM 思考: "需要先加法，再乘法"
   决定: 调用 add(40, 12)

3. 条件判断
   should_continue() → "continue"（有工具调用）

4. Tools 节点执行
   执行: add(40, 12)
   结果: 52

5. 返回 Agent 节点
   Agent 看到结果 52
   决定: 调用 multiply(52, 6)

6. 条件判断
   should_continue() → "continue"（有工具调用）

7. Tools 节点执行
   执行: multiply(52, 6)
   结果: 312

8. 返回 Agent 节点
   Agent 看到结果 312
   生成最终答案: "40 + 12 = 52, 52 × 6 = 312"

9. 条件判断
   should_continue() → "end"（没有工具调用）

10. 结束执行
```

## 消息类型

### 1. HumanMessage
- 用户的输入消息

### 2. AIMessage
- LLM 的响应
- 可能包含工具调用请求

### 3. ToolMessage
- 工具执行的结果
- 包含 tool_call_id 关联到对应的调用

### 4. SystemMessage
- 系统指令
- 定义 AI 的角色和行为

## 关键概念

### 1. 工具绑定
```python
llm = llm.bind_tools(tools)
```
- 让 LLM 知道可用的工具
- LLM 会自动选择合适的工具

### 2. 流式输出
```python
app.stream(inputs, stream_mode="values")
```
- 实时查看执行过程
- 每个节点执行后返回完整状态

### 3. 状态管理
- 状态在节点间传递
- 每个节点返回状态更新
- 自动合并到全局状态

## 初学者常见问题

### Q1: 为什么需要 ReAct 模式？
**A**: 单纯的 LLM 无法执行具体操作（如数学计算），ReAct 让 LLM 能够：
- 分解复杂任务
- 调用工具完成具体操作
- 基于结果继续推理

### Q2: Agent 如何知道何时停止？
**A**: 通过 `should_continue` 函数判断：
- 如果最后的消息包含工具调用 → 继续
- 如果最后的消息是纯文本回答 → 结束

### Q3: 为什么要用图结构？
**A**: 图结构让执行流程：
- 更加清晰可控
- 便于调试和观察
- 支持复杂的分支和循环逻辑

### Q4: 工具调用失败怎么办？
**A**: ToolNode 会自动处理错误：
- 捕获工具执行异常
- 返回错误信息给 Agent
- Agent 可以基于错误重新推理

## 练习建议

1. **添加新工具**：尝试添加除法、幂运算等工具
2. **修改提示词**：改变 SystemMessage 看看 Agent 行为的变化
3. **复杂任务**：给 Agent 更复杂的多步骤任务
4. **错误处理**：故意传入错误参数，观察 Agent 如何处理
5. **自定义路由**：修改 `should_continue` 添加更多分支逻辑

## 调试技巧

1. **打印中间状态**
```python
def model_call(state: AgentState) -> AgentState:
    print(f"当前消息数: {len(state['messages'])}")
    # ... 其他代码
```

2. **查看工具调用**
```python
if last_message.tool_calls:
    print(f"工具调用: {last_message.tool_calls}")
```

3. **可视化图结构**
```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

## 进阶学习

1. **添加记忆功能**：让 Agent 记住之前的对话
2. **并行工具调用**：同时执行多个工具
3. **工具链**：一个工具的输出作为另一个工具的输入
4. **错误恢复**：实现更智能的错误处理和重试机制
5. **自定义工具节点**：不使用预构建的 ToolNode，自己实现工具执行逻辑