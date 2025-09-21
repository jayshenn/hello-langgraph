## 类型注解（Type Annotations）

### 字典（Dictionary）

普通字典：

```python
movie = {"name": "Avengers Endgame", "year": 2019}
```

* 允许基于唯一键进行高效数据检索
* 灵活、易于实现
* 但在确保数据结构一致性方面有挑战，尤其是大型项目
* 不会检查数据是否为正确的类型或结构

---

### 类型化字典（Typed Dictionary）

```python
from typing import TypedDict

class Movie(TypedDict): 
    name: str
    year: int

movie = Movie(name="Avengers Endgame", year=2019)
```

* **类型安全（Type Safety）**：显式定义数据结构，减少运行时错误
* **可读性增强**：便于调试，使代码更易于理解

---

### 联合类型（Union）

```python
from typing import Union

def square(x: Union[int, float]) -> float:
    return x * x

x = 5        # ✅ 整数没问题
x = 1.234    # ✅ 浮点数也可以
x = "Hello"  # ❌ 字符串会报错
```

* Union 允许你声明一个值可以是多种类型
* 灵活且易于编码
* 提供类型安全性，可帮助捕获错误用法

---

### 可选类型（Optional）

```python
from typing import Optional

def nice_message(name: Optional[str]) -> None:
    if name is None:
        print("Hey random person!") 
    else:
        print(f"Hi there, {name}!")
```

* 在此例中，`name` 可以是 **字符串** 或 **None**
* 它不能是其他类型

---

### 任意类型（Any）

```python
from typing import Any

def print_value(x: Any):
    print(x)

print_value("I pretend to be Batman in the shower sometimes")
```

* **Any**：任何东西都可以！

---

### Lambda 函数

```python
square = lambda x: x * x
print(square(10))

nums = [1, 2, 3, 4]
squares = list(map(lambda x: x * x, nums))
```

* Lambda 是书写小函数的快捷方式

---

## LangGraph 核心元素（Elements）

### 状态（State）

* **State** 是一个共享数据结构，保存应用当前的信息或上下文。
* 简单理解：它就像应用的“记忆”，记录变量和数据，供节点（nodes）读取和修改。

**类比**：会议室里的白板——参与者（节点）在白板（状态）上写和读信息，以保持同步。

---

### 节点（Nodes）

* 节点是单独的函数或操作，执行特定任务。
* 每个节点接收输入（通常是当前状态）、处理后，输出更新后的状态。

**类比**：流水线上的工位——每个工位只做一件事：装零件、喷漆、质检……

---

### 图（Graph）

* Graph 是整个任务的结构，展示了节点之间的连接与执行顺序。
* 它是工作流的可视化表示，包含顺序与条件路径。

**类比**：公路地图——展示了城市之间的不同路线，路口就是选择点。

---

### 边（Edges）

* **Edges** 是节点之间的连接，决定执行流程。
* 它告诉我们当前节点完成后，下一个执行哪个节点。

**类比**：铁路轨道——每条轨道（边）连接车站（节点）。

---

### 条件边（Conditional Edges）

* 特殊的边，根据特定条件或逻辑决定执行下一个节点。

**类比**：交通信号灯——绿灯直行，红灯停止，黄灯减速，不同颜色决定下一步动作。

---

### START 节点

* START 是虚拟入口点，标记工作流的开始。
* 自身不执行任何操作，只是起点。

**类比**：比赛起跑线。

---

### END 节点

* END 节点表示工作流的结束。
* 到达该节点，图的执行停止。

**类比**：比赛终点线。

---

### 工具（Tools）

* 工具是节点可调用的函数或实用工具，如调用 API。
* 它增强节点功能。
* 节点是图结构的一部分，工具只是节点内部使用的功能。

**类比**：工具箱里的工具——锤子钉钉子，螺丝刀拧螺丝，各司其职。

---

### 工具节点（ToolNode）

* ToolNode 是一种特殊节点，其主要任务是运行工具。
* 它把工具的输出写回状态，供其他节点使用。

**类比**：工人操作机器，工人（ToolNode）控制机器（Tool），并把结果交回流水线。

---

### 状态图（StateGraph）

* **StateGraph** 是 LangGraph 中用于构建和编译图结构的类。
* 它管理节点、边和整体状态，确保数据正确流动。

**类比**：建筑蓝图——蓝图规定了建筑的设计与连接方式，StateGraph 定义了工作流的结构和流转。

---

### 可运行单元（Runnable）

* Runnable 是 LangGraph 中的标准化可执行组件，完成特定任务。
* 它是构建模块，可组合成复杂系统。

**类比**：乐高积木——每个砖块（Runnable）都能拼装出复杂的结构。

---

### 消息（Messages）

* **Human Message**：用户输入
* **System Message**：系统指令或上下文
* **Function Message**：函数调用的结果
* **AI Message**：AI 模型生成的响应
* **Tool Message**：类似 Function Message，但专用于工具调用

---

## Graph I

### Hello World Graph

目标：

1. 理解并定义 AgentState 结构
2. 创建简单的节点函数来处理和更新状态
3. 设置一个基本的 LangGraph 结构
4. 编译并调用一个 LangGraph
5. 理解数据如何在单节点中流动

**练习**：

创建一个“个性化赞美代理”。

输入：`{"name": "Bob"}`
输出：`"Bob, you're doing an amazing job learning LangGraph!"`

提示：你需要**拼接（concatenate）状态**，而不是覆盖（replace）。

---

## Graph II

### 多输入图（Multiple Inputs Graph）

目标：

1. 定义更复杂的 AgentState
2. 创建能处理列表数据的节点
3. 设置一个能处理并输出计算结果的 LangGraph
4. 使用结构化输入调用图并获取输出

主要目标：学习如何处理多个输入

**练习**：

创建一个图，输入包括整数列表、姓名和操作符。如果操作符是 `"+"`，就把列表元素相加；如果是 `"*"`，就相乘。

输入：

```json
{"name": "Jack Sparrow", "values": [1,2,3,4], "operation": "*"}
```

输出：

```
"Hi Jack Sparrow, your answer is: 24"
```

提示：在节点中需要使用 **if 判断**。

---

## Graph III

### 顺序图（Sequential Graph）

目标：

1. 创建多个节点，按顺序处理并更新状态
2. 将节点连接成图
3. 调用图，查看状态如何逐步变化

主要目标：创建并处理多个节点

**练习**：

要求：

1. 输入用户的姓名、年龄和技能列表。
2. 状态依次通过三个节点：

   * 节点 1：个性化欢迎信息
   * 节点 2：描述用户年龄
   * 节点 3：列出用户技能
3. 最终结果字段应为组合消息，例如：

输出：

```
“Linda, welcome to the system! You are 31 years old! You have skills in: Python, Machine Learning, and LangGraph”
```

提示：你需要调用 `add_edge` 两次。

---

## Graph IV

### 条件图（Conditional Graph）

目标：

1. 使用条件逻辑将数据流路由到不同节点
2. 使用 START 和 END 明确管理入口和出口
3. 创建多个节点执行不同操作（加法、减法）
4. 创建路由节点（router node）负责决策

主要目标：学习如何使用 `add_conditional_edges()`

**练习**：

构建指定的图，需要用到 **2 条条件边**。

输入：

```python
initial_state = AgentState(
    number1=10, operation="-", number2=5, number3=7, number4=2,
    operation2="+", finalNumber=0, finalNumber2=0
)
```

---

## Graph V

### 循环图（Looping Graph）

目标：

1. 实现循环逻辑，让数据流返回节点
2. 创建单个条件边用于决策和控制流程

主要目标：编写循环逻辑

**练习**：

实现一个 **自动大小猜数字游戏（Higher or Lower Game）**。

* 范围：1 \~ 20
* 最多 7 次猜测
* 如果猜对则结束，否则继续循环，直到 7 次用完
* 每次猜测后，提示节点输出“更大/更小”，并根据提示调整范围

输入：

```json
{"player_name": "Student", "guesses": [], "attempts": 0, "lower_bound": 1, "upper_bound": 20}
```

提示：每次猜测后需要根据提示调整边界。

---

## Agent I

### 简单机器人（Simple Bot）

目标：

1. 定义包含 HumanMessage 列表的状态结构
2. 使用 LangChain 的 ChatOpenAI 初始化 GPT-4o 模型
3. 发送和处理不同类型的消息
4. 构建并编译 Agent 图

主要目标：学习如何将 LLM 集成到 LangGraph 中

---

## Agent II

### 聊天机器人（Chatbot）

目标：

1. 使用不同类型的消息（HumanMessage, AIMessage）
2. 使用这两种消息维护完整对话历史
3. 使用 GPT-4o 模型（ChatOpenAI）
4. 创建一个更复杂的对话循环

主要目标：为 Agent 创建记忆机制

---

## Agent III

### ReAct Agent（推理 + 行动代理）

目标：

1. 学习如何在 LangGraph 中创建工具（Tools）
2. 构建一个 ReAct 图
3. 使用不同类型的消息（如 ToolMessages）
4. 测试图的鲁棒性

主要目标：构建一个稳健的 ReAct Agent

---

## Agent IV

### 起草助手（DRAFTER）

**任务（Boss’s Orders）：**

我们的公司效率低下！起草文档、邮件花了太多时间。

你的任务：

* 构建一个 **AI 文档起草系统**
* 支持 **人机协作（Human-AI Collaboration）**：人类可持续提供反馈，AI 根据反馈修改草稿
* 当人类确认满意时，AI 停止
* 系统需快速且能保存草稿

---

## Agent V

### RAG（检索增强生成）
