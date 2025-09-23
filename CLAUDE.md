# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 LangGraph 学习项目，专注于教授如何使用 LangGraph 框架构建有状态的 AI Agent 和工作流应用。项目包含从基础概念到高级应用的完整学习路径，以及实际的代码示例和练习。

## 开发环境设置

### 必需的环境变量
项目使用 OPENAI 作为统一的 LLM API 访问入口。在开始开发前，需要配置以下环境变量：

复制 `.env.example` 为 `.env` 并配置：
```bash
OPENAI_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://OPENAI.ai/api/v1
OPENAI_MODEL=openai/gpt-4o
```

### 常用开发命令

```bash
# 创建并激活虚拟环境（项目已有 .venv）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 运行 Python 脚本
python Agents/01-Agent_Bot.py
python Agents/02-Memory_Agent.py

# 启动 Jupyter 环境运行 notebook
jupyter notebook
# 或
jupyter lab

# 运行具体的练习
python Foundations/01-Python前置/练习/typing_exercises.py
```

## 项目架构

### 核心目录结构

- **`Foundations/`**: 系统性学习资料，分为9个模块
  - `01-Python前置`: 类型系统、异步编程等 Python 基础
  - `02-图论与状态机`: 图的基本概念和状态机原理
  - `03-LangGraph基础`: 状态、节点、图的核心概念
  - `04-进阶特性`: 条件路由、循环、记忆等高级功能
  - `05-工具与集成`: 工具调用、LLM 集成、RAG 系统
  - `06-开发实践`: 调试、错误处理、性能优化
  - `07-部署与运维`: Docker、Kubernetes 部署
  - `08-项目案例`: 实际应用示例
  - `09-速查手册`: 快速参考文档

- **`Agents/`**: Agent 实现示例
  - `01-Agent_Bot.py`: 简单的 LLM 集成示例
  - `02-Memory_Agent.py`: 带记忆功能的对话 Agent

- **`Graphs/`**: 不同类型的图实现
  - `01-Hello_Word.ipynb`: 基础图结构
  - `02-Multiple_Inputs.ipynb`: 多输入处理
  - `03-Sequential_Agent.ipynb`: 顺序执行图
  - `04-Conditional_Agent.ipynb`: 条件分支图
  - `05-Looping.ipynb`: 循环控制图

- **`Exercises/`**: 练习题和解答
  - 对应不同难度级别的 Graph 练习

## 核心开发模式

### 1. 状态定义模式
所有 LangGraph 应用都使用 TypedDict 定义状态结构：
```python
from typing import TypedDict, List
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: List[HumanMessage]  # 消息列表
    # 其他状态字段...
```

### 2. 节点函数模式
节点函数接收状态并返回更新的状态：
```python
def node_function(state: AgentState) -> AgentState:
    # 处理逻辑
    return {"messages": updated_messages}
```

### 3. 图构建模式
使用 StateGraph 构建图结构：
```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)
workflow.add_node("node_name", node_function)
workflow.add_edge(START, "node_name")
workflow.add_edge("node_name", END)
app = workflow.compile()
```

### 4. LLM 集成模式
使用 OPENAI 统一访问不同 LLM：
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## 学习路径建议

1. **基础路径**: 先完成 `Foundations/01-Python前置` 的内容
2. **图论理解**: 学习 `Foundations/02-图论与状态机`
3. **实践开始**: 运行 `Graphs/` 目录下的 notebook
4. **Agent 开发**: 学习 `Agents/` 中的实现
5. **项目实战**: 参考 `Foundations/08-项目案例` 构建应用

## 重要约定

- 所有代码注释和文档使用中文
- 状态更新使用字典合并而非覆盖
- 环境变量通过 `.env` 文件管理，不提交到版本控制
- 使用 OPENAI 而非直接调用 OpenAI API，支持多种 LLM 模型