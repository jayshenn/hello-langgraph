# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 LangGraph 学习项目，包含系统化的教程练习和示例代码。项目通过 Jupyter Notebook 提供交互式学习体验，从基础图形构建逐步进阶到复杂的循环和条件逻辑实现。

## 常用开发命令

### 环境管理
```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动 Jupyter Lab
source .venv/bin/activate && jupyter lab

# 安装新依赖并更新 requirements.txt
source .venv/bin/activate && pip install package_name && pip freeze > requirements.txt

# 检查已安装的包版本
source .venv/bin/activate && pip list
```

### 运行特定 Notebook
```bash
# 在命令行中运行 Notebook（用于测试）
source .venv/bin/activate && jupyter nbconvert --to notebook --execute Exercises/Exercise_Graph1.ipynb
```

## 项目结构与学习路径

### 核心目录
- `Exercises/`: LangGraph 练习题（Exercise_Graph1.ipynb ~ Exercise_Graph5.ipynb）
- `Graphs/`: 示例实现（01-Hello_Word.ipynb ~ 05-Looping.ipynb）
- `LangGraph学习笔记.md`: 详细的理论参考文档
- `data/`, `models/`: 数据和模型文件存储（目前为空）

### 递进式学习体系
1. **Graph I (Hello World)**: 单节点图，基础状态管理
2. **Graph II (Multiple Inputs)**: 多输入处理，状态累积
3. **Graph III (Sequential)**: 序列化节点链，数据流传递
4. **Graph IV (Conditional)**: 条件分支，决策逻辑
5. **Graph V (Looping)**: 循环结构，复杂游戏逻辑

## LangGraph 架构模式

### 状态管理模式
```python
class AgentState(TypedDict):
    # 使用 TypedDict 确保类型安全
    field_name: str
    numeric_field: int
    list_field: List[Any]
```

### 图构建模式
```python
# 标准图构建流程
graph = StateGraph(AgentState)
graph.add_node("node_name", node_function)
graph.set_entry_point("start_node")  # 或使用 START
graph.set_finish_point("end_node")   # 或使用 END
app = graph.compile()
```

### 节点函数模式
- 必须接收 `state: AgentState` 参数
- 必须返回 `AgentState` 对象
- 通过修改状态字段实现数据传递
- 支持状态字段的覆盖或累积更新

### 条件逻辑模式
```python
# 条件边的实现
graph.add_conditional_edges(
    "source_node",
    condition_function,  # 返回字符串键
    {
        "condition_result": "target_node",
        "other_result": END
    }
)
```

## 核心依赖

- `langgraph`: 图形工作流框架
- `langchain`, `langchain_openai`: LLM 集成（为 Agent 系列准备）
- `chromadb`, `langchain_chroma`: 向量数据库（RAG 功能）
- `typing`: 类型注解支持