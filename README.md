# 🚀 LangGraph 学习项目

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

一个全面的 LangGraph 学习项目，提供从基础概念到高级应用的完整学习路径。通过系统化的教程、丰富的示例代码和实际项目案例，帮助开发者掌握 LangGraph 框架，构建有状态的 AI Agent 和工作流应用。

## ✨ 项目特性

- 📚 **系统化学习路径** - 从 Python 基础到 LangGraph 高级特性的完整体系
- 💻 **丰富示例代码** - 28+ Python 文件，10+ Jupyter Notebooks
- 📖 **详细文档指南** - 48+ 文档文件，涵盖理论与实践
- 🛠️ **实际项目案例** - 聊天机器人、RAG 系统、多 Agent 架构等
- 🎯 **渐进式难度** - 从 Hello World 到复杂工作流的学习曲线
- 🔧 **开箱即用** - 完整的开发环境配置和最佳实践

## 🚀 快速开始

### 前置要求

- Python 3.8+
- 基础的 Python 编程知识
- OpenRouter API 密钥（用于访问 LLM）

### 环境设置

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/hello-langgraph.git
   cd hello-langgraph
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，添加你的 OpenRouter API 密钥
   ```

### 第一个示例

运行你的第一个 LangGraph 应用：

```bash
# 简单的 Agent 机器人
python Agents/01-Agent_Bot.py

# 或者使用 Jupyter Notebook
jupyter notebook Graphs/01-Hello_Word.ipynb
```

## 📁 项目结构

```
hello-langgraph/
├── 📂 Foundations/          # 系统学习资料
│   ├── 01-Python前置/      # Python 基础知识
│   ├── 02-图论与状态机/    # 图论和状态机原理
│   ├── 03-LangGraph基础/   # LangGraph 核心概念
│   ├── 04-进阶特性/        # 条件路由、循环、记忆
│   ├── 05-工具与集成/      # 工具调用、LLM 集成
│   ├── 06-开发实践/        # 调试、测试、性能优化
│   ├── 07-部署与运维/      # Docker、K8s 部署
│   ├── 08-项目案例/        # 实际应用示例
│   └── 09-速查手册/        # 快速参考文档
├── 📂 Agents/              # Agent 实现示例
│   ├── 01-Agent_Bot.py     # 简单 LLM 集成
│   └── 02-Memory_Agent.py  # 带记忆的对话 Agent
├── 📂 Graphs/              # 图结构示例
│   ├── 01-Hello_Word.ipynb      # 基础图结构
│   ├── 02-Multiple_Inputs.ipynb # 多输入处理
│   ├── 03-Sequential_Agent.ipynb# 顺序执行图
│   ├── 04-Conditional_Agent.ipynb# 条件分支图
│   └── 05-Looping.ipynb         # 循环控制图
├── 📂 Exercises/           # 练习题集合
│   └── Exercise_Graph*.ipynb    # 不同难度练习
├── 📄 LangGraph学习笔记.md # 核心概念总结
├── 📄 CLAUDE.md            # Claude Code 指南
└── 📄 requirements.txt     # 项目依赖
```

## 🎯 学习路径

### 🌱 零基础路径（4-5 周）
**适合：Python 初学者、编程新手**

1. **第1周：Python 前置知识**
   - 类型系统和 TypedDict
   - 异步编程基础
   - 完成 `Foundations/01-Python前置/` 练习

2. **第2周：图论基础**
   - 图的基本概念
   - 状态机原理
   - 工作流模式理解

3. **第3周：LangGraph 入门**
   - 运行 `Graphs/01-Hello_Word.ipynb`
   - 学习状态、节点、图的概念
   - 完成基础练习

4. **第4周：实践应用**
   - 构建简单聊天机器人
   - 理解 Agent 工作原理

### 🚀 进阶路径（2-3 周）
**适合：有 Python 基础的开发者**

1. **第1周：核心概念**
   - 快速浏览 `Foundations/03-LangGraph基础/`
   - 完成所有 `Graphs/` 示例

2. **第2周：高级特性**
   - 条件路由和循环控制
   - 记忆和持久化
   - 工具集成

3. **第3周：项目实战**
   - RAG 系统构建
   - 多 Agent 架构

### ⚡ 专家路径（1 周）
**适合：有 AI/ML 经验的开发者**

1. 直接学习 `Foundations/04-进阶特性/` 及后续章节
2. 运行 `Foundations/08-项目案例/` 中的复杂示例
3. 参考 `Foundations/07-部署与运维/` 进行生产部署

## 🔧 配置说明

### OpenRouter API 配置

项目使用 OpenRouter 作为统一的 LLM API 入口，支持多种模型：

```bash
# .env 文件配置
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o
```

支持的模型包括：
- OpenAI GPT-4/GPT-3.5
- Anthropic Claude
- Google Gemini
- 其他开源模型

## 💡 核心概念示例

### 基本状态定义
```python
from typing import TypedDict, List
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: List[HumanMessage]
    user_name: str
    step_count: int
```

### 简单图构建
```python
from langgraph.graph import StateGraph, START, END

def process_message(state: AgentState) -> AgentState:
    # 处理逻辑
    return {"messages": updated_messages}

workflow = StateGraph(AgentState)
workflow.add_node("process", process_message)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

app = workflow.compile()
```

## 🌟 应用案例

### 1. 智能聊天机器人
- 支持上下文记忆
- 多轮对话管理
- 个性化响应

### 2. RAG 知识问答系统
- 文档检索增强
- 向量数据库集成
- 智能答案生成

### 3. 多 Agent 协作系统
- 任务分解与分配
- Agent 间通信
- 结果汇总整合

### 4. 工作流自动化
- 条件分支逻辑
- 循环控制结构
- 人机交互节点

## 🤝 贡献指南

我们欢迎社区贡献！你可以通过以下方式参与：

- 🐛 **报告问题** - 发现 bug 或文档错误
- 💡 **功能建议** - 提出新的示例或改进想法
- 📝 **文档改进** - 完善教程和说明文档
- 🔧 **代码贡献** - 添加新的示例或优化现有代码

### 贡献步骤

1. Fork 项目仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 📞 支持与社区

- 📧 **问题反馈**：提交 GitHub Issues
- 💬 **讨论交流**：GitHub Discussions
- 📖 **文档**：查看 `Foundations/` 目录
- 🔍 **快速查找**：使用 `Foundations/09-速查手册/`

## 📄 许可证

本项目基于 MIT 许可证开源。详情请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下项目和社区的支持：

- [LangGraph](https://github.com/langchain-ai/langgraph) - 强大的状态图框架
- [LangChain](https://github.com/langchain-ai/langchain) - AI 应用开发生态
- [OpenRouter](https://openrouter.ai/) - 统一 LLM API 服务

---

⭐ 如果这个项目对你有帮助，请给我们一个 Star！

📚 开始你的 LangGraph 学习之旅吧！
