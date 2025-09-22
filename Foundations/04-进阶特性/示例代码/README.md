# 进阶特性示例代码

本目录包含 LangGraph 进阶特性的完整示例代码，每个文件都可以独立运行，演示特定的进阶功能。

## 📁 文件结构

```
示例代码/
├── README.md                    # 本文件
├── conditional_routing.py       # 条件路由示例
├── loop_patterns.py            # 循环结构示例
├── memory_example.py           # Memory机制示例
├── human_in_loop.py            # Human-in-the-Loop示例
└── persistence_demo.py         # Persistence持久化示例
```

## 🚀 运行示例

### 前置条件

确保已安装必要的依赖：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装 LangGraph
pip install langgraph

# 运行示例代码
python 示例代码/conditional_routing.py
```

## 📝 示例说明

### 1. 条件路由示例 (`conditional_routing.py`)

**功能演示**：
- 智能意图识别
- 多层条件路由
- 上下文感知决策
- 路由历史追踪

**核心特性**：
```python
# 智能路由函数
def intelligent_router(state) -> str:
    intent = state.get("intent")
    confidence = state.get("confidence")
    context = state.get("context")

    # 基于多种因素的路由决策
    if context.get("urgency_keywords"):
        return "priority_handler"

    return route_map.get(intent, "default")
```

**运行效果**：
- 自动识别用户意图（问候、询问、请求、投诉等）
- 根据紧急程度和上下文调整路由策略
- 提供详细的路由决策过程

### 2. 循环结构示例 (`loop_patterns.py`)

**功能演示**：
- 智能重试系统（指数退避）
- 数据处理管道循环
- 质量检查和改进循环
- 性能监控和故障检测

**核心特性**：
```python
# 智能重试决策
def intelligent_continue_checker(state) -> str:
    if consecutive_failures >= 3:
        return "too_many_failures"
    if success_rate < 0.2:
        return "low_success_rate"
    return "continue"
```

**运行效果**：
- 自适应重试策略，避免无限循环
- 实时性能监控和统计
- 智能失败处理和恢复机制

### 3. Memory机制示例 (`memory_example.py`)

**功能演示**：
- 智能对话记忆管理
- 用户画像自动提取
- 个性化响应生成
- 长期记忆学习

**核心特性**：
```python
# 智能记忆管理
def memory_manager(state) -> State:
    if len(messages) > 10:
        summary = generate_conversation_summary(messages[:-5])
        state["conversation_summary"] = summary
        state["messages"] = messages[-5:]  # 保留最近5条
    return state
```

**运行效果**：
- 自动提取用户个人信息（姓名、地点、偏好）
- 基于记忆生成个性化回复
- 跨会话的学习和模式识别

### 4. Human-in-the-Loop示例 (`human_in_loop.py`)

**功能演示**：
- 智能审批工作流
- 内容审核系统
- 风险评估和人工干预
- 审批链记录

**核心特性**：
```python
# 风险评估和路由
def approval_router(state) -> str:
    risk_score = state.get("ai_analysis", {}).get("risk_score", 0)

    if risk_score >= 0.4:
        return "human_approval"  # 需要人工审批
    else:
        return "auto_approve"    # 自动批准
```

**运行效果**：
- 基于风险评估的智能审批路由
- 模拟真实的人工决策过程
- 完整的审批历史和追踪

### 5. Persistence持久化示例 (`persistence_demo.py`)

**功能演示**：
- 容错系统和自动恢复
- 状态备份和恢复机制
- 时间旅行调试器
- 检查点管理

**核心特性**：
```python
# 自动恢复机制
def recovery_handler(state) -> State:
    if last_execution.get("status") == "failed":
        last_success_step = task_progress.get("last_success_step", 0)
        task_progress["current_step"] = last_success_step
        return state
```

**运行效果**：
- 自动检测故障并恢复到安全检查点
- 完整的执行历史和状态快照
- 时间旅行功能，支持回到任意检查点

## 🎯 学习建议

### 循序渐进学习

1. **先运行基础示例**：从 `conditional_routing.py` 开始
2. **理解核心概念**：仔细阅读代码注释和输出
3. **修改参数实验**：尝试修改配置参数观察效果
4. **组合使用**：将多个特性组合到实际项目中

### 实践练习

每个示例都包含了练习建议：

```python
def practice_exercise():
    """
    练习建议：
    1. 修改路由条件，观察不同的路由效果
    2. 调整重试策略，测试不同失败场景
    3. 扩展记忆功能，添加新的用户画像字段
    4. 设计自己的审批流程
    5. 实现自定义的持久化策略
    """
    pass
```

### 调试技巧

**查看详细输出**：
```bash
# 运行时查看详细日志
python conditional_routing.py 2>&1 | tee output.log
```

**修改日志级别**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔗 相关文档

- [01-条件路由详解.md](../01-条件路由详解.md)
- [02-循环结构.md](../02-循环结构.md)
- [03-子图与多Agent.md](../03-子图与多Agent.md)
- [04-Memory机制.md](../04-Memory机制.md)
- [05-Human-in-the-Loop.md](../05-Human-in-the-Loop.md)
- [06-Persistence.md](../06-Persistence.md)

## ❓ 常见问题

### Q: 示例代码运行出错怎么办？

A: 检查以下几点：
1. 确保已激活虚拟环境
2. 安装了正确版本的 langgraph
3. Python 版本 >= 3.8

### Q: 如何修改示例适用于实际项目？

A:
1. 替换模拟的数据处理逻辑
2. 集成实际的数据库和API
3. 添加错误处理和日志记录
4. 根据业务需求调整状态结构

### Q: 示例中的 interrupt() 函数为什么没有实际暂停？

A: 示例代码为了演示方便，使用了模拟的人工决策。在实际应用中，需要：
1. 导入 `from langgraph.prebuilt import interrupt`
2. 配置适当的检查点保存器
3. 实现真实的人工交互界面

## 📧 反馈和贡献

如果您在使用示例代码时遇到问题或有改进建议，欢迎：
1. 查看项目文档中的相关章节
2. 在 Issues 中提出问题
3. 提交 Pull Request 改进代码

---

**快乐学习 LangGraph 进阶特性！** 🎉