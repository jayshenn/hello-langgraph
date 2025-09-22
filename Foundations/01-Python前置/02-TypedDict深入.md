# TypedDict 深入理解

> 🎯 **学习目标**：深入理解 TypedDict，这是 LangGraph AgentState 的核心技术

## 🤔 为什么 LangGraph 选择 TypedDict？

在 LangGraph 中，你会经常看到这样的代码：

```python
from typing import TypedDict

class AgentState(TypedDict):
    message: str
    count: int
    history: list[str]
```

让我们理解为什么不用普通字典或类，而要用 TypedDict。

## 📊 普通字典 vs TypedDict

### 普通字典的问题

```python
# 普通字典：灵活但危险
user_data = {
    "name": "Alice",
    "age": 25,
    "email": "alice@example.com"
}

# 😰 这些错误在运行时才会发现
user_data["nam"] = "Bob"        # 拼写错误，创建了新键
user_data["age"] = "twenty"     # 类型错误，字符串而不是数字
user_data["phone"]              # KeyError，键不存在

# 😕 从代码中无法知道应该有哪些键
def process_user(data):
    # 不知道 data 应该包含什么
    pass
```

### TypedDict 的优势

```python
from typing import TypedDict

class UserInfo(TypedDict):
    name: str
    age: int
    email: str

# ✅ 类型安全
user: UserInfo = {
    "name": "Alice",
    "age": 25,
    "email": "alice@example.com"
}

# ✅ IDE 会提示可用的键
print(user["name"])  # IDE 自动补全

# ✅ 类型检查器会发现错误
# user["age"] = "twenty"  # mypy 会报错
# user["phone"]           # mypy 会报错
```

## 🎯 LangGraph 中的实际应用

### 基础 AgentState 示例

```python
from typing import TypedDict, List

# 定义聊天机器人的状态
class ChatState(TypedDict):
    message: str                    # 当前消息
    conversation_history: List[str] # 对话历史
    user_id: str                    # 用户ID
    response_count: int             # 响应次数

# 创建状态实例
initial_state: ChatState = {
    "message": "你好",
    "conversation_history": [],
    "user_id": "user_123",
    "response_count": 0
}

# 在节点函数中使用
def greeting_node(state: ChatState) -> ChatState:
    """处理问候的节点"""
    new_message = f"你好，{state['user_id']}！"

    # 更新状态
    updated_state: ChatState = {
        "message": new_message,
        "conversation_history": state["conversation_history"] + [new_message],
        "user_id": state["user_id"],
        "response_count": state["response_count"] + 1
    }

    return updated_state
```

### 复杂状态管理

```python
from typing import TypedDict, Optional, Dict, Any

# 更复杂的 Agent 状态
class AdvancedAgentState(TypedDict):
    # 核心消息
    current_message: str
    conversation_history: List[str]

    # 用户信息
    user_id: str
    user_preferences: Dict[str, Any]

    # Agent 内部状态
    current_tool: Optional[str]        # 当前使用的工具
    tool_results: Dict[str, Any]       # 工具执行结果
    error_count: int                   # 错误次数
    max_iterations: int                # 最大迭代次数

    # 上下文信息
    session_id: str
    timestamp: str

# 使用示例
def web_search_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """网络搜索节点"""
    query = state["current_message"]

    # 模拟网络搜索
    search_results = f"搜索 '{query}' 的结果..."

    # 更新状态
    return {
        **state,  # 保持其他字段不变
        "current_tool": "web_search",
        "tool_results": {
            **state["tool_results"],
            "web_search": search_results
        },
        "conversation_history": state["conversation_history"] + [
            f"搜索: {query}",
            f"结果: {search_results}"
        ]
    }
```

## 🔧 TypedDict 的高级特性

### 1. 可选字段

```python
from typing import TypedDict, NotRequired

# Python 3.11+ 语法
class UserProfile(TypedDict):
    name: str                           # 必需字段
    email: str                          # 必需字段
    phone: NotRequired[str]             # 可选字段
    bio: NotRequired[str]               # 可选字段

# Python 3.8+ 兼容语法
from typing_extensions import TypedDict, NotRequired

class UserProfileCompat(TypedDict):
    name: str
    email: str
    phone: NotRequired[str]
    bio: NotRequired[str]

# 使用
user1: UserProfile = {
    "name": "Alice",
    "email": "alice@example.com"
    # phone 和 bio 可以省略
}

user2: UserProfile = {
    "name": "Bob",
    "email": "bob@example.com",
    "phone": "123-456-7890",
    "bio": "Software developer"
}
```

### 2. 继承和组合

```python
# 基础状态
class BaseState(TypedDict):
    user_id: str
    session_id: str
    timestamp: str

# 继承基础状态
class ChatbotState(BaseState):
    message: str
    conversation_history: List[str]

# 进一步扩展
class RAGState(ChatbotState):
    documents: List[str]
    retrieval_query: str
    relevant_docs: List[str]

# 使用继承的状态
def rag_node(state: RAGState) -> RAGState:
    query = state["message"]

    # 模拟文档检索
    relevant = [doc for doc in state["documents"] if query.lower() in doc.lower()]

    return {
        **state,
        "retrieval_query": query,
        "relevant_docs": relevant
    }
```

### 3. 状态更新模式

```python
from typing import TypedDict, List
from copy import deepcopy

class GameState(TypedDict):
    player_name: str
    score: int
    level: int
    inventory: List[str]

# 模式 1: 完全重构（推荐）
def add_points(state: GameState, points: int) -> GameState:
    return {
        **state,
        "score": state["score"] + points
    }

# 模式 2: 深拷贝后修改
def level_up(state: GameState) -> GameState:
    new_state = deepcopy(state)
    new_state["level"] += 1
    new_state["score"] = 0  # 重置分数
    return new_state

# 模式 3: 构造函数式更新
def add_item(state: GameState, item: str) -> GameState:
    return {
        "player_name": state["player_name"],
        "score": state["score"],
        "level": state["level"],
        "inventory": state["inventory"] + [item]
    }
```

## 🔍 实际调试技巧

### 状态验证

```python
from typing import TypedDict

class ValidatedState(TypedDict):
    message: str
    count: int

def validate_state(state: ValidatedState) -> bool:
    """验证状态是否有效"""
    try:
        # 检查必需字段
        assert "message" in state, "缺少 message 字段"
        assert "count" in state, "缺少 count 字段"

        # 检查类型
        assert isinstance(state["message"], str), "message 必须是字符串"
        assert isinstance(state["count"], int), "count 必须是整数"

        # 检查业务逻辑
        assert state["count"] >= 0, "count 不能为负数"

        return True
    except AssertionError as e:
        print(f"状态验证失败: {e}")
        return False

# 使用
good_state: ValidatedState = {"message": "hello", "count": 5}
bad_state = {"message": "hello", "count": -1}

print(validate_state(good_state))  # True
print(validate_state(bad_state))   # False
```

### 状态追踪

```python
def track_state_changes(old_state: dict, new_state: dict) -> None:
    """追踪状态变化"""
    for key in set(old_state.keys()) | set(new_state.keys()):
        old_value = old_state.get(key, "<MISSING>")
        new_value = new_state.get(key, "<MISSING>")

        if old_value != new_value:
            print(f"🔄 {key}: {old_value} → {new_value}")

# 使用示例
old = {"message": "hello", "count": 1}
new = {"message": "hi there", "count": 2}
track_state_changes(old, new)
```

## ✅ 实战练习

### 练习 1：设计聊天机器人状态

设计一个聊天机器人的 TypedDict，包含以下需求：
- 当前用户消息
- 对话历史（最多保存20条）
- 用户情感状态（positive, neutral, negative）
- 机器人的响应模式（friendly, professional, casual）

```python
# TODO: 设计你的 TypedDict
```

<details>
<summary>🔍 查看参考答案</summary>

```python
from typing import TypedDict, List, Literal

class ChatbotState(TypedDict):
    user_message: str
    conversation_history: List[str]
    user_emotion: Literal["positive", "neutral", "negative"]
    response_mode: Literal["friendly", "professional", "casual"]
    message_count: int

# 带验证的创建函数
def create_chatbot_state(
    user_message: str,
    history: List[str] = None,
    emotion: str = "neutral",
    mode: str = "friendly"
) -> ChatbotState:
    if history is None:
        history = []

    # 限制历史长度
    if len(history) > 20:
        history = history[-20:]

    return {
        "user_message": user_message,
        "conversation_history": history,
        "user_emotion": emotion,
        "response_mode": mode,
        "message_count": len(history)
    }
```
</details>

### 练习 2：状态更新函数

实现以下节点函数：

```python
def emotion_analysis_node(state: ChatbotState) -> ChatbotState:
    """分析用户情感并更新状态"""
    # TODO: 实现情感分析逻辑
    pass

def response_generator_node(state: ChatbotState) -> ChatbotState:
    """根据情感和模式生成响应"""
    # TODO: 实现响应生成逻辑
    pass
```

## 🚀 下一步学习

掌握 TypedDict 后，接下来学习：
- `03-异步编程基础.md` - 现代 Python 应用开发
- `../02-图论与状态机/01-图的基本概念.md` - 理解 LangGraph 的数学基础

## 💡 关键要点

1. **TypedDict 是字典和类的完美结合**：有字典的灵活性，又有类的类型安全
2. **LangGraph 状态管理的核心**：AgentState 就是 TypedDict
3. **状态更新要 immutable**：总是返回新的状态对象
4. **善用类型检查**：mypy 是你的好朋友

---

*现在你已经掌握了 LangGraph 状态管理的核心技术！* 🎉