# 04-Memory机制

## 🎯 学习目标

通过本章学习，你将掌握：
- LangGraph 中短期和长期记忆的概念
- MemorySaver 的使用和配置
- 对话上下文的管理策略
- 跨会话记忆持久化技术

## 📚 Memory 基础概念

### 记忆的类型

在 LangGraph 中，记忆可以分为两大类：

1. **短期记忆（Short-term Memory）**
   - 会话内的对话历史
   - 临时上下文信息
   - 当前任务的中间状态

2. **长期记忆（Long-term Memory）**
   - 用户个人偏好
   - 历史会话摘要
   - 学习到的模式和知识

### 记忆存储层次

```
┌─────────────────────────────────────┐
│           应用层记忆                │
│  (用户偏好、领域知识、个性化配置)    │
├─────────────────────────────────────┤
│           会话层记忆                │
│    (对话历史、上下文、任务状态)     │
├─────────────────────────────────────┤
│           节点层记忆                │
│   (中间结果、临时变量、缓存数据)    │
└─────────────────────────────────────┘
```

## 🧠 短期记忆实现

### 1. 基础对话记忆

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated
import operator

class ConversationState(TypedDict):
    """对话状态定义"""
    messages: Annotated[List[dict], operator.add]
    user_context: dict
    conversation_summary: str
    turn_count: int

def create_basic_memory_graph():
    """创建基础记忆图"""

    def memory_node(state: ConversationState) -> ConversationState:
        """记忆处理节点"""
        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0)

        # 更新轮次计数
        state["turn_count"] = turn_count + 1

        # 如果消息过多，进行摘要
        if len(messages) > 20:
            summary = summarize_conversation(messages[:-10])  # 保留最近10条
            state["conversation_summary"] = summary
            state["messages"] = messages[-10:]  # 只保留最近10条消息

        return state

    def chat_node(state: ConversationState) -> ConversationState:
        """聊天处理节点"""
        messages = state.get("messages", [])

        if not messages:
            return state

        # 获取最新用户消息
        latest_message = messages[-1]
        user_input = latest_message.get("content", "")

        # 模拟AI回复（实际中会调用LLM）
        ai_response = f"收到您的消息：{user_input}。这是基于记忆的回复。"

        # 添加AI回复到消息历史
        ai_message = {
            "role": "assistant",
            "content": ai_response,
            "timestamp": time.time()
        }

        return {"messages": [ai_message]}

    def summarize_conversation(messages: List[dict]) -> str:
        """总结对话内容"""
        # 简化版本，实际中会使用LLM生成摘要
        topics = []
        for msg in messages:
            content = msg.get("content", "")
            if "问题" in content or "?" in content:
                topics.append("询问问题")
            elif "帮助" in content:
                topics.append("寻求帮助")

        return f"对话涉及：{', '.join(set(topics))}"

    # 构建图
    graph = StateGraph(ConversationState)
    graph.add_node("memory", memory_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "chat")
    graph.add_edge("chat", END)

    # 配置检查点
    memory_saver = MemorySaver()
    app = graph.compile(checkpointer=memory_saver)

    return app

# 使用示例
def use_memory_graph():
    """使用记忆图的示例"""
    app = create_basic_memory_graph()

    # 配置线程ID（用于区分不同用户/会话）
    config = {"configurable": {"thread_id": "user_123"}}

    # 第一次对话
    initial_state = {
        "messages": [{"role": "user", "content": "你好，我是小明"}],
        "user_context": {},
        "conversation_summary": "",
        "turn_count": 0
    }

    result1 = app.invoke(initial_state, config=config)
    print("第一次对话结果：", result1)

    # 第二次对话（会记住之前的上下文）
    second_state = {
        "messages": [{"role": "user", "content": "我刚才说了我的名字，你还记得吗？"}]
    }

    result2 = app.invoke(second_state, config=config)
    print("第二次对话结果：", result2)
```

### 2. 智能记忆管理

```python
class SmartMemoryManager:
    """智能记忆管理器"""

    def __init__(self, max_messages: int = 50, summary_trigger: int = 30):
        self.max_messages = max_messages
        self.summary_trigger = summary_trigger

    def manage_memory(self, state: ConversationState) -> ConversationState:
        """智能记忆管理"""
        messages = state.get("messages", [])

        if len(messages) <= self.summary_trigger:
            return state

        # 分析消息重要性
        important_messages = self._filter_important_messages(messages)

        # 如果重要消息过多，进行分层摘要
        if len(important_messages) > self.max_messages:
            return self._hierarchical_summarization(state, important_messages)
        else:
            return self._simple_truncation(state, important_messages)

    def _filter_important_messages(self, messages: List[dict]) -> List[dict]:
        """过滤重要消息"""
        important_messages = []

        for msg in messages:
            content = msg.get("content", "").lower()
            importance_score = 0

            # 计算重要性分数
            if any(keyword in content for keyword in ["重要", "关键", "记住", "不要忘记"]):
                importance_score += 3

            if any(keyword in content for keyword in ["名字", "联系方式", "地址", "偏好"]):
                importance_score += 2

            if msg.get("role") == "user" and len(content) > 50:
                importance_score += 1

            if importance_score >= 2 or msg in messages[-10:]:  # 最近10条始终保留
                important_messages.append(msg)

        return important_messages

    def _hierarchical_summarization(self, state: ConversationState, messages: List[dict]) -> ConversationState:
        """分层摘要"""
        # 将消息分组
        groups = self._group_messages_by_topic(messages)

        summaries = []
        recent_messages = []

        for topic, group_messages in groups.items():
            if len(group_messages) > 5:
                summary = self._summarize_group(topic, group_messages)
                summaries.append(summary)
            else:
                recent_messages.extend(group_messages)

        # 更新状态
        state["messages"] = recent_messages
        state["conversation_summary"] = "; ".join(summaries)

        return state

    def _group_messages_by_topic(self, messages: List[dict]) -> dict:
        """按主题分组消息"""
        # 简化版本，实际中可以使用NLP技术
        groups = {"general": []}

        for msg in messages:
            content = msg.get("content", "").lower()

            if any(keyword in content for keyword in ["工作", "职业", "公司"]):
                groups.setdefault("work", []).append(msg)
            elif any(keyword in content for keyword in ["家庭", "家人", "孩子"]):
                groups.setdefault("family", []).append(msg)
            elif any(keyword in content for keyword in ["爱好", "兴趣", "喜欢"]):
                groups.setdefault("hobbies", []).append(msg)
            else:
                groups["general"].append(msg)

        return groups

    def _summarize_group(self, topic: str, messages: List[dict]) -> str:
        """生成组摘要"""
        # 简化版本，实际中会使用LLM
        msg_count = len(messages)
        return f"{topic}相关对话({msg_count}条消息)"

    def _simple_truncation(self, state: ConversationState, important_messages: List[dict]) -> ConversationState:
        """简单截断"""
        if len(important_messages) > self.max_messages:
            # 保留最重要和最近的消息
            recent_count = self.max_messages // 2
            important_count = self.max_messages - recent_count

            selected_messages = (
                important_messages[:important_count] +  # 最重要的
                important_messages[-recent_count:]      # 最近的
            )

            state["messages"] = selected_messages

        return state

def create_smart_memory_graph():
    """创建智能记忆图"""
    memory_manager = SmartMemoryManager()

    def smart_memory_node(state: ConversationState) -> ConversationState:
        """智能记忆节点"""
        return memory_manager.manage_memory(state)

    def enhanced_chat_node(state: ConversationState) -> ConversationState:
        """增强聊天节点"""
        messages = state.get("messages", [])
        summary = state.get("conversation_summary", "")

        if not messages:
            return state

        # 构建上下文（包括摘要和最近消息）
        context = ""
        if summary:
            context += f"对话背景: {summary}\n\n"

        context += "最近对话:\n"
        for msg in messages[-5:]:  # 只包含最近5条
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            context += f"{role}: {content}\n"

        # 获取当前用户输入
        latest_message = messages[-1]
        if latest_message.get("role") == "user":
            user_input = latest_message.get("content", "")

            # 生成回复（实际中调用LLM，传入完整上下文）
            ai_response = f"基于我们的对话历史，我理解您说的是：{user_input}"

            return {
                "messages": [{
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": time.time()
                }]
            }

        return state

    # 构建图
    graph = StateGraph(ConversationState)
    graph.add_node("smart_memory", smart_memory_node)
    graph.add_node("enhanced_chat", enhanced_chat_node)

    graph.set_entry_point("smart_memory")
    graph.add_edge("smart_memory", "enhanced_chat")
    graph.add_edge("enhanced_chat", END)

    return graph.compile(checkpointer=MemorySaver())
```

## 💾 长期记忆实现

### 1. 用户偏好记忆

```python
import json
import sqlite3
from datetime import datetime, timedelta

class LongTermMemoryStore:
    """长期记忆存储"""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建用户偏好表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT,
                category TEXT,
                key TEXT,
                value TEXT,
                confidence REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, category, key)
            )
        ''')

        # 创建会话摘要表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_date DATE,
                summary TEXT,
                topics TEXT,
                sentiment TEXT,
                importance_score REAL DEFAULT 0.5
            )
        ''')

        # 创建知识库表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                content TEXT,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def store_preference(self, user_id: str, category: str, key: str, value: str, confidence: float = 1.0):
        """存储用户偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences
            (user_id, category, key, value, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, category, key, value, confidence))

        conn.commit()
        conn.close()

    def get_preferences(self, user_id: str, category: str = None) -> dict:
        """获取用户偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category:
            cursor.execute('''
                SELECT key, value, confidence FROM user_preferences
                WHERE user_id = ? AND category = ?
            ''', (user_id, category))
        else:
            cursor.execute('''
                SELECT category, key, value, confidence FROM user_preferences
                WHERE user_id = ?
            ''', (user_id,))

        results = cursor.fetchall()
        conn.close()

        if category:
            return {key: {"value": value, "confidence": confidence} for key, value, confidence in results}
        else:
            preferences = {}
            for row in results:
                cat, key, value, confidence = row
                if cat not in preferences:
                    preferences[cat] = {}
                preferences[cat][key] = {"value": value, "confidence": confidence}
            return preferences

    def store_session_summary(self, user_id: str, summary: str, topics: List[str], sentiment: str = "neutral", importance: float = 0.5):
        """存储会话摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO session_summaries
            (user_id, session_date, summary, topics, sentiment, importance_score)
            VALUES (?, DATE('now'), ?, ?, ?, ?)
        ''', (user_id, summary, json.dumps(topics), sentiment, importance))

        conn.commit()
        conn.close()

    def get_session_history(self, user_id: str, days: int = 30) -> List[dict]:
        """获取会话历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT session_date, summary, topics, sentiment, importance_score
            FROM session_summaries
            WHERE user_id = ? AND session_date >= DATE('now', '-{} days')
            ORDER BY session_date DESC
        '''.format(days), (user_id,))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "date": row[0],
                "summary": row[1],
                "topics": json.loads(row[2]),
                "sentiment": row[3],
                "importance": row[4]
            }
            for row in results
        ]

class PersonalizedMemoryGraph:
    """个性化记忆图"""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    def extract_preferences(self, state: ConversationState) -> ConversationState:
        """从对话中提取偏好"""
        messages = state.get("messages", [])
        user_id = state.get("user_id", "default")

        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "").lower()

                # 提取偏好模式
                preferences = self._analyze_preferences(content)

                for category, prefs in preferences.items():
                    for key, value in prefs.items():
                        self.memory_store.store_preference(
                            user_id, category, key, value["value"], value["confidence"]
                        )

        return state

    def _analyze_preferences(self, content: str) -> dict:
        """分析文本中的偏好信息"""
        preferences = {}

        # 简化的偏好提取（实际中可以使用NLP技术）
        if "喜欢" in content:
            if "音乐" in content:
                preferences.setdefault("entertainment", {})["music_preference"] = {
                    "value": "positive", "confidence": 0.8
                }
            if "电影" in content:
                preferences.setdefault("entertainment", {})["movie_preference"] = {
                    "value": "positive", "confidence": 0.8
                }

        if "不喜欢" in content or "讨厌" in content:
            if "等待" in content:
                preferences.setdefault("behavior", {})["patience_level"] = {
                    "value": "low", "confidence": 0.9
                }

        # 时间偏好
        if any(time_word in content for time_word in ["早上", "上午"]):
            preferences.setdefault("schedule", {})["preferred_time"] = {
                "value": "morning", "confidence": 0.7
            }

        return preferences

    def personalize_response(self, state: ConversationState) -> ConversationState:
        """基于偏好个性化回复"""
        user_id = state.get("user_id", "default")
        messages = state.get("messages", [])

        if not messages or messages[-1].get("role") != "user":
            return state

        # 获取用户偏好
        preferences = self.memory_store.get_preferences(user_id)

        # 获取最近会话历史
        recent_sessions = self.memory_store.get_session_history(user_id, days=7)

        # 基于偏好和历史生成个性化回复
        user_input = messages[-1].get("content", "")
        personalized_response = self._generate_personalized_response(
            user_input, preferences, recent_sessions
        )

        return {
            "messages": [{
                "role": "assistant",
                "content": personalized_response,
                "timestamp": time.time()
            }]
        }

    def _generate_personalized_response(self, user_input: str, preferences: dict, recent_sessions: List[dict]) -> str:
        """生成个性化回复"""
        response = f"根据我对您的了解，"

        # 基于偏好调整回复
        if "entertainment" in preferences:
            music_pref = preferences["entertainment"].get("music_preference")
            if music_pref and music_pref["value"] == "positive":
                response += "考虑到您喜欢音乐，"

        # 基于历史会话调整
        if recent_sessions:
            recent_topics = []
            for session in recent_sessions[:3]:  # 最近3次会话
                recent_topics.extend(session["topics"])

            if "工作" in recent_topics:
                response += "结合我们之前讨论过的工作话题，"

        response += f"对于您的问题"{user_input}"，我建议..."

        return response

def create_personalized_memory_graph():
    """创建个性化记忆图"""
    memory_store = LongTermMemoryStore()
    personalized_memory = PersonalizedMemoryGraph(memory_store)

    class PersonalizedState(ConversationState):
        user_id: str

    def extract_preferences_node(state: PersonalizedState) -> PersonalizedState:
        """提取偏好节点"""
        return personalized_memory.extract_preferences(state)

    def personalized_response_node(state: PersonalizedState) -> PersonalizedState:
        """个性化回复节点"""
        return personalized_memory.personalize_response(state)

    def session_summary_node(state: PersonalizedState) -> PersonalizedState:
        """会话摘要节点"""
        messages = state.get("messages", [])
        user_id = state.get("user_id", "default")

        if len(messages) > 0:
            # 简化的摘要生成
            topics = ["一般对话"]  # 实际中会分析消息内容
            summary = f"用户进行了{len(messages)}轮对话"

            memory_store.store_session_summary(user_id, summary, topics)

        return state

    # 构建图
    graph = StateGraph(PersonalizedState)
    graph.add_node("extract_preferences", extract_preferences_node)
    graph.add_node("personalized_response", personalized_response_node)
    graph.add_node("session_summary", session_summary_node)

    graph.set_entry_point("extract_preferences")
    graph.add_edge("extract_preferences", "personalized_response")
    graph.add_edge("personalized_response", "session_summary")
    graph.add_edge("session_summary", END)

    return graph.compile(checkpointer=MemorySaver())
```

## 🔄 记忆检索和更新

### 1. 智能记忆检索

```python
class MemoryRetriever:
    """记忆检索器"""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    def retrieve_relevant_memory(self, user_id: str, query: str, context: dict = None) -> dict:
        """检索相关记忆"""
        relevant_memory = {
            "preferences": {},
            "session_history": [],
            "knowledge": [],
            "relevance_score": 0.0
        }

        # 检索偏好
        all_preferences = self.memory_store.get_preferences(user_id)
        relevant_preferences = self._filter_relevant_preferences(query, all_preferences)
        relevant_memory["preferences"] = relevant_preferences

        # 检索会话历史
        recent_sessions = self.memory_store.get_session_history(user_id, days=30)
        relevant_sessions = self._filter_relevant_sessions(query, recent_sessions)
        relevant_memory["session_history"] = relevant_sessions

        # 计算总体相关性分数
        relevance_score = self._calculate_relevance_score(query, relevant_memory)
        relevant_memory["relevance_score"] = relevance_score

        return relevant_memory

    def _filter_relevant_preferences(self, query: str, preferences: dict) -> dict:
        """过滤相关偏好"""
        relevant = {}
        query_lower = query.lower()

        for category, prefs in preferences.items():
            for key, value_info in prefs.items():
                # 简单的关键词匹配（实际中可以使用语义搜索）
                if any(keyword in query_lower for keyword in [category, key, value_info["value"]]):
                    relevant.setdefault(category, {})[key] = value_info

        return relevant

    def _filter_relevant_sessions(self, query: str, sessions: List[dict]) -> List[dict]:
        """过滤相关会话"""
        relevant = []
        query_lower = query.lower()

        for session in sessions:
            relevance = 0

            # 检查摘要相关性
            if any(word in session["summary"].lower() for word in query_lower.split()):
                relevance += 1

            # 检查主题相关性
            for topic in session["topics"]:
                if topic.lower() in query_lower:
                    relevance += 2

            # 根据重要性和时间加权
            relevance *= session["importance"]

            if relevance > 0:
                session["relevance"] = relevance
                relevant.append(session)

        # 按相关性排序
        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant[:5]  # 返回最相关的5个会话

    def _calculate_relevance_score(self, query: str, memory: dict) -> float:
        """计算总体相关性分数"""
        score = 0.0

        # 偏好相关性
        if memory["preferences"]:
            score += 0.3

        # 会话历史相关性
        if memory["session_history"]:
            avg_session_relevance = sum(s.get("relevance", 0) for s in memory["session_history"]) / len(memory["session_history"])
            score += 0.7 * min(avg_session_relevance / 3.0, 1.0)  # 标准化到0-1

        return score

class AdaptiveMemorySystem:
    """自适应记忆系统"""

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store
        self.retriever = MemoryRetriever(memory_store)

    def update_memory_importance(self, user_id: str, feedback: dict):
        """根据用户反馈更新记忆重要性"""
        # 这里可以根据用户的反馈调整记忆的重要性和置信度
        pass

    def forget_outdated_memory(self, user_id: str, days_threshold: int = 90):
        """遗忘过时记忆"""
        # 实现记忆衰减机制
        pass

    def consolidate_memory(self, user_id: str):
        """记忆巩固"""
        # 将短期记忆转换为长期记忆
        pass
```

## 🎮 实践练习

### 练习 1：构建学习记忆系统

创建一个能够学习和记住用户习惯的系统：

```python
def learning_memory_system():
    """
    练习：构建学习记忆系统

    要求：
    1. 记录用户的交互模式
    2. 学习用户的偏好变化
    3. 预测用户可能的需求
    4. 提供个性化建议
    """
    # TODO: 实现你的学习记忆系统
    pass
```

### 练习 2：多用户记忆隔离

实现多用户环境下的记忆隔离和共享：

```python
def multi_user_memory_system():
    """
    练习：多用户记忆系统

    要求：
    1. 确保用户记忆隔离
    2. 支持团队共享记忆
    3. 实现记忆权限管理
    4. 提供记忆同步机制
    """
    # TODO: 实现你的多用户记忆系统
    pass
```

## 🚀 高级记忆特性

### 1. 记忆压缩和检索

```python
class MemoryCompressor:
    """记忆压缩器"""

    def compress_conversation(self, messages: List[dict]) -> dict:
        """压缩对话为结构化记忆"""
        compressed = {
            "facts": [],
            "preferences": [],
            "context": [],
            "emotions": [],
            "decisions": []
        }

        for message in messages:
            content = message.get("content", "")

            # 提取事实信息
            facts = self._extract_facts(content)
            compressed["facts"].extend(facts)

            # 提取偏好信息
            preferences = self._extract_preferences(content)
            compressed["preferences"].extend(preferences)

            # 提取情感信息
            emotions = self._extract_emotions(content)
            compressed["emotions"].extend(emotions)

        return compressed

    def _extract_facts(self, content: str) -> List[str]:
        """提取事实信息"""
        # 简化实现，实际中会使用NER等技术
        facts = []
        if "我是" in content or "我叫" in content:
            facts.append(f"身份信息: {content}")
        return facts

    def _extract_preferences(self, content: str) -> List[str]:
        """提取偏好信息"""
        preferences = []
        if "喜欢" in content:
            preferences.append(f"偏好: {content}")
        return preferences

    def _extract_emotions(self, content: str) -> List[str]:
        """提取情感信息"""
        emotions = []
        if any(word in content for word in ["开心", "高兴", "愉快"]):
            emotions.append("positive")
        elif any(word in content for word in ["难过", "伤心", "沮丧"]):
            emotions.append("negative")
        return emotions
```

### 2. 记忆一致性检查

```python
class MemoryConsistencyChecker:
    """记忆一致性检查器"""

    def check_consistency(self, user_id: str, new_memory: dict, existing_memories: List[dict]) -> dict:
        """检查记忆一致性"""
        conflicts = []
        confirmations = []

        for existing in existing_memories:
            conflict_score = self._calculate_conflict(new_memory, existing)
            if conflict_score > 0.8:
                conflicts.append({
                    "existing": existing,
                    "new": new_memory,
                    "conflict_score": conflict_score,
                    "type": "high_conflict"
                })
            elif conflict_score > 0.3:
                confirmations.append({
                    "existing": existing,
                    "new": new_memory,
                    "similarity_score": 1 - conflict_score,
                    "type": "confirmation"
                })

        return {
            "conflicts": conflicts,
            "confirmations": confirmations,
            "consistency_score": 1 - (len(conflicts) / max(len(existing_memories), 1))
        }

    def _calculate_conflict(self, memory1: dict, memory2: dict) -> float:
        """计算记忆冲突分数"""
        # 简化实现，实际中需要更复杂的语义比较
        if memory1.get("type") != memory2.get("type"):
            return 0.0

        content1 = memory1.get("content", "").lower()
        content2 = memory2.get("content", "").lower()

        # 检查直接矛盾
        if ("喜欢" in content1 and "不喜欢" in content2) or ("不喜欢" in content1 and "喜欢" in content2):
            return 1.0

        return 0.0
```

## 📚 推荐阅读

- [LangGraph 官方文档 - Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- 项目示例：`Agents/02-Memory_Agent.py`

---

**下一章预告**：学习完 Memory 机制后，我们将探索 Human-in-the-Loop 特性，了解如何在 AI 系统中优雅地集成人工干预机制。