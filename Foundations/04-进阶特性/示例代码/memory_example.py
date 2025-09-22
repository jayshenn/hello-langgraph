#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory机制示例代码
演示 LangGraph 中记忆管理的实现
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated
import operator
import time
import json


class MemoryState(TypedDict):
    """记忆状态定义"""
    messages: Annotated[List[dict], operator.add]
    user_profile: dict
    conversation_summary: str
    interaction_count: int
    memory_metadata: dict


def create_smart_memory_system():
    """创建智能记忆系统"""

    def memory_manager(state: MemoryState) -> MemoryState:
        """记忆管理器"""
        messages = state.get("messages", [])
        interaction_count = state.get("interaction_count", 0)

        print(f"🧠 记忆管理器：处理第 {interaction_count + 1} 次交互")

        # 更新交互计数
        state["interaction_count"] = interaction_count + 1

        # 如果消息过多，进行智能摘要
        if len(messages) > 10:
            print("📝 消息过多，进行智能摘要...")
            summary = generate_conversation_summary(messages[:-5])  # 保留最近5条
            state["conversation_summary"] = summary
            state["messages"] = messages[-5:]  # 只保留最近5条消息

            # 更新记忆元数据
            metadata = state.get("memory_metadata", {})
            metadata.update({
                "last_summary_time": time.time(),
                "summarized_messages_count": len(messages) - 5,
                "total_interactions": interaction_count + 1
            })
            state["memory_metadata"] = metadata

        return state

    def profile_extractor(state: MemoryState) -> MemoryState:
        """用户画像提取器"""
        messages = state.get("messages", [])
        user_profile = state.get("user_profile", {})

        print("👤 提取用户画像信息...")

        # 从最新消息中提取用户信息
        latest_messages = messages[-3:] if len(messages) >= 3 else messages

        for message in latest_messages:
            if message.get("role") == "user":
                content = message.get("content", "").lower()

                # 提取个人信息
                if "我叫" in content or "我是" in content:
                    # 简化的姓名提取
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ["我叫", "我是"] and i + 1 < len(words):
                            user_profile["name"] = words[i + 1]

                # 提取偏好信息
                if "喜欢" in content:
                    preferences = user_profile.get("preferences", [])
                    preference_text = content.split("喜欢")[1].split()[0]
                    if preference_text not in preferences:
                        preferences.append(preference_text)
                    user_profile["preferences"] = preferences

                # 提取地理信息
                if any(city in content for city in ["北京", "上海", "广州", "深圳"]):
                    for city in ["北京", "上海", "广州", "深圳"]:
                        if city in content:
                            user_profile["location"] = city
                            break

        state["user_profile"] = user_profile

        if user_profile:
            print(f"   📊 更新用户画像: {json.dumps(user_profile, ensure_ascii=False)}")

        return state

    def contextual_responder(state: MemoryState) -> MemoryState:
        """上下文响应器"""
        messages = state.get("messages", [])
        user_profile = state.get("user_profile", {})
        conversation_summary = state.get("conversation_summary", "")

        if not messages or messages[-1].get("role") != "user":
            return state

        user_message = messages[-1].get("content", "")
        print(f"💬 响应用户消息: {user_message}")

        # 构建个性化响应
        response = generate_personalized_response(
            user_message, user_profile, conversation_summary
        )

        # 添加AI响应到消息历史
        ai_message = {
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "personalization_used": bool(user_profile)
        }

        return {"messages": [ai_message]}

    # 构建记忆图
    graph = StateGraph(MemoryState)
    graph.add_node("memory_manager", memory_manager)
    graph.add_node("profile_extractor", profile_extractor)
    graph.add_node("contextual_responder", contextual_responder)

    graph.set_entry_point("memory_manager")
    graph.add_edge("memory_manager", "profile_extractor")
    graph.add_edge("profile_extractor", "contextual_responder")
    graph.add_edge("contextual_responder", END)

    # 配置记忆保存器
    memory_saver = MemorySaver()
    app = graph.compile(checkpointer=memory_saver)

    return app


def generate_conversation_summary(messages: List[dict]) -> str:
    """生成对话摘要"""
    if not messages:
        return ""

    topics = []
    user_interests = []

    for message in messages:
        content = message.get("content", "").lower()

        # 提取主题
        if any(word in content for word in ["关于", "讨论", "说说"]):
            topics.append("话题讨论")

        if any(word in content for word in ["问题", "怎么", "为什么"]):
            topics.append("问题咨询")

        if any(word in content for word in ["喜欢", "爱好", "兴趣"]):
            user_interests.append("个人兴趣")

    # 生成摘要
    summary_parts = []
    if topics:
        summary_parts.append(f"讨论了{len(set(topics))}个主题")
    if user_interests:
        summary_parts.append("分享了个人兴趣")

    return "，".join(summary_parts) if summary_parts else "进行了一般性对话"


def generate_personalized_response(user_message: str, user_profile: dict, summary: str) -> str:
    """生成个性化响应"""
    response_parts = []

    # 使用用户姓名
    name = user_profile.get("name")
    if name:
        response_parts.append(f"{name}")

    # 基于位置信息
    location = user_profile.get("location")
    if location and "天气" in user_message:
        response_parts.append(f"在{location}的天气确实需要关注")

    # 基于偏好信息
    preferences = user_profile.get("preferences", [])
    if preferences and any(pref in user_message for pref in preferences):
        response_parts.append("看来这和您的兴趣爱好相关")

    # 基于对话历史
    if summary and "之前" in user_message:
        response_parts.append(f"回顾我们的对话，{summary}")

    # 构建最终响应
    if response_parts:
        personalized_prefix = "，".join(response_parts) + "。"
    else:
        personalized_prefix = ""

    # 基本响应
    basic_response = f"我理解您说的'{user_message}'。"

    return personalized_prefix + basic_response


def create_long_term_memory_demo():
    """创建长期记忆演示"""

    class LongTermState(TypedDict):
        user_id: str
        session_memories: List[dict]
        learned_patterns: dict
        user_expertise: dict

    def pattern_learner(state: LongTermState) -> LongTermState:
        """模式学习器"""
        user_id = state.get("user_id", "")
        session_memories = state.get("session_memories", [])

        print(f"🔍 为用户 {user_id} 学习交互模式...")

        # 分析用户行为模式
        patterns = {}

        # 分析问题类型偏好
        question_types = []
        for memory in session_memories:
            if "type" in memory:
                question_types.append(memory["type"])

        if question_types:
            from collections import Counter
            type_counts = Counter(question_types)
            patterns["preferred_question_types"] = dict(type_counts.most_common(3))

        # 分析活跃时间
        timestamps = [m.get("timestamp", 0) for m in session_memories if "timestamp" in m]
        if timestamps:
            hours = [time.localtime(ts).tm_hour for ts in timestamps]
            from collections import Counter
            hour_counts = Counter(hours)
            patterns["active_hours"] = dict(hour_counts.most_common(3))

        learned_patterns = state.get("learned_patterns", {})
        learned_patterns.update(patterns)
        state["learned_patterns"] = learned_patterns

        print(f"   📈 学习到的模式: {json.dumps(patterns, ensure_ascii=False)}")

        return state

    def expertise_tracker(state: LongTermState) -> LongTermState:
        """专业度跟踪器"""
        session_memories = state.get("session_memories", [])
        user_expertise = state.get("user_expertise", {})

        # 基于用户问题和回复质量评估专业度
        for memory in session_memories:
            topic = memory.get("topic", "general")
            quality = memory.get("quality_score", 0.5)

            if topic not in user_expertise:
                user_expertise[topic] = {"score": 0.5, "interactions": 0}

            # 更新专业度分数（简化算法）
            current_score = user_expertise[topic]["score"]
            interactions = user_expertise[topic]["interactions"]

            new_score = (current_score * interactions + quality) / (interactions + 1)
            user_expertise[topic] = {
                "score": new_score,
                "interactions": interactions + 1
            }

        state["user_expertise"] = user_expertise

        print(f"🎓 用户专业度更新: {json.dumps(user_expertise, ensure_ascii=False)}")

        return state

    # 构建长期记忆图
    graph = StateGraph(LongTermState)
    graph.add_node("pattern_learner", pattern_learner)
    graph.add_node("expertise_tracker", expertise_tracker)

    graph.set_entry_point("pattern_learner")
    graph.add_edge("pattern_learner", "expertise_tracker")
    graph.add_edge("expertise_tracker", END)

    return graph.compile()


def demo_memory_system():
    """演示记忆系统"""
    print("🚀 Memory机制演示开始")
    print("=" * 60)

    # 演示1: 智能记忆系统
    print("\n🎯 演示1: 智能对话记忆")
    print("-" * 40)

    memory_app = create_smart_memory_system()

    # 模拟多轮对话
    conversation_scenarios = [
        "你好，我叫张三",
        "我来自北京，喜欢编程",
        "你能帮我解决一个Python问题吗？",
        "我之前说过我来自哪里？",
        "我的兴趣爱好是什么？",
        "今天北京天气怎么样？",
        "谢谢你记住了我的信息",
        "还有其他问题吗？",
        "我想了解更多关于AI的知识",
        "再见"
    ]

    config = {"configurable": {"thread_id": "user_conversation_001"}}

    for i, user_input in enumerate(conversation_scenarios, 1):
        print(f"\n👤 用户 ({i}/10): {user_input}")

        state = {
            "messages": [{"role": "user", "content": user_input, "timestamp": time.time()}],
            "user_profile": {},
            "conversation_summary": "",
            "interaction_count": 0,
            "memory_metadata": {}
        }

        result = memory_app.invoke(state, config=config)

        # 显示AI回复
        if result["messages"]:
            ai_response = result["messages"][-1]["content"]
            print(f"🤖 AI: {ai_response}")

        # 显示记忆信息
        if result.get("user_profile"):
            print(f"💾 用户画像: {json.dumps(result['user_profile'], ensure_ascii=False)}")

        time.sleep(1)  # 模拟对话间隔

    # 演示2: 长期记忆学习
    print("\n\n🎯 演示2: 长期记忆学习")
    print("-" * 40)

    long_term_app = create_long_term_memory_demo()

    # 模拟历史会话记录
    historical_sessions = [
        {"type": "technical", "topic": "python", "quality_score": 0.8, "timestamp": time.time() - 86400},
        {"type": "casual", "topic": "general", "quality_score": 0.6, "timestamp": time.time() - 43200},
        {"type": "technical", "topic": "javascript", "quality_score": 0.9, "timestamp": time.time() - 3600},
        {"type": "help", "topic": "python", "quality_score": 0.7, "timestamp": time.time() - 1800},
    ]

    long_term_state = {
        "user_id": "zhang_san",
        "session_memories": historical_sessions,
        "learned_patterns": {},
        "user_expertise": {}
    }

    result = long_term_app.invoke(long_term_state)

    print("\n🎉 Memory机制演示完成！")
    print("\n📊 最终学习结果:")
    print(f"   🧠 学习模式: {json.dumps(result.get('learned_patterns', {}), ensure_ascii=False)}")
    print(f"   🎓 专业度评估: {json.dumps(result.get('user_expertise', {}), ensure_ascii=False)}")


if __name__ == "__main__":
    demo_memory_system()