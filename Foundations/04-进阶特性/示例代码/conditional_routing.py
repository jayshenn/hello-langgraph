#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件路由示例代码
演示 LangGraph 中高级条件路由的实现
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time
import random


class RouterState(TypedDict):
    """路由状态定义"""
    user_input: str
    intent: str
    confidence: float
    context: dict
    response: str
    route_history: List[str]


def create_intelligent_router():
    """创建智能路由系统"""

    def intent_classifier(state: RouterState) -> RouterState:
        """意图分类节点"""
        user_input = state.get("user_input", "").lower()

        # 简化的意图分类逻辑
        intents = {
            "问候": ["你好", "hello", "hi", "嗨"],
            "询问": ["什么", "怎么", "为什么", "如何", "?", "？"],
            "请求": ["请", "帮", "可以", "能否"],
            "投诉": ["问题", "错误", "不满意", "投诉"],
            "结束": ["再见", "bye", "结束", "退出"]
        }

        detected_intent = "其他"
        confidence = 0.5

        for intent, keywords in intents.items():
            if any(keyword in user_input for keyword in keywords):
                detected_intent = intent
                confidence = 0.8 + random.random() * 0.2  # 0.8-1.0
                break

        state["intent"] = detected_intent
        state["confidence"] = confidence

        # 记录路由历史
        route_history = state.get("route_history", [])
        route_history.append(f"intent_classifier -> {detected_intent}({confidence:.2f})")
        state["route_history"] = route_history

        return state

    def context_analyzer(state: RouterState) -> RouterState:
        """上下文分析节点"""
        user_input = state.get("user_input", "")
        route_history = state.get("route_history", [])

        # 分析上下文
        context = {
            "input_length": len(user_input),
            "has_question_mark": "?" in user_input or "？" in user_input,
            "previous_interactions": len(route_history),
            "urgency_keywords": any(word in user_input.lower() for word in ["急", "马上", "立即", "紧急"]),
            "analysis_time": time.time()
        }

        state["context"] = context

        route_history.append(f"context_analyzer -> 上下文已分析")
        state["route_history"] = route_history

        return state

    def greeting_handler(state: RouterState) -> RouterState:
        """问候处理节点"""
        user_input = state.get("user_input", "")
        context = state.get("context", {})

        if context.get("previous_interactions", 0) > 0:
            response = f"又见面了！关于您的问题 '{user_input}'，我很乐意继续帮助您。"
        else:
            response = f"您好！欢迎使用我们的服务。我看到您说的是 '{user_input}'，请问需要什么帮助？"

        state["response"] = response

        route_history = state.get("route_history", [])
        route_history.append("greeting_handler -> 问候已处理")
        state["route_history"] = route_history

        return state

    def question_handler(state: RouterState) -> RouterState:
        """问题处理节点"""
        user_input = state.get("user_input", "")
        context = state.get("context", {})

        if context.get("urgency_keywords", False):
            response = f"我理解您的急迫心情。关于 '{user_input}' 这个问题，让我优先为您处理。"
        else:
            response = f"这是一个很好的问题：'{user_input}'。让我来为您详细解答。"

        state["response"] = response

        route_history = state.get("route_history", [])
        route_history.append("question_handler -> 问题已处理")
        state["route_history"] = route_history

        return state

    def request_handler(state: RouterState) -> RouterState:
        """请求处理节点"""
        user_input = state.get("user_input", "")

        response = f"我收到了您的请求：'{user_input}'。我会尽力帮助您完成这个任务。"

        state["response"] = response

        route_history = state.get("route_history", [])
        route_history.append("request_handler -> 请求已处理")
        state["route_history"] = route_history

        return state

    def complaint_handler(state: RouterState) -> RouterState:
        """投诉处理节点"""
        user_input = state.get("user_input", "")

        response = f"非常抱歉让您遇到了问题。关于 '{user_input}'，我会认真对待并尽快解决。"

        state["response"] = response

        route_history = state.get("route_history", [])
        route_history.append("complaint_handler -> 投诉已处理")
        state["route_history"] = route_history

        return state

    def default_handler(state: RouterState) -> RouterState:
        """默认处理节点"""
        user_input = state.get("user_input", "")
        confidence = state.get("confidence", 0.0)

        if confidence < 0.6:
            response = f"我不太确定如何理解 '{user_input}'，能否请您换个方式表达？"
        else:
            response = f"感谢您的输入：'{user_input}'。我已经记录下来了。"

        state["response"] = response

        route_history = state.get("route_history", [])
        route_history.append("default_handler -> 默认处理完成")
        state["route_history"] = route_history

        return state

    def intelligent_router(state: RouterState) -> str:
        """智能路由函数"""
        intent = state.get("intent", "其他")
        confidence = state.get("confidence", 0.0)
        context = state.get("context", {})

        # 基于意图和置信度的路由决策
        if confidence < 0.6:
            return "default"

        # 紧急情况优先处理
        if context.get("urgency_keywords", False):
            if intent in ["投诉", "请求"]:
                return intent
            else:
                return "request"  # 紧急情况按请求处理

        # 正常路由逻辑
        route_map = {
            "问候": "greeting",
            "询问": "question",
            "请求": "request",
            "投诉": "complaint",
            "结束": END
        }

        return route_map.get(intent, "default")

    # 构建路由图
    graph = StateGraph(RouterState)

    # 添加节点
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("context_analyzer", context_analyzer)
    graph.add_node("greeting", greeting_handler)
    graph.add_node("question", question_handler)
    graph.add_node("request", request_handler)
    graph.add_node("complaint", complaint_handler)
    graph.add_node("default", default_handler)

    # 设置入口点
    graph.set_entry_point("intent_classifier")

    # 添加边
    graph.add_edge("intent_classifier", "context_analyzer")

    # 添加条件边
    graph.add_conditional_edges(
        "context_analyzer",
        intelligent_router,
        {
            "greeting": "greeting",
            "question": "question",
            "request": "request",
            "complaint": "complaint",
            "default": "default",
            END: END
        }
    )

    # 所有处理节点都结束到END
    for handler in ["greeting", "question", "request", "complaint", "default"]:
        graph.add_edge(handler, END)

    return graph.compile()


def demo_conditional_routing():
    """演示条件路由"""
    print("🚀 条件路由演示开始")
    print("=" * 50)

    app = create_intelligent_router()

    # 测试用例
    test_cases = [
        "你好，我想了解你们的服务",
        "为什么我的订单还没发货？",
        "请帮我查询账户余额",
        "你们的服务有问题，我要投诉",
        "急！我的账户被锁定了，请立即处理",
        "abcdefg",  # 低置信度测试
        "再见"
    ]

    for i, user_input in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {user_input}")
        print("-" * 30)

        initial_state = {
            "user_input": user_input,
            "intent": "",
            "confidence": 0.0,
            "context": {},
            "response": "",
            "route_history": []
        }

        result = app.invoke(initial_state)

        print(f"📝 意图识别: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"🎯 最终回复: {result['response']}")
        print(f"🛤️  路由历史:")
        for step in result['route_history']:
            print(f"   - {step}")


if __name__ == "__main__":
    demo_conditional_routing()