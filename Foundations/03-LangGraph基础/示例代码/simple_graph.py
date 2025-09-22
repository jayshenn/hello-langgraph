#!/usr/bin/env python3
"""
simple_graph.py - LangGraph 最简单完整示例

这个示例演示了 LangGraph 的基本构建流程：
1. 状态定义
2. 节点创建
3. 图构建
4. 编译与执行

运行方式：
    python simple_graph.py
"""

import time
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# 1. 状态定义：定义整个图中传递的数据结构
# =============================================================================

class SimpleState(TypedDict):
    """
    简单状态定义

    包含基本的聊天状态信息：
    - user_input: 用户输入
    - processed_input: 处理后的输入
    - response: 系统响应
    - step_count: 处理步数
    - is_complete: 是否完成
    """
    user_input: str
    processed_input: Optional[str]
    response: Optional[str]
    step_count: int
    is_complete: bool


# =============================================================================
# 2. 节点函数：定义图中的处理逻辑
# =============================================================================

def input_validation_node(state: SimpleState) -> SimpleState:
    """
    输入验证节点

    验证用户输入的有效性，并进行基本的清理工作
    """
    print(f"🔍 [输入验证] 处理用户输入: {state['user_input']}")

    user_input = state["user_input"]

    # 基本验证
    if not user_input or not user_input.strip():
        return {
            **state,
            "processed_input": None,
            "response": "抱歉，我没有收到有效的输入。请重新输入。",
            "step_count": state.get("step_count", 0) + 1,
            "is_complete": True
        }

    # 输入清理
    cleaned_input = user_input.strip()

    # 模拟处理时间
    time.sleep(0.5)

    return {
        **state,
        "processed_input": cleaned_input,
        "step_count": state.get("step_count", 0) + 1
    }


def intent_analysis_node(state: SimpleState) -> SimpleState:
    """
    意图分析节点

    分析用户输入的意图，决定如何响应
    """
    print(f"🧠 [意图分析] 分析输入: {state['processed_input']}")

    processed_input = state["processed_input"]

    # 简单的意图识别
    if any(keyword in processed_input.lower() for keyword in ["你好", "hello", "hi"]):
        intent = "greeting"
    elif any(keyword in processed_input.lower() for keyword in ["再见", "bye", "goodbye"]):
        intent = "farewell"
    elif any(keyword in processed_input.lower() for keyword in ["帮助", "help"]):
        intent = "help"
    elif "?" in processed_input or "吗" in processed_input:
        intent = "question"
    else:
        intent = "general"

    # 模拟分析时间
    time.sleep(0.3)

    return {
        **state,
        "intent": intent,
        "step_count": state["step_count"] + 1
    }


def response_generation_node(state: SimpleState) -> SimpleState:
    """
    响应生成节点

    根据分析的意图生成相应的回复
    """
    intent = state.get("intent", "general")
    processed_input = state["processed_input"]

    print(f"💬 [响应生成] 基于意图 '{intent}' 生成响应")

    # 根据意图生成响应
    responses = {
        "greeting": f"你好！很高兴见到你。你说的是：{processed_input}",
        "farewell": f"再见！希望很快再次见到你。",
        "help": f"我可以帮助你处理各种问题。请告诉我你需要什么帮助。",
        "question": f"这是一个很好的问题：{processed_input}。让我想想...",
        "general": f"我理解你说的是：{processed_input}。这很有趣！"
    }

    response = responses.get(intent, f"我收到了你的消息：{processed_input}")

    # 模拟生成时间
    time.sleep(0.4)

    return {
        **state,
        "response": response,
        "step_count": state["step_count"] + 1,
        "is_complete": True
    }


def error_handling_node(state: SimpleState) -> SimpleState:
    """
    错误处理节点

    处理执行过程中的错误情况
    """
    print("❌ [错误处理] 处理异常情况")

    return {
        **state,
        "response": "抱歉，处理过程中出现了错误。请稍后重试。",
        "step_count": state.get("step_count", 0) + 1,
        "is_complete": True
    }


# =============================================================================
# 3. 路由函数：控制图的执行流程
# =============================================================================

def should_continue_processing(state: SimpleState) -> str:
    """
    决定是否继续处理的路由函数

    根据输入验证的结果决定下一步操作
    """
    if state.get("processed_input") is None:
        # 输入无效，直接结束
        return "end"
    else:
        # 输入有效，继续分析
        return "analyze"


def route_after_analysis(state: SimpleState) -> str:
    """
    分析后的路由函数

    在意图分析后决定是否继续生成响应
    """
    intent = state.get("intent")

    if intent in ["greeting", "farewell", "help", "question", "general"]:
        return "generate_response"
    else:
        # 未知意图，进入错误处理
        return "handle_error"


# =============================================================================
# 4. 图构建：组装所有组件
# =============================================================================

def create_simple_graph() -> StateGraph:
    """
    创建简单的聊天图

    Returns:
        StateGraph: 配置好的图对象
    """
    print("🔧 构建 LangGraph...")

    # 创建图实例
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("validate_input", input_validation_node)
    graph.add_node("analyze_intent", intent_analysis_node)
    graph.add_node("generate_response", response_generation_node)
    graph.add_node("handle_error", error_handling_node)

    # 设置入口点
    graph.add_edge(START, "validate_input")

    # 添加条件边：根据输入验证结果决定路径
    graph.add_conditional_edges(
        "validate_input",
        should_continue_processing,
        {
            "analyze": "analyze_intent",
            "end": END
        }
    )

    # 添加条件边：根据意图分析结果决定路径
    graph.add_conditional_edges(
        "analyze_intent",
        route_after_analysis,
        {
            "generate_response": "generate_response",
            "handle_error": "handle_error"
        }
    )

    # 添加结束边
    graph.add_edge("generate_response", END)
    graph.add_edge("handle_error", END)

    print("✅ 图构建完成!")
    return graph


# =============================================================================
# 5. 主执行函数
# =============================================================================

def run_simple_example():
    """运行简单示例"""
    print("=" * 60)
    print("🚀 LangGraph 简单示例")
    print("=" * 60)

    # 创建图
    graph = create_simple_graph()

    # 创建检查点保存器（用于状态持久化）
    checkpointer = MemorySaver()

    # 编译图
    app = graph.compile(checkpointer=checkpointer)
    print("📦 图编译完成!")

    # 测试用例
    test_cases = [
        "你好",
        "帮助",
        "今天天气怎么样？",
        "再见",
        "",  # 空输入测试
        "   ",  # 空白输入测试
        "这是一个普通的消息"
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: '{test_input}'")
        print("-" * 40)

        # 创建配置（每个测试用例使用不同的线程）
        config = {
            "configurable": {
                "thread_id": f"test_case_{i}"
            }
        }

        # 创建初始状态
        initial_state = {
            "user_input": test_input,
            "processed_input": None,
            "response": None,
            "step_count": 0,
            "is_complete": False
        }

        try:
            # 执行图
            result = app.invoke(initial_state, config=config)

            # 显示结果
            print(f"✅ 处理完成:")
            print(f"   - 输入: {result['user_input']}")
            print(f"   - 响应: {result['response']}")
            print(f"   - 步数: {result['step_count']}")
            print(f"   - 完成: {result['is_complete']}")

        except Exception as e:
            print(f"❌ 执行出错: {e}")

        print("-" * 40)


def run_streaming_example():
    """运行流式执行示例"""
    print("\n" + "=" * 60)
    print("🌊 LangGraph 流式执行示例")
    print("=" * 60)

    # 创建图和应用
    graph = create_simple_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    test_input = "你好，请帮助我理解 LangGraph"
    print(f"📝 流式处理: '{test_input}'")
    print("-" * 40)

    config = {
        "configurable": {
            "thread_id": "streaming_test"
        }
    }

    initial_state = {
        "user_input": test_input,
        "processed_input": None,
        "response": None,
        "step_count": 0,
        "is_complete": False
    }

    print("🌊 开始流式执行...")

    # 流式执行，实时显示每个节点的更新
    for i, chunk in enumerate(app.stream(initial_state, config=config)):
        print(f"  📦 流式更新 {i+1}: {chunk}")

    print("✅ 流式执行完成!")


def interactive_chat():
    """交互式聊天示例"""
    print("\n" + "=" * 60)
    print("💬 LangGraph 交互式聊天")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出聊天")
    print("-" * 40)

    # 创建图和应用
    graph = create_simple_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 使用同一个线程保持对话上下文
    config = {
        "configurable": {
            "thread_id": "interactive_chat"
        }
    }

    chat_count = 0

    while True:
        try:
            # 获取用户输入
            user_input = input(f"\n👤 你 ({chat_count + 1}): ").strip()

            # 检查退出条件
            if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                print("👋 再见！感谢使用 LangGraph 聊天示例！")
                break

            if not user_input:
                print("⚠️  请输入有效内容")
                continue

            # 创建状态
            initial_state = {
                "user_input": user_input,
                "processed_input": None,
                "response": None,
                "step_count": 0,
                "is_complete": False
            }

            print("🤖 助手正在思考...")

            # 执行处理
            result = app.invoke(initial_state, config=config)

            # 显示响应
            print(f"🤖 助手: {result['response']}")

            chat_count += 1

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 处理出错: {e}")


# =============================================================================
# 6. 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("🎉 欢迎使用 LangGraph 简单示例!")

    # 运行基本示例
    run_simple_example()

    # 运行流式示例
    run_streaming_example()

    # 提供交互选项
    print("\n" + "=" * 60)
    choice = input("是否要进入交互式聊天模式? (y/N): ").strip().lower()

    if choice in ['y', 'yes', '是', 'true']:
        interactive_chat()
    else:
        print("👋 示例结束，感谢使用!")

    print("\n🎓 恭喜！你已经掌握了 LangGraph 的基本用法！")
    print("📚 接下来可以学习更高级的功能：")
    print("   - 04-进阶特性/01-条件路由.md")
    print("   - 04-进阶特性/02-循环结构.md")
    print("   - 04-进阶特性/03-子图与多Agent.md")