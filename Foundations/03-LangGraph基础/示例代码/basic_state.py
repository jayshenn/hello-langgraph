"""
LangGraph 基础状态管理示例

这个文件展示了如何在 LangGraph 中设计和使用状态。
通过运行这些例子，你将理解状态的核心概念。
"""

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

# =============================================================================
# 示例 1: 最简单的状态
# =============================================================================

class SimpleState(TypedDict):
    """最简单的状态定义"""
    message: str
    count: int

def simple_increment(state: SimpleState) -> SimpleState:
    """简单的状态更新"""
    return {
        "message": f"处理消息: {state['message']}",
        "count": state["count"] + 1
    }

def demo_simple_state():
    """演示简单状态的使用"""
    print("=== 简单状态示例 ===")

    # 创建图
    graph = StateGraph(SimpleState)
    graph.add_node("process", simple_increment)
    graph.set_entry_point("process")
    graph.set_finish_point("process")
    app = graph.compile()

    # 运行
    initial_state = {"message": "Hello", "count": 0}
    result = app.invoke(initial_state)

    print(f"初始状态: {initial_state}")
    print(f"最终状态: {result}")
    print()


# =============================================================================
# 示例 2: 复杂状态管理
# =============================================================================

class ChatbotState(TypedDict):
    """聊天机器人的复杂状态"""
    # 用户输入
    user_input: str
    user_id: str

    # 对话管理
    conversation_history: List[str]
    current_step: str

    # 上下文信息
    user_preferences: Dict[str, Any]
    session_metadata: Dict[str, str]

    # 处理状态
    confidence_score: float
    error_count: int
    needs_human_review: bool

def initialize_session(state: ChatbotState) -> ChatbotState:
    """初始化会话"""
    return {
        **state,
        "current_step": "initialized",
        "conversation_history": state.get("conversation_history", []) + [
            f"用户 {state['user_id']} 开始新会话"
        ],
        "error_count": 0,
        "needs_human_review": False
    }

def process_user_input(state: ChatbotState) -> ChatbotState:
    """处理用户输入"""
    user_input = state["user_input"]

    # 模拟情感分析
    confidence = 0.8 if len(user_input) > 10 else 0.5

    # 生成响应
    response = f"我理解您说的'{user_input}'"

    return {
        **state,
        "current_step": "processed",
        "conversation_history": state["conversation_history"] + [
            f"用户: {user_input}",
            f"助手: {response}"
        ],
        "confidence_score": confidence,
        "needs_human_review": confidence < 0.6
    }

def demo_complex_state():
    """演示复杂状态管理"""
    print("=== 复杂状态示例 ===")

    # 创建图
    graph = StateGraph(ChatbotState)
    graph.add_node("init", initialize_session)
    graph.add_node("process", process_user_input)

    graph.set_entry_point("init")
    graph.add_edge("init", "process")
    graph.set_finish_point("process")

    app = graph.compile()

    # 测试数据
    initial_state = {
        "user_input": "你好",
        "user_id": "user123",
        "conversation_history": [],
        "current_step": "start",
        "user_preferences": {"language": "zh"},
        "session_metadata": {"timestamp": "2024-01-01"},
        "confidence_score": 0.0,
        "error_count": 0,
        "needs_human_review": False
    }

    result = app.invoke(initial_state)

    print(f"用户输入: {initial_state['user_input']}")
    print(f"处理步骤: {result['current_step']}")
    print(f"置信度: {result['confidence_score']}")
    print(f"对话历史: {result['conversation_history']}")
    print(f"需要人工审核: {result['needs_human_review']}")
    print()


# =============================================================================
# 示例 3: 状态累积和更新模式
# =============================================================================

class AccumulativeState(TypedDict):
    """累积式状态"""
    items: List[str]
    total_count: int
    processing_log: List[str]

def add_item(state: AccumulativeState) -> AccumulativeState:
    """添加项目到状态"""
    new_item = f"item_{len(state['items']) + 1}"

    return {
        **state,
        "items": state["items"] + [new_item],
        "total_count": state["total_count"] + 1,
        "processing_log": state["processing_log"] + [f"添加了 {new_item}"]
    }

def validate_items(state: AccumulativeState) -> AccumulativeState:
    """验证项目"""
    valid_items = [item for item in state["items"] if "item_" in item]

    return {
        **state,
        "items": valid_items,
        "total_count": len(valid_items),
        "processing_log": state["processing_log"] + [f"验证完成，有效项目: {len(valid_items)}"]
    }

def demo_accumulative_state():
    """演示累积状态"""
    print("=== 累积状态示例 ===")

    # 创建图
    graph = StateGraph(AccumulativeState)
    graph.add_node("add1", add_item)
    graph.add_node("add2", add_item)
    graph.add_node("add3", add_item)
    graph.add_node("validate", validate_items)

    graph.set_entry_point("add1")
    graph.add_edge("add1", "add2")
    graph.add_edge("add2", "add3")
    graph.add_edge("add3", "validate")
    graph.set_finish_point("validate")

    app = graph.compile()

    # 运行
    initial_state = {
        "items": [],
        "total_count": 0,
        "processing_log": ["开始处理"]
    }

    result = app.invoke(initial_state)

    print(f"最终项目: {result['items']}")
    print(f"项目总数: {result['total_count']}")
    print("处理日志:")
    for log in result["processing_log"]:
        print(f"  - {log}")
    print()


# =============================================================================
# 示例 4: 状态验证和错误处理
# =============================================================================

class ValidatedState(TypedDict):
    """带验证的状态"""
    data: Dict[str, Any]
    validation_errors: List[str]
    is_valid: bool
    processed: bool

def validate_data(state: ValidatedState) -> ValidatedState:
    """验证数据"""
    data = state["data"]
    errors = []

    # 验证规则
    if "name" not in data:
        errors.append("缺少必需字段: name")

    if "age" in data and not isinstance(data["age"], int):
        errors.append("age 必须是整数")

    if "email" in data and "@" not in data["email"]:
        errors.append("email 格式无效")

    return {
        **state,
        "validation_errors": errors,
        "is_valid": len(errors) == 0
    }

def process_valid_data(state: ValidatedState) -> ValidatedState:
    """处理有效数据"""
    if not state["is_valid"]:
        return state

    # 模拟数据处理
    processed_data = {
        **state["data"],
        "processed_at": "2024-01-01T12:00:00"
    }

    return {
        **state,
        "data": processed_data,
        "processed": True
    }

def handle_invalid_data(state: ValidatedState) -> ValidatedState:
    """处理无效数据"""
    return {
        **state,
        "processed": False,
        "data": {**state["data"], "status": "invalid"}
    }

def is_data_valid(state: ValidatedState) -> str:
    """路由函数：根据验证结果决定下一步"""
    return "process" if state["is_valid"] else "handle_error"

def demo_validated_state():
    """演示状态验证"""
    print("=== 状态验证示例 ===")

    # 创建图
    graph = StateGraph(ValidatedState)
    graph.add_node("validate", validate_data)
    graph.add_node("process", process_valid_data)
    graph.add_node("handle_error", handle_invalid_data)

    graph.set_entry_point("validate")
    graph.add_conditional_edges(
        "validate",
        is_data_valid,
        {
            "process": "process",
            "handle_error": "handle_error"
        }
    )
    graph.add_edge("process", END)
    graph.add_edge("handle_error", END)

    app = graph.compile()

    # 测试有效数据
    valid_data = {
        "data": {"name": "Alice", "age": 25, "email": "alice@example.com"},
        "validation_errors": [],
        "is_valid": False,
        "processed": False
    }

    result = app.invoke(valid_data)
    print("有效数据测试:")
    print(f"  验证结果: {result['is_valid']}")
    print(f"  处理状态: {result['processed']}")
    print(f"  最终数据: {result['data']}")

    # 测试无效数据
    invalid_data = {
        "data": {"age": "not_a_number", "email": "invalid_email"},
        "validation_errors": [],
        "is_valid": False,
        "processed": False
    }

    result = app.invoke(invalid_data)
    print("\n无效数据测试:")
    print(f"  验证结果: {result['is_valid']}")
    print(f"  验证错误: {result['validation_errors']}")
    print(f"  处理状态: {result['processed']}")
    print()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("🚀 LangGraph 状态管理示例\n")

    demo_simple_state()
    demo_complex_state()
    demo_accumulative_state()
    demo_validated_state()

    print("✅ 所有示例运行完成！")
    print("\n💡 关键要点:")
    print("1. 状态是 TypedDict，提供类型安全")
    print("2. 节点函数接收状态，返回新状态")
    print("3. 使用 **state 语法保持其他字段不变")
    print("4. 状态可以包含复杂的数据结构")
    print("5. 验证状态有助于调试和错误处理")


if __name__ == "__main__":
    main()


# =============================================================================
# 练习建议
# =============================================================================

"""
🎯 练习建议：

1. 修改 SimpleState，添加一个 timestamp 字段
2. 在 ChatbotState 中添加用户情感分析
3. 创建一个新的状态类型来管理购物车
4. 实现一个状态历史追踪功能
5. 添加状态序列化和反序列化功能

💡 思考问题：

1. 什么时候应该使用 **state 语法？
2. 如何处理状态中的嵌套数据结构？
3. 状态过大时如何优化性能？
4. 如何设计可扩展的状态结构？
"""