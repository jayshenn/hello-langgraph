"""
基础聊天机器人 - LangGraph 入门项目

这是一个简单但完整的聊天机器人实现，展示了 LangGraph 的核心概念：
- 状态管理
- 图结构设计
- 条件路由
- 多轮对话

适合初学者理解 LangGraph 的基本工作原理。
"""

from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
import re
import random

# =============================================================================
# 状态定义
# =============================================================================

class ChatbotState(TypedDict):
    """聊天机器人的状态定义"""
    # 用户输入
    user_input: str
    user_id: str

    # 对话管理
    conversation_history: List[str]
    current_intent: Literal["greeting", "order_inquiry", "technical_support", "general", "goodbye"]

    # 系统状态
    step_count: int
    confidence_score: float
    needs_human_help: bool

    # 输出
    bot_response: str


# =============================================================================
# 工具函数
# =============================================================================

def analyze_intent(user_input: str) -> tuple[str, float]:
    """
    简单的意图识别（基于关键词匹配）
    在实际应用中，这里会使用机器学习模型
    """
    user_input_lower = user_input.lower()

    # 问候意图
    greeting_keywords = ["你好", "hello", "hi", "早上好", "下午好", "晚上好"]
    if any(keyword in user_input_lower for keyword in greeting_keywords):
        return "greeting", 0.9

    # 再见意图
    goodbye_keywords = ["再见", "bye", "goodbye", "拜拜", "结束"]
    if any(keyword in user_input_lower for keyword in goodbye_keywords):
        return "goodbye", 0.9

    # 订单查询意图
    order_keywords = ["订单", "order", "查询", "状态", "物流", "发货"]
    if any(keyword in user_input_lower for keyword in order_keywords):
        return "order_inquiry", 0.8

    # 技术支持意图
    tech_keywords = ["问题", "故障", "bug", "技术", "支持", "帮助", "不能用", "错误"]
    if any(keyword in user_input_lower for keyword in tech_keywords):
        return "technical_support", 0.7

    # 默认为一般对话
    return "general", 0.5


def mock_order_lookup(user_input: str) -> str:
    """模拟订单查询"""
    # 提取可能的订单号
    order_pattern = r'\d{6,}'
    order_numbers = re.findall(order_pattern, user_input)

    if order_numbers:
        order_num = order_numbers[0]
        # 模拟订单状态
        statuses = ["已下单", "已发货", "运输中", "已送达"]
        status = random.choice(statuses)
        return f"您的订单 {order_num} 当前状态是：{status}。预计3-5个工作日送达。"
    else:
        return "请提供您的订单号，格式如：123456789"


def mock_tech_support(user_input: str) -> str:
    """模拟技术支持"""
    common_solutions = [
        "请尝试重启应用程序，这通常能解决大部分问题。",
        "请检查您的网络连接是否正常。",
        "请清除浏览器缓存后重试。",
        "请确保您使用的是最新版本的应用。"
    ]

    if "登录" in user_input or "login" in user_input.lower():
        return "登录问题通常是由密码错误引起的，请尝试重置密码。"
    elif "慢" in user_input or "卡" in user_input:
        return "系统响应慢可能是网络问题，请检查网络连接或稍后重试。"
    else:
        return random.choice(common_solutions)


# =============================================================================
# 节点函数
# =============================================================================

def intent_analysis_node(state: ChatbotState) -> ChatbotState:
    """意图分析节点"""
    user_input = state["user_input"]
    intent, confidence = analyze_intent(user_input)

    # 更新对话历史
    new_history = state["conversation_history"] + [f"用户: {user_input}"]

    return {
        **state,
        "current_intent": intent,
        "confidence_score": confidence,
        "conversation_history": new_history,
        "step_count": state["step_count"] + 1,
        "needs_human_help": confidence < 0.6  # 低置信度需要人工帮助
    }


def greeting_handler(state: ChatbotState) -> ChatbotState:
    """处理问候"""
    greetings = [
        f"您好！我是智能客服小助手，很高兴为您服务！",
        f"您好！有什么可以帮助您的吗？",
        f"欢迎！我可以帮您查询订单、解决技术问题等。"
    ]

    response = random.choice(greetings)

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"]
    }


def order_inquiry_handler(state: ChatbotState) -> ChatbotState:
    """处理订单查询"""
    user_input = state["user_input"]
    response = mock_order_lookup(user_input)

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"]
    }


def technical_support_handler(state: ChatbotState) -> ChatbotState:
    """处理技术支持"""
    user_input = state["user_input"]
    response = mock_tech_support(user_input)

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"]
    }


def general_chat_handler(state: ChatbotState) -> ChatbotState:
    """处理一般对话"""
    responses = [
        "我理解您的问题，但我主要专长于订单查询和技术支持。您可以描述具体需要什么帮助吗？",
        "我是客服机器人，主要处理订单和技术问题。请告诉我您遇到了什么具体困难？",
        "如果您有订单问题或技术困难，我很乐意帮助。请详细描述您的情况。"
    ]

    response = random.choice(responses)

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"]
    }


def goodbye_handler(state: ChatbotState) -> ChatbotState:
    """处理告别"""
    responses = [
        "感谢您的咨询，祝您生活愉快！",
        "再见！如果还有问题欢迎随时咨询。",
        "谢谢，有需要请随时联系我们！"
    ]

    response = random.choice(responses)

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"]
    }


def human_escalation_handler(state: ChatbotState) -> ChatbotState:
    """转人工客服"""
    response = "我将为您转接人工客服，请稍等片刻。客服代表马上为您服务。"

    return {
        **state,
        "bot_response": response,
        "conversation_history": state["conversation_history"] + [f"客服: {response}"],
        "needs_human_help": True
    }


# =============================================================================
# 路由函数
# =============================================================================

def intent_router(state: ChatbotState) -> str:
    """根据意图决定下一个节点"""
    # 低置信度直接转人工
    if state["needs_human_help"]:
        return "human_escalation"

    # 根据意图路由
    intent_mapping = {
        "greeting": "greeting_handler",
        "order_inquiry": "order_handler",
        "technical_support": "tech_handler",
        "general": "general_handler",
        "goodbye": "goodbye_handler"
    }

    return intent_mapping.get(state["current_intent"], "general_handler")


# =============================================================================
# 图构建
# =============================================================================

def create_chatbot_graph() -> StateGraph:
    """创建聊天机器人图"""
    # 创建图
    graph = StateGraph(ChatbotState)

    # 添加节点
    graph.add_node("intent_analysis", intent_analysis_node)
    graph.add_node("greeting_handler", greeting_handler)
    graph.add_node("order_handler", order_inquiry_handler)
    graph.add_node("tech_handler", technical_support_handler)
    graph.add_node("general_handler", general_chat_handler)
    graph.add_node("goodbye_handler", goodbye_handler)
    graph.add_node("human_escalation", human_escalation_handler)

    # 设置入口
    graph.set_entry_point("intent_analysis")

    # 添加条件路由
    graph.add_conditional_edges(
        "intent_analysis",
        intent_router,
        {
            "greeting_handler": "greeting_handler",
            "order_handler": "order_handler",
            "tech_handler": "tech_handler",
            "general_handler": "general_handler",
            "goodbye_handler": "goodbye_handler",
            "human_escalation": "human_escalation"
        }
    )

    # 所有处理节点都结束到 END
    graph.add_edge("greeting_handler", END)
    graph.add_edge("order_handler", END)
    graph.add_edge("tech_handler", END)
    graph.add_edge("general_handler", END)
    graph.add_edge("goodbye_handler", END)
    graph.add_edge("human_escalation", END)

    return graph


# =============================================================================
# 主要功能
# =============================================================================

class SimpleChatbot:
    """简单聊天机器人类"""

    def __init__(self):
        """初始化聊天机器人"""
        self.graph = create_chatbot_graph()
        self.app = self.graph.compile()
        self.session_count = 0

    def chat(self, user_input: str, user_id: str = "default_user") -> Dict[str, Any]:
        """处理单次对话"""
        # 创建初始状态
        initial_state: ChatbotState = {
            "user_input": user_input,
            "user_id": user_id,
            "conversation_history": [],
            "current_intent": "general",
            "step_count": 0,
            "confidence_score": 0.0,
            "needs_human_help": False,
            "bot_response": ""
        }

        # 执行图
        result = self.app.invoke(initial_state)

        # 返回结果
        return {
            "response": result["bot_response"],
            "intent": result["current_intent"],
            "confidence": result["confidence_score"],
            "needs_human_help": result["needs_human_help"],
            "conversation_history": result["conversation_history"]
        }

    def interactive_chat(self):
        """交互式聊天模式"""
        print("🤖 欢迎使用智能客服！输入 'quit' 退出。\n")

        conversation_history = []
        user_id = f"user_{random.randint(1000, 9999)}"

        while True:
            try:
                # 获取用户输入
                user_input = input("您: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if not user_input:
                    print("请输入您的问题。")
                    continue

                # 创建状态（包含历史对话）
                state: ChatbotState = {
                    "user_input": user_input,
                    "user_id": user_id,
                    "conversation_history": conversation_history,
                    "current_intent": "general",
                    "step_count": len(conversation_history),
                    "confidence_score": 0.0,
                    "needs_human_help": False,
                    "bot_response": ""
                }

                # 执行对话
                result = self.app.invoke(state)

                # 更新历史
                conversation_history = result["conversation_history"]

                # 显示结果
                print(f"🤖 {result['bot_response']}")

                # 显示调试信息（可选）
                if result["confidence_score"] < 0.7:
                    print(f"💡 [置信度: {result['confidence_score']:.2f}, 意图: {result['current_intent']}]")

                print()  # 空行分隔

            except KeyboardInterrupt:
                print("\n👋 对话结束！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")


def demo_single_interactions():
    """演示单次交互"""
    print("=== 单次交互演示 ===\n")

    chatbot = SimpleChatbot()

    # 测试用例
    test_cases = [
        "你好",
        "我想查询订单123456789",
        "登录不了怎么办",
        "天气怎么样",
        "再见"
    ]

    for i, user_input in enumerate(test_cases, 1):
        print(f"测试 {i}: {user_input}")
        result = chatbot.chat(user_input)

        print(f"回复: {result['response']}")
        print(f"意图: {result['intent']}, 置信度: {result['confidence']:.2f}")
        print(f"需要人工: {result['needs_human_help']}")
        print("-" * 50)


def visualize_graph():
    """可视化图结构（如果可能的话）"""
    try:
        chatbot = SimpleChatbot()
        # 尝试生成图的可视化
        print("图结构:")
        print("intent_analysis → [条件路由] → 各种处理节点 → END")
        print("\n节点列表:")
        for node in chatbot.graph.nodes:
            print(f"  - {node}")

    except Exception as e:
        print(f"无法生成可视化: {e}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("🚀 基础聊天机器人 - LangGraph 示例")
    print("=" * 50)

    while True:
        print("\n选择模式:")
        print("1. 交互式聊天")
        print("2. 演示模式")
        print("3. 图结构查看")
        print("4. 退出")

        choice = input("\n请选择 (1-4): ").strip()

        if choice == "1":
            chatbot = SimpleChatbot()
            chatbot.interactive_chat()

        elif choice == "2":
            demo_single_interactions()

        elif choice == "3":
            visualize_graph()

        elif choice == "4":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选择，请重试。")


if __name__ == "__main__":
    main()


# =============================================================================
# 学习要点和练习建议
# =============================================================================

"""
🎯 学习要点:

1. **状态设计**: ChatbotState 包含了对话的所有必要信息
2. **节点职责**: 每个节点有单一、明确的职责
3. **条件路由**: intent_router 根据意图决定下一步
4. **状态更新**: 使用 **state 语法保持状态一致性
5. **错误处理**: 低置信度自动转人工客服

🔧 练习建议:

1. **添加新意图**: 实现"退款申请"意图和处理节点
2. **改进意图识别**: 使用更复杂的规则或模型
3. **添加状态验证**: 验证用户输入的有效性
4. **实现会话持久化**: 将对话历史保存到文件
5. **添加情感分析**: 识别用户情绪并相应调整回复

💡 扩展方向:

1. 集成真实的 LLM API (OpenAI, Claude 等)
2. 添加数据库支持存储用户信息
3. 实现多轮对话的上下文理解
4. 添加语音识别和语音合成
5. 创建 Web 界面或 API 服务

这个基础版本为你提供了 LangGraph 的核心概念实践。
理解了这些基础后，就可以构建更复杂的应用了！
"""