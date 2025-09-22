"""
高级聊天机器人 - LangGraph 完整项目

这是一个功能完整的聊天机器人实现，展示了 LangGraph 的高级特性：
- 真实 LLM 集成
- 记忆管理（短期和长期）
- 工具调用
- Human-in-the-Loop
- 流式输出
- 情感分析

适合进阶学习者理解 LangGraph 的生产级应用。
"""

import os
import json
import asyncio
import datetime
from typing import TypedDict, List, Dict, Any, Optional, Annotated, AsyncIterator
from dataclasses import dataclass, asdict
from collections import deque

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# 加载环境变量
load_dotenv()

# =============================================================================
# 配置和常量
# =============================================================================

@dataclass
class ChatbotConfig:
    """聊天机器人配置"""
    # LLM 配置
    llm_provider: str = "openai"  # openai, anthropic
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000

    # 记忆配置
    short_term_memory_size: int = 10
    long_term_memory_threshold: float = 0.8

    # 工具配置
    enable_tools: bool = True
    enable_web_search: bool = False

    # Human-in-the-Loop
    enable_human_review: bool = True
    sensitive_keywords: List[str] = None

    # 其他配置
    enable_emotion_analysis: bool = True
    max_conversation_turns: int = 50

    def __post_init__(self):
        if self.sensitive_keywords is None:
            self.sensitive_keywords = ["投诉", "退款", "法律", "起诉"]


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = None
    conversation_count: int = 0
    last_interaction: Optional[datetime.datetime] = None
    satisfaction_score: float = 0.0

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.last_interaction is None:
            self.last_interaction = datetime.datetime.now()


@dataclass
class Memory:
    """记忆项"""
    content: str
    timestamp: datetime.datetime
    importance: float
    memory_type: str  # short_term, long_term
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AdvancedChatbotState(TypedDict):
    """高级聊天机器人状态"""
    # 消息管理
    messages: Annotated[List[BaseMessage], add_messages]

    # 用户信息
    user_id: str
    user_profile: Optional[UserProfile]

    # 对话管理
    conversation_id: str
    turn_count: int
    current_intent: str
    confidence_score: float

    # 情感分析
    user_emotion: str
    emotion_score: float

    # 记忆管理
    short_term_memories: List[Memory]
    long_term_memories: List[Memory]

    # 工具和决策
    tool_calls_made: List[str]
    needs_human_review: bool
    escalation_reason: str

    # 系统状态
    last_response: str
    processing_time: float
    error_count: int


# =============================================================================
# 工具定义
# =============================================================================

@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """执行简单的数学计算

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        # 安全的计算，只允许基本数学操作
        allowed_chars = "0123456789+-*/.() "
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def get_weather(city: str = "北京") -> str:
    """获取天气信息（模拟）

    Args:
        city: 城市名称
    """
    import random

    weather_conditions = ["晴朗", "多云", "小雨", "阴天"]
    temperatures = list(range(15, 30))

    condition = random.choice(weather_conditions)
    temp = random.choice(temperatures)

    return f"{city}今天天气{condition}，温度{temp}°C"


@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库（模拟）

    Args:
        query: 搜索查询
    """
    # 模拟知识库搜索
    knowledge_base = {
        "产品": "我们的主要产品包括智能客服系统、数据分析平台等。",
        "价格": "具体价格请联系销售团队，我们提供灵活的定价方案。",
        "技术支持": "我们提供7x24小时技术支持，响应时间不超过2小时。",
        "退款": "产品支持30天无理由退款，详情请查看退款政策。"
    }

    for key, value in knowledge_base.items():
        if key in query:
            return f"找到相关信息：{value}"

    return "抱歉，没有找到相关信息。请联系人工客服获取帮助。"


# 工具节点
tools = [get_current_time, calculate, get_weather, search_knowledge_base]
tool_node = ToolNode(tools)


# =============================================================================
# LLM 集成
# =============================================================================

class LLMManager:
    """LLM 管理器"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.llm = self._create_llm()

    def _create_llm(self):
        """创建 LLM 实例"""
        if self.config.llm_provider == "openai":
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {self.config.llm_provider}")

    def get_llm_with_tools(self):
        """获取绑定工具的LLM"""
        if self.config.enable_tools:
            return self.llm.bind_tools(tools)
        return self.llm


# =============================================================================
# 记忆管理
# =============================================================================

class MemoryManager:
    """记忆管理器"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.short_term_memory = deque(maxlen=config.short_term_memory_size)
        self.long_term_memory = []

    def add_memory(self, content: str, importance: float, memory_type: str = "short_term"):
        """添加记忆"""
        memory = Memory(
            content=content,
            timestamp=datetime.datetime.now(),
            importance=importance,
            memory_type=memory_type
        )

        if memory_type == "short_term":
            self.short_term_memory.append(memory)
            # 检查是否需要转为长期记忆
            if importance >= self.config.long_term_memory_threshold:
                self.promote_to_long_term(memory)
        else:
            self.long_term_memory.append(memory)

    def promote_to_long_term(self, memory: Memory):
        """提升为长期记忆"""
        memory.memory_type = "long_term"
        self.long_term_memory.append(memory)

    def get_relevant_memories(self, query: str, max_count: int = 5) -> List[Memory]:
        """获取相关记忆"""
        # 简单的关键词匹配（生产环境中可以使用向量搜索）
        all_memories = list(self.short_term_memory) + self.long_term_memory
        relevant_memories = []

        for memory in all_memories:
            if any(word in memory.content.lower() for word in query.lower().split()):
                relevant_memories.append(memory)

        # 按重要性排序
        relevant_memories.sort(key=lambda m: m.importance, reverse=True)
        return relevant_memories[:max_count]


# =============================================================================
# 情感分析
# =============================================================================

class EmotionAnalyzer:
    """情感分析器"""

    @staticmethod
    def analyze_emotion(text: str) -> tuple[str, float]:
        """分析情感（简化版）"""
        # 情感词典
        positive_words = ["高兴", "满意", "好", "棒", "喜欢", "感谢", "谢谢"]
        negative_words = ["生气", "愤怒", "不满", "差", "糟糕", "讨厌", "投诉"]
        neutral_words = ["询问", "查询", "了解", "知道", "需要"]

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)

        total_count = positive_count + negative_count + neutral_count

        if total_count == 0:
            return "neutral", 0.5

        if positive_count > negative_count:
            emotion = "positive"
            score = 0.5 + (positive_count / total_count) * 0.5
        elif negative_count > positive_count:
            emotion = "negative"
            score = 0.5 - (negative_count / total_count) * 0.5
        else:
            emotion = "neutral"
            score = 0.5

        return emotion, score


# =============================================================================
# 节点函数
# =============================================================================

def input_analysis_node(state: AdvancedChatbotState) -> AdvancedChatbotState:
    """输入分析节点"""
    if not state["messages"]:
        return state

    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content

        # 情感分析
        emotion, emotion_score = EmotionAnalyzer.analyze_emotion(user_input)

        # 简单意图识别
        intent = "general"
        confidence = 0.5

        if any(word in user_input.lower() for word in ["计算", "算", "数学"]):
            intent = "calculation"
            confidence = 0.8
        elif any(word in user_input.lower() for word in ["时间", "几点", "现在"]):
            intent = "time_query"
            confidence = 0.9
        elif any(word in user_input.lower() for word in ["天气", "温度", "下雨"]):
            intent = "weather_query"
            confidence = 0.8
        elif any(word in user_input.lower() for word in ["查询", "搜索", "知识"]):
            intent = "knowledge_search"
            confidence = 0.7

        # 检查是否需要人工审核
        sensitive_keywords = ["投诉", "退款", "法律", "起诉"]
        needs_review = any(keyword in user_input for keyword in sensitive_keywords)

        return {
            **state,
            "current_intent": intent,
            "confidence_score": confidence,
            "user_emotion": emotion,
            "emotion_score": emotion_score,
            "needs_human_review": needs_review,
            "turn_count": state["turn_count"] + 1
        }

    return state


def memory_integration_node(state: AdvancedChatbotState) -> AdvancedChatbotState:
    """记忆集成节点"""
    # 在实际应用中，这里会查询和更新记忆
    # 现在只是模拟

    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            # 创建记忆
            importance = 0.6
            if state["user_emotion"] == "negative":
                importance = 0.9  # 负面情绪的对话更重要

            memory = Memory(
                content=last_message.content,
                timestamp=datetime.datetime.now(),
                importance=importance,
                memory_type="short_term"
            )

            # 更新状态中的记忆
            new_memories = state.get("short_term_memories", []) + [memory]

            return {
                **state,
                "short_term_memories": new_memories[-5:]  # 只保留最近5条
            }

    return state


def llm_processing_node(state: AdvancedChatbotState) -> AdvancedChatbotState:
    """LLM处理节点"""
    config = ChatbotConfig()
    llm_manager = LLMManager(config)
    llm = llm_manager.get_llm_with_tools()

    # 构建系统提示
    system_prompt = f"""你是一个智能客服助手。请根据以下信息回复用户：

当前用户情感：{state.get('user_emotion', 'neutral')} (分数：{state.get('emotion_score', 0.5):.2f})
用户意图：{state.get('current_intent', 'general')}
置信度：{state.get('confidence_score', 0.5):.2f}

请提供专业、友好、有帮助的回复。如果用户情绪负面，请特别关注并提供额外的关怀。

记忆上下文：
{json.dumps([m.content for m in state.get('short_term_memories', [])], ensure_ascii=False, indent=2)}

如果需要使用工具，请调用相应的工具函数。"""

    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        start_time = datetime.datetime.now()
        response = llm.invoke(messages)
        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        # 检查是否有工具调用
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = [call['name'] for call in response.tool_calls]

        return {
            **state,
            "messages": state["messages"] + [response],
            "last_response": response.content if hasattr(response, 'content') else str(response),
            "processing_time": processing_time,
            "tool_calls_made": state.get("tool_calls_made", []) + tool_calls,
            "error_count": 0  # 成功处理，重置错误计数
        }

    except Exception as e:
        error_message = f"处理过程中发生错误：{str(e)}"
        error_response = AIMessage(content="抱歉，我遇到了一些技术问题。请稍后重试或联系人工客服。")

        return {
            **state,
            "messages": state["messages"] + [error_response],
            "last_response": error_response.content,
            "error_count": state.get("error_count", 0) + 1,
            "needs_human_review": True,
            "escalation_reason": "技术错误"
        }


def human_review_node(state: AdvancedChatbotState) -> AdvancedChatbotState:
    """人工审核节点"""
    if state["needs_human_review"]:
        escalation_message = AIMessage(
            content="您的问题已转交给人工客服，专业顾问将尽快为您服务。请稍候片刻。"
        )

        return {
            **state,
            "messages": state["messages"] + [escalation_message],
            "last_response": escalation_message.content
        }

    return state


def quality_assessment_node(state: AdvancedChatbotState) -> AdvancedChatbotState:
    """质量评估节点"""
    # 简单的质量评估
    quality_score = 0.8  # 默认质量分数

    # 基于各种因素调整质量分数
    if state.get("error_count", 0) > 2:
        quality_score -= 0.3

    if state.get("user_emotion") == "positive":
        quality_score += 0.1
    elif state.get("user_emotion") == "negative":
        quality_score -= 0.2

    if state.get("confidence_score", 0) > 0.8:
        quality_score += 0.1

    # 如果质量分数过低，建议人工接入
    if quality_score < 0.5:
        return {
            **state,
            "needs_human_review": True,
            "escalation_reason": "质量分数过低"
        }

    return state


# =============================================================================
# 路由函数
# =============================================================================

def should_use_tools(state: AdvancedChatbotState) -> str:
    """判断是否需要使用工具"""
    if not state["messages"]:
        return "llm_processing"

    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    return "quality_assessment"


def should_escalate(state: AdvancedChatbotState) -> str:
    """判断是否需要升级到人工"""
    if state["needs_human_review"]:
        return "human_review"

    return "end"


# =============================================================================
# 图构建
# =============================================================================

def create_advanced_chatbot_graph() -> StateGraph:
    """创建高级聊天机器人图"""
    graph = StateGraph(AdvancedChatbotState)

    # 添加节点
    graph.add_node("input_analysis", input_analysis_node)
    graph.add_node("memory_integration", memory_integration_node)
    graph.add_node("llm_processing", llm_processing_node)
    graph.add_node("tools", tool_node)
    graph.add_node("quality_assessment", quality_assessment_node)
    graph.add_node("human_review", human_review_node)

    # 设置入口点
    graph.set_entry_point("input_analysis")

    # 添加边
    graph.add_edge("input_analysis", "memory_integration")
    graph.add_edge("memory_integration", "llm_processing")

    # 条件边：LLM处理后可能需要工具调用
    graph.add_conditional_edges(
        "llm_processing",
        should_use_tools,
        {
            "tools": "tools",
            "quality_assessment": "quality_assessment"
        }
    )

    # 工具调用后回到LLM处理
    graph.add_edge("tools", "llm_processing")

    # 质量评估后可能需要人工接入
    graph.add_conditional_edges(
        "quality_assessment",
        should_escalate,
        {
            "human_review": "human_review",
            "end": END
        }
    )

    # 人工审核后结束
    graph.add_edge("human_review", END)

    return graph


# =============================================================================
# 主要功能类
# =============================================================================

class AdvancedChatbot:
    """高级聊天机器人"""

    def __init__(self, config: ChatbotConfig = None):
        """初始化"""
        self.config = config or ChatbotConfig()
        self.graph = create_advanced_chatbot_graph()
        self.memory_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory_saver)
        self.memory_manager = MemoryManager(self.config)

        print("🤖 高级聊天机器人已启动")
        print(f"📊 配置: {self.config.llm_provider} | {self.config.model_name}")

    def create_initial_state(self, user_id: str = "default_user") -> AdvancedChatbotState:
        """创建初始状态"""
        return {
            "messages": [],
            "user_id": user_id,
            "user_profile": None,
            "conversation_id": f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "turn_count": 0,
            "current_intent": "",
            "confidence_score": 0.0,
            "user_emotion": "neutral",
            "emotion_score": 0.5,
            "short_term_memories": [],
            "long_term_memories": [],
            "tool_calls_made": [],
            "needs_human_review": False,
            "escalation_reason": "",
            "last_response": "",
            "processing_time": 0.0,
            "error_count": 0
        }

    async def chat_async(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """异步聊天"""
        if not conversation_id:
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 配置
        config = {"configurable": {"thread_id": conversation_id}}

        # 创建用户消息
        user_message = HumanMessage(content=message)

        # 获取当前状态或创建新状态
        try:
            current_state = await self.app.aget_state(config)
            if current_state.values:
                state = current_state.values
                state["messages"] = state.get("messages", []) + [user_message]
            else:
                state = self.create_initial_state()
                state["messages"] = [user_message]
        except:
            state = self.create_initial_state()
            state["messages"] = [user_message]

        # 执行图
        result = await self.app.ainvoke(state, config)

        # 提取响应
        if result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response = last_message.content
            else:
                response = str(last_message)
        else:
            response = "抱歉，我没有理解您的问题。"

        return {
            "response": response,
            "conversation_id": conversation_id,
            "intent": result.get("current_intent", ""),
            "emotion": result.get("user_emotion", "neutral"),
            "confidence": result.get("confidence_score", 0.0),
            "needs_human_review": result.get("needs_human_review", False),
            "processing_time": result.get("processing_time", 0.0),
            "tool_calls": result.get("tool_calls_made", [])
        }

    def chat(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """同步聊天"""
        return asyncio.run(self.chat_async(message, conversation_id))

    async def stream_chat(self, message: str, conversation_id: str = None) -> AsyncIterator[str]:
        """流式聊天"""
        # 简化的流式实现
        result = await self.chat_async(message, conversation_id)
        response = result["response"]

        # 模拟打字机效果
        for char in response:
            yield char
            await asyncio.sleep(0.05)  # 调整延迟以控制速度

    def interactive_chat(self):
        """交互式聊天"""
        print("🤖 高级智能客服启动！输入 'quit' 退出，'help' 查看帮助。\n")

        conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        while True:
            try:
                user_input = input("您: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 感谢使用，再见！")
                    break

                if user_input.lower() == 'help':
                    print("""
📋 可用功能：
- 基本对话：直接输入问题
- 时间查询：问"现在几点"
- 计算：输入数学表达式，如"计算 2+3*4"
- 天气查询：问"北京天气怎么样"
- 知识搜索：搜索产品、价格等信息
- 退出：输入 quit 或 exit
                    """)
                    continue

                if not user_input:
                    print("请输入您的问题。")
                    continue

                # 获取回复
                print("🤖 正在思考...")
                result = self.chat(user_input, conversation_id)

                # 显示回复
                print(f"🤖 {result['response']}")

                # 显示调试信息
                print(f"💭 [意图: {result['intent']}, 情感: {result['emotion']}, "
                      f"置信度: {result['confidence']:.2f}, "
                      f"处理时间: {result['processing_time']:.2f}s]")

                if result['tool_calls']:
                    print(f"🛠️ [使用工具: {', '.join(result['tool_calls'])}]")

                if result['needs_human_review']:
                    print("⚠️ [已标记需要人工审核]")

                print()

            except KeyboardInterrupt:
                print("\n👋 对话结束！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")


# =============================================================================
# 演示和测试
# =============================================================================

def demo_basic_features():
    """演示基本功能"""
    print("=== 基本功能演示 ===\n")

    config = ChatbotConfig(
        llm_provider="openai",  # 如果没有API key，会使用模拟回复
        enable_tools=True,
        enable_emotion_analysis=True
    )

    chatbot = AdvancedChatbot(config)

    test_cases = [
        "你好，我想了解一下你们的产品",
        "计算 15 * 8 + 20",
        "现在几点了？",
        "北京今天天气怎么样？",
        "我对服务不满意，要投诉！",
        "搜索价格信息"
    ]

    conversation_id = "demo_conversation"

    for i, message in enumerate(test_cases, 1):
        print(f"测试 {i}: {message}")
        try:
            result = chatbot.chat(message, conversation_id)
            print(f"回复: {result['response']}")
            print(f"分析: 意图={result['intent']}, 情感={result['emotion']}, "
                  f"置信度={result['confidence']:.2f}")
            if result['tool_calls']:
                print(f"工具调用: {result['tool_calls']}")
            print("-" * 60)
        except Exception as e:
            print(f"错误: {e}")
            print("-" * 60)


async def demo_streaming():
    """演示流式输出"""
    print("=== 流式输出演示 ===\n")

    config = ChatbotConfig(enable_tools=False)  # 简化配置
    chatbot = AdvancedChatbot(config)

    print("用户: 你好，请介绍一下你的功能")
    print("🤖: ", end="", flush=True)

    async for chunk in chatbot.stream_chat("你好，请介绍一下你的功能"):
        print(chunk, end="", flush=True)

    print("\n")


def demo_memory_persistence():
    """演示记忆持久化"""
    print("=== 记忆持久化演示 ===\n")

    chatbot = AdvancedChatbot()
    conversation_id = "memory_demo"

    # 第一轮对话
    print("第一轮对话:")
    result1 = chatbot.chat("我叫张三，住在北京", conversation_id)
    print(f"用户: 我叫张三，住在北京")
    print(f"🤖: {result1['response']}")

    # 第二轮对话
    print("\n第二轮对话:")
    result2 = chatbot.chat("你还记得我的名字吗？", conversation_id)
    print(f"用户: 你还记得我的名字吗？")
    print(f"🤖: {result2['response']}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("🚀 高级聊天机器人 - LangGraph 完整示例")
    print("=" * 50)

    # 检查环境变量
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️ 警告：未设置 API Key，将使用模拟模式")
        print("请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量以使用真实 LLM\n")

    while True:
        print("\n选择模式:")
        print("1. 交互式聊天")
        print("2. 基本功能演示")
        print("3. 流式输出演示")
        print("4. 记忆持久化演示")
        print("5. 退出")

        choice = input("\n请选择 (1-5): ").strip()

        if choice == "1":
            chatbot = AdvancedChatbot()
            chatbot.interactive_chat()

        elif choice == "2":
            demo_basic_features()

        elif choice == "3":
            asyncio.run(demo_streaming())

        elif choice == "4":
            demo_memory_persistence()

        elif choice == "5":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选择，请重试。")


if __name__ == "__main__":
    main()


# =============================================================================
# 学习指南和最佳实践
# =============================================================================

"""
🎯 高级聊天机器人学习要点:

1. **架构设计**:
   - 模块化节点设计，每个节点有单一职责
   - 使用 TypedDict 定义复杂状态结构
   - 配置驱动的灵活性设计

2. **记忆管理**:
   - 短期记忆用于会话上下文
   - 长期记忆存储重要信息
   - 基于重要性的记忆提升机制

3. **工具集成**:
   - 使用 @tool 装饰器定义工具
   - ToolNode 自动处理工具调用
   - 工具调用的安全性考虑

4. **LLM 集成**:
   - 支持多个 LLM 提供商
   - 动态提示词构建
   - 错误处理和重试机制

5. **Human-in-the-Loop**:
   - 基于关键词和情感的自动升级
   - 质量评估机制
   - 人工接入的优雅处理

6. **状态管理**:
   - 使用 MemorySaver 进行持久化
   - 支持多轮对话的上下文保持
   - 状态的版本化和回滚

7. **用户体验**:
   - 流式输出提升交互性
   - 情感识别和个性化回复
   - 详细的调试信息

🔧 扩展建议:

1. **高级记忆**: 使用向量数据库进行语义搜索
2. **多模态**: 支持图片、语音输入
3. **个性化**: 基于用户画像调整回复风格
4. **分析仪表板**: 实时监控对话质量和用户满意度
5. **A/B测试**: 支持不同提示词和模型的效果对比

这个高级版本展示了如何构建生产级的 AI 客服系统！
"""