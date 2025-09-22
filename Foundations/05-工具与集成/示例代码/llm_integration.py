#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 集成示例

本文件演示了如何在 LangGraph 中集成和配置各种 LLM 模型：
1. 多种模型提供商配置
2. Tool Calling 机制
3. 流式输出处理
4. 模型降级策略
5. Token 管理和成本控制

运行方式:
    python llm_integration.py
"""

import os
import time
import asyncio
import tiktoken
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableWithFallbacks


# ================================
# 1. 模型配置工厂
# ================================

class ModelFactory:
    """LLM 模型工厂"""

    @staticmethod
    def create_openai_model(model_name: str = "gpt-3.5-turbo", **kwargs):
        """创建 OpenAI 模型"""
        try:
            from langchain_openai import ChatOpenAI

            default_config = {
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 1000,
                "timeout": 30,
                "max_retries": 2
            }
            default_config.update(kwargs)

            return ChatOpenAI(**default_config)
        except ImportError:
            print("警告: langchain_openai 未安装，返回模拟模型")
            return MockLLM("OpenAI", model_name)

    @staticmethod
    def create_anthropic_model(model_name: str = "claude-3-haiku-20240307", **kwargs):
        """创建 Anthropic 模型"""
        try:
            from langchain_anthropic import ChatAnthropic

            default_config = {
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            default_config.update(kwargs)

            return ChatAnthropic(**default_config)
        except ImportError:
            print("警告: langchain_anthropic 未安装，返回模拟模型")
            return MockLLM("Anthropic", model_name)

    @staticmethod
    def create_azure_model(deployment_name: str = "gpt-35-turbo", **kwargs):
        """创建 Azure OpenAI 模型"""
        try:
            from langchain_openai import AzureChatOpenAI

            default_config = {
                "deployment_name": deployment_name,
                "temperature": 0.1,
                "max_tokens": 1000,
                "api_version": "2024-02-15-preview"
            }
            default_config.update(kwargs)

            return AzureChatOpenAI(**default_config)
        except ImportError:
            print("警告: Azure OpenAI 配置未完成，返回模拟模型")
            return MockLLM("Azure", deployment_name)

    @staticmethod
    def create_google_model(model_name: str = "gemini-pro", **kwargs):
        """创建 Google 模型"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            default_config = {
                "model": model_name,
                "temperature": 0.1
            }
            default_config.update(kwargs)

            return ChatGoogleGenerativeAI(**default_config)
        except ImportError:
            print("警告: langchain_google_genai 未安装，返回模拟模型")
            return MockLLM("Google", model_name)


# ================================
# 2. 模拟 LLM（用于演示）
# ================================

class MockLLM:
    """模拟 LLM 类"""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.call_count = 0

    def invoke(self, messages: List) -> AIMessage:
        """模拟调用"""
        self.call_count += 1

        # 解析消息
        if isinstance(messages, list):
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
        else:
            content = str(messages)

        # 生成模拟响应
        responses = {
            "hello": "Hello! I'm a mock LLM response.",
            "天气": f"今天天气晴朗，温度适宜。({self.provider} {self.model})",
            "计算": "我可以帮您计算数学问题。",
            "工具": "我理解您需要使用工具来完成任务。",
            "默认": f"这是来自 {self.provider} {self.model} 的模拟响应 #{self.call_count}"
        }

        # 选择响应
        response_text = responses.get("默认", responses["默认"])
        for key, response in responses.items():
            if key in content.lower():
                response_text = response
                break

        return AIMessage(content=response_text)

    async def ainvoke(self, messages: List) -> AIMessage:
        """异步模拟调用"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return self.invoke(messages)

    def stream(self, messages: List):
        """模拟流式输出"""
        response = self.invoke(messages)
        words = response.content.split()

        for word in words:
            chunk = AIMessage(content=word + " ")
            yield chunk
            time.sleep(0.1)  # 模拟流式延迟

    def bind_tools(self, tools: List):
        """绑定工具"""
        self.tools = tools
        return MockLLMWithTools(self, tools)


class MockLLMWithTools:
    """带工具的模拟 LLM"""

    def __init__(self, llm: MockLLM, tools: List):
        self.llm = llm
        self.tools = tools

    def invoke(self, messages: List) -> AIMessage:
        """带工具调用的模拟响应"""
        last_message = messages[-1] if messages else ""
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # 检查是否需要调用工具
        tool_keywords = {
            "计算": "calculator",
            "天气": "weather",
            "时间": "current_time",
            "随机": "random_number"
        }

        for keyword, tool_name in tool_keywords.items():
            if keyword in content:
                # 模拟工具调用
                tool_call = {
                    "name": tool_name,
                    "args": {"input": content},
                    "id": f"call_{int(time.time())}"
                }

                response = AIMessage(
                    content="",
                    tool_calls=[tool_call]
                )
                return response

        # 普通响应
        return self.llm.invoke(messages)


# ================================
# 3. 工具定义
# ================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    try:
        # 安全的数学计算
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含不允许的字符"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def weather_info(city: str) -> str:
    """获取天气信息

    Args:
        city: 城市名称

    Returns:
        天气信息
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，15°C，微风",
        "上海": "多云，18°C，东南风",
        "广州": "小雨，22°C，湿度较高",
        "深圳": "晴天，25°C，海风习习"
    }

    return weather_data.get(city, f"{city}的天气信息暂不可用")


@tool
def current_time() -> str:
    """获取当前时间

    Returns:
        当前时间字符串
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


@tool
def random_number(min_val: int = 1, max_val: int = 100) -> int:
    """生成随机数

    Args:
        min_val: 最小值
        max_val: 最大值

    Returns:
        随机数
    """
    import random
    return random.randint(min_val, max_val)


# ================================
# 4. 模型配置管理器
# ================================

@dataclass
class ModelConfig:
    """模型配置"""
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    streaming: bool = False
    tools: List = None


class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.models = {}
        self.current_model = None
        self.tools = [calculator, weather_info, current_time, random_number]

    def register_model(self, name: str, config: ModelConfig):
        """注册模型"""
        factory_methods = {
            "openai": ModelFactory.create_openai_model,
            "anthropic": ModelFactory.create_anthropic_model,
            "azure": ModelFactory.create_azure_model,
            "google": ModelFactory.create_google_model
        }

        if config.provider not in factory_methods:
            raise ValueError(f"不支持的模型提供商: {config.provider}")

        # 创建模型
        model_kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }

        if config.streaming:
            model_kwargs["streaming"] = True

        model = factory_methods[config.provider](config.model, **model_kwargs)

        # 绑定工具
        if config.tools or self.tools:
            tools_to_bind = config.tools or self.tools
            model = model.bind_tools(tools_to_bind)

        self.models[name] = {
            "model": model,
            "config": config
        }

        print(f"模型 '{name}' 注册成功 ({config.provider} {config.model})")

    def get_model(self, name: str):
        """获取模型"""
        if name not in self.models:
            return None
        return self.models[name]["model"]

    def set_current_model(self, name: str):
        """设置当前模型"""
        if name in self.models:
            self.current_model = name
            print(f"当前模型设置为: {name}")
        else:
            print(f"模型 '{name}' 不存在")

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        return [
            {
                "name": name,
                "provider": info["config"].provider,
                "model": info["config"].model,
                "current": name == self.current_model
            }
            for name, info in self.models.items()
        ]

    def create_fallback_model(self, primary: str, fallbacks: List[str]):
        """创建带降级的模型"""
        primary_model = self.get_model(primary)
        fallback_models = [self.get_model(name) for name in fallbacks]

        if not primary_model:
            raise ValueError(f"主模型 '{primary}' 不存在")

        fallback_models = [model for model in fallback_models if model is not None]

        if not fallback_models:
            print("警告: 没有有效的降级模型")
            return primary_model

        return RunnableWithFallbacks(
            runnable=primary_model,
            fallbacks=fallback_models
        )


# ================================
# 5. 对话管理器
# ================================

class ConversationManager:
    """对话管理器"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.conversation_history = []
        self.system_prompt = "你是一个有用的AI助手，能够使用工具来帮助用户解决问题。"

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def add_message(self, message):
        """添加消息到历史"""
        self.conversation_history.append(message)

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def get_messages(self) -> List:
        """获取完整消息列表"""
        messages = []

        # 添加系统消息
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # 添加对话历史
        messages.extend(self.conversation_history)

        return messages

    def chat(self, user_input: str, model_name: str = None) -> Dict[str, Any]:
        """进行对话"""
        # 使用指定模型或当前模型
        if model_name:
            model = self.model_manager.get_model(model_name)
        elif self.model_manager.current_model:
            model = self.model_manager.get_model(self.model_manager.current_model)
        else:
            return {"error": "没有可用的模型"}

        if not model:
            return {"error": f"模型不存在: {model_name or self.model_manager.current_model}"}

        # 添加用户消息
        user_message = HumanMessage(content=user_input)
        self.add_message(user_message)

        try:
            # 获取模型响应
            messages = self.get_messages()
            start_time = time.time()
            response = model.invoke(messages)
            duration = time.time() - start_time

            # 检查工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                return self._handle_tool_calls(response, model, duration)
            else:
                # 普通响应
                self.add_message(response)
                return {
                    "type": "text",
                    "content": response.content,
                    "duration": duration,
                    "model": model_name or self.model_manager.current_model
                }

        except Exception as e:
            return {"error": f"对话失败: {str(e)}"}

    def _handle_tool_calls(self, response, model, initial_duration: float) -> Dict[str, Any]:
        """处理工具调用"""
        tool_results = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 查找并执行工具
            tool_result = self._execute_tool(tool_name, tool_args)
            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result
            })

        # 将工具调用和结果添加到历史
        self.add_message(response)

        # 构建最终响应
        final_response = f"我使用了以下工具:\n"
        for result in tool_results:
            final_response += f"• {result['tool']}: {result['result']}\n"

        return {
            "type": "tool_call",
            "content": final_response,
            "tool_calls": tool_results,
            "duration": initial_duration
        }

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行工具"""
        tools_map = {tool.name: tool for tool in self.model_manager.tools}

        if tool_name not in tools_map:
            return f"工具 {tool_name} 不存在"

        try:
            tool = tools_map[tool_name]
            result = tool.invoke(args)
            return str(result)
        except Exception as e:
            return f"工具执行失败: {str(e)}"

    async def stream_chat(self, user_input: str, model_name: str = None):
        """流式对话"""
        # 使用指定模型或当前模型
        if model_name:
            model = self.model_manager.get_model(model_name)
        elif self.model_manager.current_model:
            model = self.model_manager.get_model(self.model_manager.current_model)
        else:
            yield {"error": "没有可用的模型"}
            return

        if not model:
            yield {"error": f"模型不存在: {model_name or self.model_manager.current_model}"}
            return

        # 添加用户消息
        user_message = HumanMessage(content=user_input)
        self.add_message(user_message)

        try:
            messages = self.get_messages()
            full_response = ""

            # 流式获取响应
            for chunk in model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield {
                        "type": "stream",
                        "content": chunk.content,
                        "full_content": full_response
                    }

            # 添加完整响应到历史
            self.add_message(AIMessage(content=full_response))

            yield {
                "type": "complete",
                "content": full_response
            }

        except Exception as e:
            yield {"error": f"流式对话失败: {str(e)}"}


# ================================
# 6. Token 管理器
# ================================

class TokenManager:
    """Token 管理器"""

    def __init__(self):
        self.costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """计算 token 数量"""
        try:
            # 尝试使用 tiktoken
            encoding = tiktoken.encoding_for_model(model.split("-")[0] + "-" + model.split("-")[1])
            return len(encoding.encode(text))
        except:
            # 简单估算：中文字符=1.5 tokens，英文单词=1 token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            words = len(text.split())
            return int(chinese_chars * 1.5 + words)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> Dict[str, float]:
        """估算成本"""
        # 模糊匹配模型名称
        model_key = None
        for key in self.costs:
            if key in model.lower():
                model_key = key
                break

        if not model_key:
            return {"error": f"未知模型: {model}"}

        cost_config = self.costs[model_key]
        input_cost = (input_tokens / 1000) * cost_config["input"]
        output_cost = (output_tokens / 1000) * cost_config["output"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": model_key,
            "currency": "USD"
        }

    def track_usage(self, messages: List, response: str, model: str) -> Dict[str, Any]:
        """跟踪使用情况"""
        # 计算输入 tokens
        input_text = "\n".join([
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in messages
        ])
        input_tokens = self.count_tokens(input_text, model)

        # 计算输出 tokens
        output_tokens = self.count_tokens(response, model)

        # 估算成本
        cost = self.estimate_cost(input_tokens, output_tokens, model)

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "usage": cost,
            "messages_count": len(messages)
        }


# ================================
# 7. 示例和测试
# ================================

def setup_demo_models():
    """设置演示模型"""
    manager = ModelManager()

    # 注册多个模型配置
    configs = [
        ModelConfig("openai", "gpt-3.5-turbo", temperature=0.1, max_tokens=500),
        ModelConfig("openai", "gpt-4", temperature=0.2, max_tokens=1000),
        ModelConfig("anthropic", "claude-3-haiku-20240307", temperature=0.1),
        ModelConfig("google", "gemini-pro", temperature=0.1),
    ]

    for i, config in enumerate(configs):
        name = f"{config.provider}_{config.model.replace('-', '_')}"
        try:
            manager.register_model(name, config)
        except Exception as e:
            print(f"注册模型 {name} 失败: {e}")

    # 设置默认模型
    available_models = manager.list_models()
    if available_models:
        manager.set_current_model(available_models[0]["name"])

    return manager


def test_basic_chat():
    """测试基础对话"""
    print("\n💬 基础对话测试")
    print("=" * 50)

    # 设置模型
    manager = setup_demo_models()
    conversation = ConversationManager(manager)

    # 设置系统提示词
    conversation.set_system_prompt("你是一个友好的AI助手，擅长数学计算和提供实用信息。")

    # 测试对话
    test_inputs = [
        "你好，请介绍一下自己",
        "请计算 15 * 23",
        "北京今天天气怎么样？",
        "现在几点了？"
    ]

    for user_input in test_inputs:
        print(f"\n👤 用户: {user_input}")
        response = conversation.chat(user_input)

        if "error" in response:
            print(f"❌ 错误: {response['error']}")
        else:
            print(f"🤖 助手: {response['content']}")
            if response["type"] == "tool_call":
                print(f"🔧 使用了 {len(response['tool_calls'])} 个工具")


async def test_streaming_chat():
    """测试流式对话"""
    print("\n🌊 流式对话测试")
    print("=" * 50)

    manager = setup_demo_models()
    conversation = ConversationManager(manager)

    user_input = "请用200字左右介绍一下人工智能的发展历程"
    print(f"\n👤 用户: {user_input}")
    print("🤖 助手: ", end="", flush=True)

    async for chunk in conversation.stream_chat(user_input):
        if chunk.get("type") == "stream":
            print(chunk["content"], end="", flush=True)
        elif chunk.get("type") == "complete":
            print("\n✅ 流式输出完成")
        elif "error" in chunk:
            print(f"\n❌ 错误: {chunk['error']}")


def test_model_comparison():
    """测试模型对比"""
    print("\n🔄 模型对比测试")
    print("=" * 50)

    manager = setup_demo_models()
    conversation = ConversationManager(manager)

    available_models = manager.list_models()
    test_question = "请简单解释什么是机器学习？"

    print(f"测试问题: {test_question}\n")

    for model_info in available_models:
        model_name = model_info["name"]
        print(f"📱 {model_name} ({model_info['provider']} {model_info['model']}):")

        # 清空历史以确保公平对比
        conversation.clear_history()

        response = conversation.chat(test_question, model_name)

        if "error" in response:
            print(f"❌ {response['error']}")
        else:
            print(f"💭 {response['content'][:100]}...")
            print(f"⏱️  耗时: {response['duration']:.2f}s")

        print()


def test_token_management():
    """测试 Token 管理"""
    print("\n📊 Token 管理测试")
    print("=" * 50)

    token_manager = TokenManager()

    # 测试文本
    test_texts = [
        "Hello, world!",
        "你好，世界！",
        "这是一个更长的测试文本，包含中英文混合内容。This is a longer test text with mixed Chinese and English content."
    ]

    for text in test_texts:
        tokens = token_manager.count_tokens(text)
        print(f"文本: {text[:30]}...")
        print(f"Token 数量: {tokens}")

        # 模拟响应
        response = "这是一个模拟的AI响应。"
        cost = token_manager.estimate_cost(tokens, token_manager.count_tokens(response), "gpt-3.5-turbo")
        print(f"估算成本: ${cost.get('total_cost', 0):.6f} USD")
        print()


def test_fallback_model():
    """测试模型降级"""
    print("\n🔄 模型降级测试")
    print("=" * 50)

    manager = setup_demo_models()
    available_models = [model["name"] for model in manager.list_models()]

    if len(available_models) < 2:
        print("需要至少2个模型来测试降级功能")
        return

    try:
        # 创建带降级的模型
        fallback_model = manager.create_fallback_model(
            primary=available_models[0],
            fallbacks=available_models[1:]
        )

        print(f"主模型: {available_models[0]}")
        print(f"降级模型: {available_models[1:]}")

        # 测试调用
        response = fallback_model.invoke([HumanMessage(content="你好")])
        print(f"响应: {response.content}")

    except Exception as e:
        print(f"降级测试失败: {e}")


async def run_all_tests():
    """运行所有测试"""
    print("🚀 LLM 集成示例测试")
    print("=" * 60)

    # 基础对话测试
    test_basic_chat()

    # 流式对话测试
    await test_streaming_chat()

    # 模型对比测试
    test_model_comparison()

    # Token 管理测试
    test_token_management()

    # 降级测试
    test_fallback_model()

    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")


if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(run_all_tests())