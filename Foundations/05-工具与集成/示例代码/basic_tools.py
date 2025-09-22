#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础工具定义示例

本文件演示了如何在 LangGraph 中定义和使用各种类型的工具：
1. 使用 @tool 装饰器创建简单工具
2. 使用 BaseTool 类创建复杂工具
3. 带参数验证的工具
4. 错误处理和重试机制
5. 状态感知工具

运行方式:
    python basic_tools.py
"""

import time
import random
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain_core.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


# ================================
# 1. 基础工具示例
# ================================

@tool
def simple_calculator(expression: str) -> str:
    """简单计算器

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        # 安全的表达式计算（仅支持基本运算）
        allowed_chars = set("0123456789+-*/().= ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含不允许的字符"

        # 简单计算
        result = eval(expression)
        return f"{expression} = {result}"

    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def text_analyzer(text: str) -> Dict[str, Any]:
    """文本分析器

    Args:
        text: 要分析的文本

    Returns:
        文本分析结果
    """
    if not text or not isinstance(text, str):
        return {"error": "无效的文本输入"}

    # 基础文本分析
    words = text.split()
    sentences = text.split('。') + text.split('.') + text.split('!')
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_chinese": any('\u4e00' <= c <= '\u9fff' for c in text),
        "has_english": any(c.isalpha() and ord(c) < 128 for c in text),
        "has_numbers": any(c.isdigit() for c in text),
        "longest_word": max(words, key=len) if words else "",
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }


@tool
def random_data_generator(data_type: str, count: int = 5) -> List[Any]:
    """随机数据生成器

    Args:
        data_type: 数据类型 ('number', 'name', 'email', 'color')
        count: 生成数量

    Returns:
        随机数据列表
    """
    if count <= 0 or count > 100:
        return ["错误: 数量必须在 1-100 之间"]

    generators = {
        'number': lambda: random.randint(1, 1000),
        'name': lambda: random.choice(['张三', '李四', '王五', '赵六', '钱七', 'Alice', 'Bob', 'Charlie']),
        'email': lambda: f"user{random.randint(1, 999)}@{random.choice(['gmail.com', 'qq.com', 'outlook.com'])}",
        'color': lambda: random.choice(['红色', '蓝色', '绿色', '黄色', '紫色', '橙色', '黑色', '白色'])
    }

    if data_type not in generators:
        return [f"错误: 不支持的数据类型 {data_type}"]

    generator = generators[data_type]
    return [generator() for _ in range(count)]


# ================================
# 2. BaseTool 复杂工具示例
# ================================

class WeatherSimulatorInput(BaseModel):
    """天气模拟器输入模型"""
    city: str = Field(description="城市名称")
    days: int = Field(default=1, description="预测天数", ge=1, le=7)
    units: str = Field(default="celsius", description="温度单位")

    @validator('city')
    def validate_city(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("城市名称不能为空")
        return v.strip()

    @validator('units')
    def validate_units(cls, v):
        if v not in ['celsius', 'fahrenheit']:
            raise ValueError("温度单位必须是 'celsius' 或 'fahrenheit'")
        return v


class WeatherSimulatorTool(BaseTool):
    """天气模拟器工具"""

    name: str = "weather_simulator"
    description: str = "模拟指定城市的天气预报"
    args_schema: type[BaseModel] = WeatherSimulatorInput

    def _run(
        self,
        city: str,
        days: int = 1,
        units: str = "celsius",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行天气模拟"""
        try:
            # 模拟天气数据
            weather_conditions = ['晴天', '多云', '小雨', '阴天', '雷阵雨']
            forecasts = []

            for day in range(days):
                # 生成随机天气
                base_temp = random.randint(15, 30)
                if units == "fahrenheit":
                    temp = base_temp * 9/5 + 32
                    temp_unit = "°F"
                else:
                    temp = base_temp
                    temp_unit = "°C"

                forecast = {
                    "day": day + 1,
                    "date": f"2024-01-{15 + day:02d}",
                    "condition": random.choice(weather_conditions),
                    "temperature": f"{temp:.1f}{temp_unit}",
                    "humidity": f"{random.randint(40, 80)}%",
                    "wind_speed": f"{random.randint(5, 25)} km/h"
                }
                forecasts.append(forecast)

            return {
                "city": city,
                "forecast_days": days,
                "units": units,
                "forecasts": forecasts,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {"error": f"天气模拟失败: {str(e)}"}

    async def _arun(
        self,
        city: str,
        days: int = 1,
        units: str = "celsius",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """异步执行天气模拟"""
        # 对于这个示例，我们只是调用同步版本
        return self._run(city, days, units, run_manager)


class DataProcessorTool(BaseTool):
    """数据处理器工具"""

    name: str = "data_processor"
    description: str = "处理和分析数据集"

    def _run(
        self,
        data: List[Dict[str, Any]],
        operation: str = "summary",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行数据处理"""
        if not data or not isinstance(data, list):
            return {"error": "无效的数据输入"}

        try:
            if operation == "summary":
                return self._generate_summary(data)
            elif operation == "filter":
                return self._filter_data(data)
            elif operation == "aggregate":
                return self._aggregate_data(data)
            else:
                return {"error": f"不支持的操作: {operation}"}

        except Exception as e:
            return {"error": f"数据处理失败: {str(e)}"}

    def _generate_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据摘要"""
        if not data:
            return {"summary": "空数据集"}

        total_records = len(data)
        fields = set()
        for record in data:
            if isinstance(record, dict):
                fields.update(record.keys())

        return {
            "total_records": total_records,
            "fields": list(fields),
            "field_count": len(fields),
            "sample_record": data[0] if data else None
        }

    def _filter_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """过滤数据"""
        # 简单过滤：移除空值记录
        filtered_data = [record for record in data if record and isinstance(record, dict)]

        return {
            "original_count": len(data),
            "filtered_count": len(filtered_data),
            "removed_count": len(data) - len(filtered_data),
            "filtered_data": filtered_data[:5]  # 只返回前5条
        }

    def _aggregate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合数据"""
        numeric_fields = {}
        text_fields = {}

        for record in data:
            if isinstance(record, dict):
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)
                    elif isinstance(value, str):
                        if key not in text_fields:
                            text_fields[key] = []
                        text_fields[key].append(value)

        # 计算数值字段统计
        numeric_stats = {}
        for field, values in numeric_fields.items():
            if values:
                numeric_stats[field] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }

        # 计算文本字段统计
        text_stats = {}
        for field, values in text_fields.items():
            if values:
                unique_values = set(values)
                text_stats[field] = {
                    "count": len(values),
                    "unique_count": len(unique_values),
                    "most_common": max(unique_values, key=values.count) if unique_values else None
                }

        return {
            "numeric_fields": numeric_stats,
            "text_fields": text_stats,
            "total_records": len(data)
        }


# ================================
# 3. 错误处理和重试机制
# ================================

class ReliableAPITool(BaseTool):
    """可靠的 API 工具（带重试机制）"""

    name: str = "reliable_api"
    description: str = "具有重试机制的可靠 API 调用工具"

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _run(
        self,
        endpoint: str,
        method: str = "GET",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行带重试的 API 调用"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # 模拟 API 调用
                result = self._simulate_api_call(endpoint, method)

                if attempt > 0:
                    result["retries"] = attempt

                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"第 {attempt + 1} 次尝试失败，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                    # 指数退避
                    self.retry_delay *= 1.5

        return {
            "error": f"API 调用失败，已重试 {self.max_retries} 次",
            "last_error": str(last_error),
            "endpoint": endpoint
        }

    def _simulate_api_call(self, endpoint: str, method: str) -> Dict[str, Any]:
        """模拟 API 调用"""
        # 30% 的概率失败
        if random.random() < 0.3:
            raise Exception("模拟网络错误")

        # 模拟响应数据
        mock_responses = {
            "/users": {"users": [{"id": 1, "name": "User1"}, {"id": 2, "name": "User2"}]},
            "/products": {"products": [{"id": 1, "name": "Product1", "price": 99.99}]},
            "/status": {"status": "healthy", "timestamp": time.time()}
        }

        return {
            "status": "success",
            "method": method,
            "endpoint": endpoint,
            "data": mock_responses.get(endpoint, {"message": "默认响应"}),
            "timestamp": time.time()
        }


# ================================
# 4. 状态感知工具
# ================================

class StatefulProcessorTool(BaseTool):
    """状态感知处理器工具"""

    name: str = "stateful_processor"
    description: str = "能够访问和修改状态的处理器工具"

    def _run(
        self,
        operation: str,
        data: Any = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行状态感知操作"""
        # 在实际应用中，这里会从 run_manager 或其他方式获取状态
        # 这里我们模拟一个简单的状态管理

        try:
            if operation == "process_data":
                return self._process_data(data)
            elif operation == "get_status":
                return self._get_status()
            elif operation == "reset":
                return self._reset_state()
            else:
                return {"error": f"未知操作: {operation}"}

        except Exception as e:
            return {"error": f"状态操作失败: {str(e)}"}

    def _process_data(self, data: Any) -> Dict[str, Any]:
        """处理数据"""
        if data is None:
            return {"error": "缺少数据"}

        # 模拟数据处理
        processed_data = {
            "original": data,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_id": random.randint(1000, 9999)
        }

        if isinstance(data, str):
            processed_data["length"] = len(data)
            processed_data["word_count"] = len(data.split())
        elif isinstance(data, (list, dict)):
            processed_data["size"] = len(data)

        return {
            "status": "processed",
            "result": processed_data
        }

    def _get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "tool_name": self.name,
            "status": "active",
            "last_operation": "get_status",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory_usage": f"{random.randint(50, 200)}MB"
        }

    def _reset_state(self) -> Dict[str, Any]:
        """重置状态"""
        return {
            "status": "reset",
            "message": "状态已重置",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


# ================================
# 5. 工具集合管理器
# ================================

class ToolCollectionManager:
    """工具集合管理器"""

    def __init__(self):
        self.tools = []
        self._register_all_tools()

    def _register_all_tools(self):
        """注册所有工具"""
        # 注册函数工具
        function_tools = [
            simple_calculator,
            text_analyzer,
            random_data_generator
        ]

        # 注册类工具
        class_tools = [
            WeatherSimulatorTool(),
            DataProcessorTool(),
            ReliableAPITool(max_retries=2),
            StatefulProcessorTool()
        ]

        self.tools.extend(function_tools)
        self.tools.extend(class_tools)

    def get_all_tools(self) -> List:
        """获取所有工具"""
        return self.tools

    def get_tool_by_name(self, name: str):
        """根据名称获取工具"""
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == name:
                return tool
        return None

    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有工具信息"""
        tool_info = []
        for tool in self.tools:
            if hasattr(tool, 'name'):
                tool_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "type": "class_tool" if hasattr(tool, '_run') else "function_tool"
                })
        return tool_info

    def test_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """测试指定工具"""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return {"error": f"工具 {tool_name} 不存在"}

        try:
            if hasattr(tool, 'invoke'):
                result = tool.invoke(kwargs)
            elif hasattr(tool, '_run'):
                result = tool._run(**kwargs)
            else:
                result = tool(**kwargs)

            return {
                "tool": tool_name,
                "success": True,
                "result": result
            }

        except Exception as e:
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }


# ================================
# 6. 示例和测试
# ================================

def run_tool_examples():
    """运行工具示例"""
    print("🛠️  基础工具定义示例")
    print("=" * 60)

    # 创建工具管理器
    manager = ToolCollectionManager()

    # 列出所有工具
    print("\n📋 可用工具列表:")
    for tool_info in manager.list_tools():
        print(f"• {tool_info['name']}: {tool_info['description']} ({tool_info['type']})")

    # 测试函数工具
    print("\n🧮 测试简单计算器:")
    calc_result = manager.test_tool("simple_calculator", expression="2 + 3 * 4")
    print(f"结果: {calc_result}")

    print("\n📝 测试文本分析器:")
    text_result = manager.test_tool("text_analyzer", text="Hello 世界! 这是一个测试文本。")
    print(f"结果: {text_result}")

    print("\n🎲 测试随机数据生成器:")
    random_result = manager.test_tool("random_data_generator", data_type="name", count=3)
    print(f"结果: {random_result}")

    # 测试类工具
    print("\n🌤️  测试天气模拟器:")
    weather_result = manager.test_tool("weather_simulator", city="北京", days=3, units="celsius")
    print(f"结果: {weather_result}")

    print("\n📊 测试数据处理器:")
    sample_data = [
        {"name": "Alice", "age": 25, "city": "Beijing"},
        {"name": "Bob", "age": 30, "city": "Shanghai"},
        {"name": "Charlie", "age": 35, "city": "Guangzhou"}
    ]
    data_result = manager.test_tool("data_processor", data=sample_data, operation="summary")
    print(f"结果: {data_result}")

    print("\n🔄 测试可靠 API 工具:")
    api_result = manager.test_tool("reliable_api", endpoint="/users", method="GET")
    print(f"结果: {api_result}")

    print("\n📈 测试状态感知工具:")
    state_result = manager.test_tool("stateful_processor", operation="process_data", data="测试数据")
    print(f"结果: {state_result}")


def demonstrate_tool_integration():
    """演示工具集成"""
    print("\n🔧 工具集成演示")
    print("=" * 60)

    manager = ToolCollectionManager()

    # 模拟一个完整的工作流
    print("\n📝 步骤1: 生成测试数据")
    names = manager.test_tool("random_data_generator", data_type="name", count=5)
    print(f"生成的姓名: {names['result']}")

    print("\n📊 步骤2: 创建数据结构")
    test_data = []
    if names['success']:
        for i, name in enumerate(names['result'][:3]):
            test_data.append({
                "id": i + 1,
                "name": name,
                "age": random.randint(20, 60),
                "score": random.randint(60, 100)
            })

    print(f"创建的数据: {test_data}")

    print("\n📈 步骤3: 分析数据")
    analysis = manager.test_tool("data_processor", data=test_data, operation="aggregate")
    print(f"数据分析结果: {analysis['result']}")

    print("\n☁️  步骤4: 获取环境信息")
    weather = manager.test_tool("weather_simulator", city="上海", days=1)
    if weather['success']:
        forecast = weather['result']['forecasts'][0]
        print(f"天气信息: {forecast['condition']}, {forecast['temperature']}")

    print("\n✅ 工作流完成!")


if __name__ == "__main__":
    # 运行所有示例
    run_tool_examples()
    demonstrate_tool_integration()

    print("\n" + "=" * 60)
    print("🎉 基础工具示例运行完成!")
    print("你可以参考这些示例来创建自己的工具。")