"""
@tool 装饰器详解示例

本文件展示 @tool 装饰器的详细用法和工作原理：

1. @tool 装饰器基础
2. 工具函数的关键要素
3. 不同类型的工具示例
4. 工具的 JSON Schema
5. 工具绑定和调用流程
6. ReAct 模式实际应用
7. 最佳实践和注意事项

核心要点：
- @tool 将 Python 函数转换为 LLM 可调用的工具
- 文档字符串是 LLM 理解工具用途的关键
- 类型注解必须准确，用于生成 JSON Schema
- 工具让 LLM 具备执行具体任务的能力
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import asyncio
import time

load_dotenv()

# 1. @tool 装饰器基础
print("=" * 50)
print("1. @tool 装饰器基础")
print("=" * 50)

"""
@tool 装饰器的作用：
- 将普通 Python 函数转换为 LLM 可调用的工具
- 自动提取函数签名、参数类型、文档字符串
- 生成符合 OpenAI Function Calling 协议的 JSON Schema
- 使 LLM 具备执行具体任务的能力
"""

# 最基本的工具示例
@tool
def simple_calculator(a: int, b: int, operation: str) -> int:
    """
    执行基本数学运算的计算器

    Args:
        a: 第一个数字
        b: 第二个数字
        operation: 运算类型 (add, subtract, multiply, divide)

    Returns:
        计算结果
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a // b if b != 0 else 0
    else:
        return 0

print("基础工具创建完成：simple_calculator")

# 2. 工具函数的关键要素
print("\n" + "=" * 50)
print("2. 工具函数的关键要素")
print("=" * 50)

"""
工具函数必须包含的要素：

1. 类型注解（必需）
   - 所有参数必须有类型提示
   - 返回值建议有类型提示
   - 用于生成 JSON Schema

2. 文档字符串（强烈推荐）
   - 描述函数功能和用途
   - LLM 通过它理解何时使用工具
   - 包含参数和返回值说明

3. 函数实现
   - 实际执行的逻辑
   - 应该处理异常情况
   - 返回有意义的结果
"""

# 展示工具的 JSON Schema
print("工具的 JSON Schema 示例：")
print(json.dumps(simple_calculator.args, indent=2, ensure_ascii=False))

# 3. 不同类型的工具示例
print("\n" + "=" * 50)
print("3. 不同类型的工具示例")
print("=" * 50)

# 数据处理工具
@tool
def analyze_text(text: str) -> Dict[str, Any]:
    """
    分析文本的基本统计信息

    Args:
        text: 要分析的文本

    Returns:
        包含统计信息的字典
    """
    words = text.split()
    return {
        "字符数": len(text),
        "单词数": len(words),
        "行数": len(text.split('\n')),
        "最长单词": max(words, key=len) if words else "",
        "最短单词": min(words, key=len) if words else ""
    }

# 文件操作工具
@tool
def read_file_info(file_path: str) -> Dict[str, str]:
    """
    读取文件的基本信息

    Args:
        file_path: 文件路径

    Returns:
        文件信息字典
    """
    try:
        import os
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            return {
                "文件大小": f"{stat.st_size} 字节",
                "文件类型": "目录" if os.path.isdir(file_path) else "文件",
                "是否存在": "是",
                "扩展名": os.path.splitext(file_path)[1]
            }
        else:
            return {"是否存在": "否", "错误": "文件不存在"}
    except Exception as e:
        return {"错误": str(e)}

# 列表处理工具
@tool
def process_numbers(numbers: List[int], operation: str) -> Dict[str, float]:
    """
    对数字列表执行统计操作

    Args:
        numbers: 数字列表
        operation: 操作类型 (sum, avg, max, min, all)

    Returns:
        统计结果
    """
    if not numbers:
        return {"错误": "数字列表为空"}

    results = {}
    if operation in ["sum", "all"]:
        results["总和"] = sum(numbers)
    if operation in ["avg", "all"]:
        results["平均值"] = sum(numbers) / len(numbers)
    if operation in ["max", "all"]:
        results["最大值"] = max(numbers)
    if operation in ["min", "all"]:
        results["最小值"] = min(numbers)
    if operation == "all":
        results["数量"] = len(numbers)

    return results

# 时间工具
@tool
def get_current_time(format_type: str = "datetime") -> str:
    """
    获取当前时间

    Args:
        format_type: 格式类型 (datetime, date, time, timestamp)

    Returns:
        格式化的时间字符串
    """
    from datetime import datetime
    now = datetime.now()

    if format_type == "datetime":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    elif format_type == "timestamp":
        return str(int(now.timestamp()))
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")

print("创建了多种类型的工具：")
print("- analyze_text: 文本分析")
print("- read_file_info: 文件信息")
print("- process_numbers: 数字统计")
print("- get_current_time: 时间获取")

# 4. 工具绑定和调用流程
print("\n" + "=" * 50)
print("4. 工具绑定和调用流程")
print("=" * 50)

# 创建工具列表
tools = [
    simple_calculator,
    analyze_text,
    read_file_info,
    process_numbers,
    get_current_time
]

print(f"总共创建了 {len(tools)} 个工具")

# 如果配置了 API，演示工具绑定
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model = os.getenv("OPENROUTER_MODEL")

if api_key and base_url and model:
    print("\n配置了 API，演示工具绑定过程：")

    # 创建 LLM 并绑定工具
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    print("工具绑定完成！LLM 现在可以调用以下工具：")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}: {tool.description}")

    # 演示工具调用过程
    print("\n演示工具调用过程：")

    def demonstrate_tool_calling():
        """演示工具调用的完整流程"""

        # 用户消息
        user_message = "请帮我计算 15 乘以 8，然后分析文本 'Hello World from LangGraph!'"
        print(f"用户问题: {user_message}")

        try:
            # LLM 生成响应（可能包含工具调用）
            response = llm_with_tools.invoke([HumanMessage(content=user_message)])
            print(f"\nLLM 响应类型: {type(response).__name__}")

            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"LLM 决定调用 {len(response.tool_calls)} 个工具:")

                tool_results = []
                for tool_call in response.tool_calls:
                    print(f"\n工具调用: {tool_call['name']}")
                    print(f"参数: {tool_call['args']}")

                    # 手动执行工具（实际使用中 ToolNode 会自动处理）
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    # 找到对应的工具并执行
                    for tool in tools:
                        if tool.name == tool_name:
                            result = tool.invoke(tool_args)
                            print(f"工具结果: {result}")
                            tool_results.append(result)
                            break

                return tool_results
            else:
                print("LLM 没有调用任何工具")
                print(f"直接回复: {response.content}")

        except Exception as e:
            print(f"调用过程中出错: {e}")
            return None

    demonstrate_tool_calling()

else:
    print("未配置 API 信息，跳过工具绑定演示")

# 5. 手动演示工具功能
print("\n" + "=" * 50)
print("5. 手动演示工具功能")
print("=" * 50)

# 演示各个工具的功能
print("演示 simple_calculator:")
calc_result = simple_calculator.invoke({"a": 15, "b": 8, "operation": "multiply"})
print(f"15 × 8 = {calc_result}")

print("\n演示 analyze_text:")
text_result = analyze_text.invoke({"text": "Hello World from LangGraph! This is a test."})
print(f"文本分析结果: {text_result}")

print("\n演示 process_numbers:")
numbers_result = process_numbers.invoke({"numbers": [1, 2, 3, 4, 5], "operation": "all"})
print(f"数字处理结果: {numbers_result}")

print("\n演示 get_current_time:")
time_result = get_current_time.invoke({"format_type": "datetime"})
print(f"当前时间: {time_result}")

# 6. 高级工具示例
print("\n" + "=" * 50)
print("6. 高级工具示例")
print("=" * 50)

# 带有错误处理的工具
@tool
def safe_divide(a: float, b: float) -> Dict[str, Any]:
    """
    安全的除法运算，包含错误处理

    Args:
        a: 被除数
        b: 除数

    Returns:
        包含结果或错误信息的字典
    """
    try:
        if b == 0:
            return {"错误": "除数不能为零", "结果": None}
        result = a / b
        return {"结果": result, "错误": None}
    except Exception as e:
        return {"错误": str(e), "结果": None}

# 可选参数的工具
@tool
def format_message(message: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> str:
    """
    格式化消息，支持可选的前缀和后缀

    Args:
        message: 主要消息内容
        prefix: 可选的前缀
        suffix: 可选的后缀

    Returns:
        格式化后的消息
    """
    result = message
    if prefix:
        result = f"{prefix}{result}"
    if suffix:
        result = f"{result}{suffix}"
    return result

print("高级工具演示：")
print("安全除法:", safe_divide.invoke({"a": 10, "b": 0}))
print("消息格式化:", format_message.invoke({"message": "Hello", "prefix": "[INFO] ", "suffix": " !!!"}))

# 7. 最佳实践和注意事项
print("\n" + "=" * 50)
print("7. 最佳实践和注意事项")
print("=" * 50)

"""
@tool 装饰器的最佳实践：

1. 文档字符串要清晰准确
   - 描述工具的具体功能
   - 说明何时应该使用这个工具
   - 包含参数和返回值说明

2. 类型注解必须准确
   - 所有参数都要有类型提示
   - 支持 Union、Optional、List、Dict 等复杂类型
   - 类型信息用于生成 JSON Schema

3. 错误处理要完善
   - 处理可能的异常情况
   - 返回有意义的错误信息
   - 避免工具执行失败导致整个流程中断

4. 函数要保持简单
   - 每个工具专注一个功能
   - 避免过于复杂的逻辑
   - 便于 LLM 理解和选择

5. 返回结果要有用
   - 返回 LLM 能够理解的格式
   - 包含足够的信息用于后续处理
   - 避免返回过于复杂的对象

6. 性能考虑
   - 避免耗时过长的操作
   - 考虑异步工具（需要特殊处理）
   - 合理设置超时机制
"""

print("""
总结 - @tool 装饰器的核心价值：

1. 扩展 LLM 能力
   - 让 LLM 能够执行具体的计算和操作
   - 访问外部数据和服务
   - 与系统和应用程序交互

2. 实现 ReAct 模式
   - Reasoning: LLM 推理需要什么工具
   - Acting: 执行工具获取结果
   - 循环进行直到解决问题

3. 提供结构化接口
   - 通过 JSON Schema 定义工具接口
   - LLM 可以准确调用工具
   - 支持复杂的参数传递

4. 促进模块化
   - 工具可以复用
   - 便于测试和维护
   - 支持组合使用
""")

print("\n" + "=" * 50)
print("演示完成！")
print("=" * 50)