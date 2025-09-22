#!/usr/bin/env python3
"""
LangGraph 调试示例代码

演示各种调试技巧和工具的使用方法。
"""

from typing import TypedDict, List, Dict, Any
from langgraph import StateGraph, START, END
import json
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DebugState(TypedDict):
    """调试示例状态"""
    input_data: str
    processed_data: str
    step_count: int
    debug_info: Dict[str, Any]
    execution_log: List[str]

class DebugFormatter:
    """调试输出格式化器"""

    @staticmethod
    def print_state(state: DebugState, title: str = "状态"):
        """美化打印状态"""
        print(f"\n{'='*20} {title} {'='*20}")
        for key, value in state.items():
            if isinstance(value, dict):
                print(f"📋 {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                print(f"📝 {key} ({len(value)} 项):")
                for i, item in enumerate(value[-3:]):
                    print(f"    [{i}] {item}")
                if len(value) > 3:
                    print(f"    ... (还有 {len(value)-3} 项)")
            else:
                print(f"🔤 {key}: {value}")
        print("="*50 + "\n")

    @staticmethod
    def print_execution_flow(node_name: str, step: int, status: str = "执行中"):
        """打印执行流程"""
        status_emoji = {
            "开始": "🚀",
            "执行中": "⚙️",
            "完成": "✅",
            "错误": "❌"
        }

        emoji = status_emoji.get(status, "🔄")
        print(f"{emoji} 步骤 {step}: {node_name} - {status}")

def debug_log(func):
    """节点函数调试装饰器"""
    def wrapper(state: DebugState) -> DebugState:
        func_name = func.__name__
        start_time = time.time()

        # 记录开始
        DebugFormatter.print_execution_flow(func_name, state['step_count'] + 1, "开始")
        DebugFormatter.print_state(state, f"{func_name} 输入状态")

        logger.debug(f"开始执行节点: {func_name}")

        try:
            result = func(state)
            execution_time = time.time() - start_time

            # 添加执行日志
            execution_log = result.get('execution_log', [])
            execution_log.append(f"{func_name}: 执行成功，耗时 {execution_time:.3f}s")
            result['execution_log'] = execution_log

            # 记录完成
            DebugFormatter.print_state(result, f"{func_name} 输出状态")
            DebugFormatter.print_execution_flow(func_name, result['step_count'], "完成")

            logger.debug(f"节点执行完成: {func_name}, 耗时: {execution_time:.3f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            # 记录错误
            DebugFormatter.print_execution_flow(func_name, state['step_count'], "错误")

            logger.error(f"节点执行失败: {func_name}, 错误: {str(e)}, 耗时: {execution_time:.3f}s")

            raise

    return wrapper

@debug_log
def input_validation_node(state: DebugState) -> DebugState:
    """输入验证节点"""
    input_data = state.get('input_data', '')

    # 添加调试检查点
    logger.debug(f"验证输入数据: {input_data}")

    if not input_data.strip():
        raise ValueError("输入数据不能为空")

    if len(input_data) > 1000:
        logger.warning(f"输入数据较长: {len(input_data)} 字符")

    debug_info = state.get('debug_info', {}).copy()
    debug_info.update({
        'validation_timestamp': datetime.now().isoformat(),
        'input_length': len(input_data),
        'validation_passed': True
    })

    return {
        **state,
        'debug_info': debug_info,
        'step_count': state['step_count'] + 1
    }

@debug_log
def data_processing_node(state: DebugState) -> DebugState:
    """数据处理节点"""
    input_data = state['input_data']

    # 模拟处理步骤
    logger.debug("开始数据处理...")

    # 步骤1：清理数据
    cleaned_data = input_data.strip()
    logger.debug(f"数据清理完成，长度从 {len(input_data)} 变为 {len(cleaned_data)}")

    # 步骤2：转换数据
    processed_data = cleaned_data.upper()
    logger.debug(f"数据转换完成: {processed_data[:50]}...")

    # 步骤3：添加处理标记
    final_data = f"[已处理] {processed_data}"

    # 更新调试信息
    debug_info = state.get('debug_info', {}).copy()
    debug_info.update({
        'processing_steps': ['clean', 'transform', 'mark'],
        'processing_timestamp': datetime.now().isoformat(),
        'original_length': len(input_data),
        'processed_length': len(final_data)
    })

    return {
        **state,
        'processed_data': final_data,
        'debug_info': debug_info,
        'step_count': state['step_count'] + 1
    }

@debug_log
def output_formatting_node(state: DebugState) -> DebugState:
    """输出格式化节点"""
    processed_data = state.get('processed_data', '')

    logger.debug("开始输出格式化...")

    # 创建格式化输出
    formatted_output = {
        'result': processed_data,
        'metadata': {
            'total_steps': state['step_count'] + 1,
            'processing_info': state.get('debug_info', {}),
            'execution_log': state.get('execution_log', [])
        }
    }

    # 模拟可能的错误情况
    if 'error' in processed_data.lower():
        logger.error("检测到错误关键词，触发调试断点")
        # 在实际调试时可以取消注释下面这行
        # import pdb; pdb.set_trace()

    debug_info = state.get('debug_info', {}).copy()
    debug_info.update({
        'formatting_timestamp': datetime.now().isoformat(),
        'output_format': 'json',
        'final_check': 'passed'
    })

    return {
        **state,
        'processed_data': json.dumps(formatted_output, indent=2, ensure_ascii=False),
        'debug_info': debug_info,
        'step_count': state['step_count'] + 1
    }

def create_debug_graph() -> StateGraph:
    """创建调试图"""
    graph = StateGraph(DebugState)

    # 添加节点
    graph.add_node("validate", input_validation_node)
    graph.add_node("process", data_processing_node)
    graph.add_node("format", output_formatting_node)

    # 定义执行流程
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "process")
    graph.add_edge("process", "format")
    graph.add_edge("format", END)

    return graph.compile()

def run_debug_example():
    """运行调试示例"""
    print("🔍 LangGraph 调试示例")
    print("="*50)

    # 创建图
    app = create_debug_graph()

    # 测试用例
    test_cases = [
        {
            "name": "正常输入",
            "input": "hello world"
        },
        {
            "name": "长文本输入",
            "input": "这是一个很长的文本输入，用来测试调试功能" * 10
        },
        {
            "name": "包含错误关键词",
            "input": "this contains an error"
        },
        {
            "name": "空输入测试",
            "input": ""
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {test_case['name']}")
        print("-" * 40)

        initial_state = {
            'input_data': test_case['input'],
            'processed_data': '',
            'step_count': 0,
            'debug_info': {'test_case': test_case['name']},
            'execution_log': []
        }

        try:
            # 使用流式执行进行调试
            print("📡 开始流式执行:")
            events = list(app.stream(initial_state))

            print(f"\n📊 流式事件总数: {len(events)}")
            for j, event in enumerate(events, 1):
                print(f"  事件 {j}: {list(event.keys())}")

            # 获取最终结果
            final_result = app.invoke(initial_state)

            print(f"\n✅ 测试完成!")
            print(f"📈 执行步骤数: {final_result['step_count']}")
            print(f"📋 执行日志: {len(final_result['execution_log'])} 条")

            # 显示最终输出（截取前200字符）
            output = final_result['processed_data']
            if len(output) > 200:
                output = output[:200] + "..."
            print(f"📤 最终输出: {output}")

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            logger.exception("测试执行出错")

        print("\n" + "="*50)

def demonstrate_state_inspection():
    """演示状态检查功能"""
    print("\n🔍 状态检查演示")
    print("="*30)

    app = create_debug_graph()

    initial_state = {
        'input_data': 'state inspection demo',
        'processed_data': '',
        'step_count': 0,
        'debug_info': {},
        'execution_log': []
    }

    print("📊 逐步状态检查:")

    # 模拟逐步执行
    current_state = initial_state
    node_sequence = ["validate", "process", "format"]

    for node_name in node_sequence:
        print(f"\n🎯 执行节点: {node_name}")

        # 这里模拟单节点执行（实际中可以用checkpointing）
        if node_name == "validate":
            current_state = input_validation_node(current_state)
        elif node_name == "process":
            current_state = data_processing_node(current_state)
        elif node_name == "format":
            current_state = output_formatting_node(current_state)

        # 显示当前状态
        DebugFormatter.print_state(current_state, f"执行 {node_name} 后的状态")

if __name__ == "__main__":
    # 运行主要的调试示例
    run_debug_example()

    # 演示状态检查
    demonstrate_state_inspection()

    print("\n🎉 调试示例演示完成!")
    print("💡 提示: 在实际调试时，可以：")
    print("   1. 取消注释 pdb.set_trace() 来设置断点")
    print("   2. 修改日志级别来控制输出详细程度")
    print("   3. 使用 LangGraph Studio 进行可视化调试")