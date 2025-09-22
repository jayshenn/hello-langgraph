#!/usr/bin/env python3
"""
LangGraph 测试示例代码

演示全面的测试策略，包括单元测试、集成测试、性能测试等。
"""

import unittest
import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import TypedDict, List, Dict, Any, Optional
from langgraph import StateGraph, START, END

# 测试状态定义
class TestState(TypedDict):
    input_data: str
    processed_data: str
    step_count: int
    metadata: Dict[str, Any]
    test_results: Dict[str, Any]

# 被测试的节点函数
def validation_node(state: TestState) -> TestState:
    """输入验证节点"""
    input_data = state.get('input_data', '')

    if not input_data.strip():
        raise ValueError("输入数据不能为空")

    if len(input_data) > 1000:
        raise ValueError("输入数据过长")

    return {
        **state,
        'step_count': state['step_count'] + 1,
        'metadata': {
            **state.get('metadata', {}),
            'validation_passed': True,
            'input_length': len(input_data)
        }
    }

def processing_node(state: TestState) -> TestState:
    """数据处理节点"""
    input_data = state['input_data']

    # 简单的数据处理
    processed = input_data.upper().strip()

    return {
        **state,
        'processed_data': f"[处理] {processed}",
        'step_count': state['step_count'] + 1,
        'metadata': {
            **state.get('metadata', {}),
            'processing_method': 'uppercase',
            'processed_length': len(processed)
        }
    }

def external_api_node(state: TestState) -> TestState:
    """外部API调用节点"""
    import requests

    try:
        # 模拟API调用
        response = requests.get(
            "https://httpbin.org/json",
            timeout=5
        )
        response.raise_for_status()

        api_data = response.json()

        return {
            **state,
            'step_count': state['step_count'] + 1,
            'test_results': {
                **state.get('test_results', {}),
                'api_call_success': True,
                'api_response': api_data
            }
        }
    except Exception as e:
        return {
            **state,
            'step_count': state['step_count'] + 1,
            'test_results': {
                **state.get('test_results', {}),
                'api_call_success': False,
                'api_error': str(e)
            }
        }

def conditional_router(state: TestState) -> str:
    """条件路由函数"""
    input_length = len(state.get('input_data', ''))

    if input_length > 50:
        return "long_processing"
    elif input_length > 10:
        return "medium_processing"
    else:
        return "short_processing"

def long_processing_node(state: TestState) -> TestState:
    """长文本处理节点"""
    time.sleep(0.1)  # 模拟处理时间

    return {
        **state,
        'processed_data': f"[长文本处理] {state['input_data'][:50]}...",
        'step_count': state['step_count'] + 1
    }

def medium_processing_node(state: TestState) -> TestState:
    """中等文本处理节点"""
    return {
        **state,
        'processed_data': f"[中等处理] {state['input_data']}",
        'step_count': state['step_count'] + 1
    }

def short_processing_node(state: TestState) -> TestState:
    """短文本处理节点"""
    return {
        **state,
        'processed_data': f"[短文本] {state['input_data']}",
        'step_count': state['step_count'] + 1
    }

# 图创建函数
def create_simple_graph() -> StateGraph:
    """创建简单的测试图"""
    graph = StateGraph(TestState)

    graph.add_node("validate", validation_node)
    graph.add_node("process", processing_node)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "process")
    graph.add_edge("process", END)

    return graph.compile()

def create_conditional_graph() -> StateGraph:
    """创建条件路由图"""
    graph = StateGraph(TestState)

    graph.add_node("validate", validation_node)
    graph.add_node("long_proc", long_processing_node)
    graph.add_node("medium_proc", medium_processing_node)
    graph.add_node("short_proc", short_processing_node)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate",
        conditional_router,
        {
            "long_processing": "long_proc",
            "medium_processing": "medium_proc",
            "short_processing": "short_proc"
        }
    )

    graph.add_edge("long_proc", END)
    graph.add_edge("medium_proc", END)
    graph.add_edge("short_proc", END)

    return graph.compile()

def create_api_graph() -> StateGraph:
    """创建包含API调用的图"""
    graph = StateGraph(TestState)

    graph.add_node("validate", validation_node)
    graph.add_node("api_call", external_api_node)
    graph.add_node("process", processing_node)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "api_call")
    graph.add_edge("api_call", "process")
    graph.add_edge("process", END)

    return graph.compile()

# 单元测试类
class TestNodeFunctions(unittest.TestCase):
    """节点函数单元测试"""

    def setUp(self):
        """测试前准备"""
        self.base_state = {
            'input_data': 'test input',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        }

    def test_validation_node_success(self):
        """测试验证节点成功情况"""
        result = validation_node(self.base_state)

        self.assertEqual(result['step_count'], 1)
        self.assertTrue(result['metadata']['validation_passed'])
        self.assertEqual(result['metadata']['input_length'], len('test input'))

    def test_validation_node_empty_input(self):
        """测试验证节点空输入"""
        state = {**self.base_state, 'input_data': ''}

        with self.assertRaises(ValueError) as context:
            validation_node(state)

        self.assertIn("不能为空", str(context.exception))

    def test_validation_node_long_input(self):
        """测试验证节点过长输入"""
        state = {**self.base_state, 'input_data': 'x' * 1001}

        with self.assertRaises(ValueError) as context:
            validation_node(state)

        self.assertIn("过长", str(context.exception))

    def test_processing_node(self):
        """测试处理节点"""
        result = processing_node(self.base_state)

        self.assertEqual(result['processed_data'], "[处理] TEST INPUT")
        self.assertEqual(result['step_count'], 1)
        self.assertEqual(result['metadata']['processing_method'], 'uppercase')

    def test_conditional_router(self):
        """测试条件路由函数"""
        # 短文本
        short_state = {**self.base_state, 'input_data': 'short'}
        self.assertEqual(conditional_router(short_state), "short_processing")

        # 中等文本
        medium_state = {**self.base_state, 'input_data': 'medium length text'}
        self.assertEqual(conditional_router(medium_state), "medium_processing")

        # 长文本
        long_state = {**self.base_state, 'input_data': 'x' * 60}
        self.assertEqual(conditional_router(long_state), "long_processing")

class TestGraphExecution(unittest.TestCase):
    """图执行集成测试"""

    def setUp(self):
        """测试前准备"""
        self.simple_app = create_simple_graph()
        self.conditional_app = create_conditional_graph()

        self.base_state = {
            'input_data': 'integration test',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        }

    def test_simple_graph_execution(self):
        """测试简单图执行"""
        result = self.simple_app.invoke(self.base_state)

        self.assertEqual(result['step_count'], 2)  # 两个节点
        self.assertIn("[处理]", result['processed_data'])
        self.assertTrue(result['metadata']['validation_passed'])

    def test_conditional_graph_short_text(self):
        """测试条件图短文本路径"""
        state = {**self.base_state, 'input_data': 'short'}
        result = self.conditional_app.invoke(state)

        self.assertEqual(result['step_count'], 2)  # 验证 + 短文本处理
        self.assertIn("[短文本]", result['processed_data'])

    def test_conditional_graph_long_text(self):
        """测试条件图长文本路径"""
        long_text = 'x' * 60
        state = {**self.base_state, 'input_data': long_text}
        result = self.conditional_app.invoke(state)

        self.assertEqual(result['step_count'], 2)  # 验证 + 长文本处理
        self.assertIn("[长文本处理]", result['processed_data'])

    def test_graph_streaming(self):
        """测试图流式执行"""
        events = list(self.simple_app.stream(self.base_state))

        self.assertGreater(len(events), 0)

        # 验证事件包含预期的节点
        node_names = []
        for event in events:
            node_names.extend(event.keys())

        self.assertIn("validate", node_names)
        self.assertIn("process", node_names)

class TestAPIIntegration(unittest.TestCase):
    """API集成测试"""

    def setUp(self):
        """测试前准备"""
        self.base_state = {
            'input_data': 'api test',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        }

    @patch('requests.get')
    def test_external_api_success(self, mock_get):
        """测试API调用成功"""
        # 设置模拟响应
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = external_api_node(self.base_state)

        # 验证API被调用
        mock_get.assert_called_once()

        # 验证结果
        self.assertTrue(result['test_results']['api_call_success'])
        self.assertEqual(result['test_results']['api_response'], {"test": "data"})

    @patch('requests.get')
    def test_external_api_failure(self, mock_get):
        """测试API调用失败"""
        # 设置模拟异常
        mock_get.side_effect = Exception("Network error")

        result = external_api_node(self.base_state)

        # 验证错误处理
        self.assertFalse(result['test_results']['api_call_success'])
        self.assertIn("Network error", result['test_results']['api_error'])

    @patch('requests.get')
    def test_api_graph_execution(self, mock_get):
        """测试包含API的完整图执行"""
        # 设置模拟
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        app = create_api_graph()
        result = app.invoke(self.base_state)

        # 验证完整流程
        self.assertEqual(result['step_count'], 3)  # 三个节点
        self.assertTrue(result['test_results']['api_call_success'])
        self.assertIn("[处理]", result['processed_data'])

class TestPerformance(unittest.TestCase):
    """性能测试"""

    def setUp(self):
        """测试前准备"""
        self.app = create_simple_graph()
        self.base_state = {
            'input_data': 'performance test',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        }

    def test_execution_time(self):
        """测试执行时间"""
        start_time = time.time()
        result = self.app.invoke(self.base_state)
        execution_time = time.time() - start_time

        # 验证执行时间在合理范围内
        self.assertLess(execution_time, 1.0)  # 应该在1秒内完成
        self.assertEqual(result['step_count'], 2)

    def test_concurrent_execution(self):
        """测试并发执行"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def single_execution():
            return self.app.invoke(self.base_state)

        # 并发执行多次
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(single_execution) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # 验证所有执行都成功
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertEqual(result['step_count'], 2)

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import gc

        # 强制垃圾回收
        gc.collect()

        # 记录初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # 执行多次
        for _ in range(100):
            self.app.invoke(self.base_state)

        # 强制垃圾回收
        gc.collect()

        # 检查内存增长
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # 内存增长应该在合理范围内（小于10MB）
        self.assertLess(memory_growth, 10 * 1024 * 1024)

# Pytest 测试函数
@pytest.mark.parametrize("input_data,expected_route", [
    ("short", "short_processing"),
    ("medium length text", "medium_processing"),
    ("x" * 60, "long_processing")
])
def test_conditional_routing_parametrized(input_data, expected_route):
    """参数化测试条件路由"""
    state = {
        'input_data': input_data,
        'processed_data': '',
        'step_count': 0,
        'metadata': {},
        'test_results': {}
    }

    route = conditional_router(state)
    assert route == expected_route

@pytest.mark.asyncio
async def test_async_graph_execution():
    """测试异步图执行"""
    app = create_simple_graph()

    state = {
        'input_data': 'async test',
        'processed_data': '',
        'step_count': 0,
        'metadata': {},
        'test_results': {}
    }

    result = await app.ainvoke(state)

    assert result['step_count'] == 2
    assert "[处理]" in result['processed_data']

class TestDataValidation:
    """数据验证测试"""

    @pytest.fixture
    def validation_test_cases(self):
        """测试用例夹具"""
        return [
            {
                "input": "valid input",
                "should_pass": True,
                "description": "正常输入"
            },
            {
                "input": "",
                "should_pass": False,
                "description": "空输入"
            },
            {
                "input": "   ",
                "should_pass": False,
                "description": "只有空格"
            },
            {
                "input": "x" * 1001,
                "should_pass": False,
                "description": "过长输入"
            }
        ]

    def test_validation_cases(self, validation_test_cases):
        """测试验证用例"""
        for case in validation_test_cases:
            state = {
                'input_data': case['input'],
                'processed_data': '',
                'step_count': 0,
                'metadata': {},
                'test_results': {}
            }

            if case['should_pass']:
                result = validation_node(state)
                assert result['metadata']['validation_passed']
            else:
                with pytest.raises(ValueError):
                    validation_node(state)

# 测试套件运行函数
def run_all_tests():
    """运行所有测试"""
    print("🧪 运行 LangGraph 测试套件")
    print("="*40)

    # 运行 unittest
    print("\n📋 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # 运行 pytest（如果安装了）
    try:
        import pytest
        print("\n🔬 运行 pytest 测试...")
        pytest.main(["-v", __file__])
    except ImportError:
        print("⚠️  pytest 未安装，跳过 pytest 测试")

# 测试报告生成
class TestReporter:
    """测试报告生成器"""

    def __init__(self):
        self.results = []

    def add_result(self, test_name: str, passed: bool, execution_time: float, details: str = ""):
        """添加测试结果"""
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "execution_time": execution_time,
            "details": details,
            "timestamp": time.time()
        })

    def generate_report(self, filename: str = "test_report.json"):
        """生成测试报告"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        total_time = sum(r['execution_time'] for r in self.results)

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time
            },
            "results": self.results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 测试报告已生成: {filename}")
        return report

def demo_comprehensive_testing():
    """演示综合测试"""
    print("🎯 综合测试演示")
    print("="*20)

    reporter = TestReporter()

    # 测试项目列表
    test_items = [
        ("简单图执行", lambda: create_simple_graph().invoke({
            'input_data': 'demo test',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        })),
        ("条件路由", lambda: conditional_router({'input_data': 'test'})),
        ("验证节点", lambda: validation_node({
            'input_data': 'valid input',
            'processed_data': '',
            'step_count': 0,
            'metadata': {},
            'test_results': {}
        }))
    ]

    for test_name, test_func in test_items:
        try:
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time

            reporter.add_result(test_name, True, execution_time, "测试通过")
            print(f"✅ {test_name}: 通过 ({execution_time:.3f}s)")

        except Exception as e:
            execution_time = time.time() - start_time
            reporter.add_result(test_name, False, execution_time, str(e))
            print(f"❌ {test_name}: 失败 - {e}")

    # 生成报告
    report = reporter.generate_report()

    print(f"\n📈 测试摘要:")
    print(f"   总测试数: {report['summary']['total_tests']}")
    print(f"   通过: {report['summary']['passed']}")
    print(f"   失败: {report['summary']['failed']}")
    print(f"   成功率: {report['summary']['success_rate']:.1%}")
    print(f"   总耗时: {report['summary']['total_execution_time']:.3f}s")

if __name__ == "__main__":
    # 演示综合测试
    demo_comprehensive_testing()

    print("\n💡 测试要点:")
    print("   1. 单元测试覆盖所有节点函数")
    print("   2. 集成测试验证完整流程")
    print("   3. 使用Mock测试外部依赖")
    print("   4. 参数化测试提高覆盖率")
    print("   5. 性能测试确保系统效率")
    print("   6. 生成详细的测试报告")

    print("\n🚀 运行完整测试套件:")
    print("   python -m unittest discover")
    print("   pytest test_graph.py -v")