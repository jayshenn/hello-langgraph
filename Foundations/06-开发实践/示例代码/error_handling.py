#!/usr/bin/env python3
"""
LangGraph 错误处理示例代码

演示各种错误处理策略和技巧。
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph import StateGraph, START, END
import time
import random
import logging
from functools import wraps
from datetime import datetime
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorState(TypedDict):
    """错误处理示例状态"""
    input_data: str
    processed_data: str
    errors: List[str]
    warnings: List[str]
    retry_count: int
    success: bool
    metadata: Dict[str, Any]

class ErrorType(Enum):
    """错误类型枚举"""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"

class ErrorTracker:
    """错误追踪器"""

    def __init__(self):
        self.errors = []
        self.recovery_attempts = {}

    def log_error(self, node_name: str, error: Exception, state: dict):
        """记录错误"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "state_snapshot": {k: str(v)[:100] for k, v in state.items()}
        }

        self.errors.append(error_info)
        logger.error(f"🚨 错误已记录: {node_name} - {error_info['error_type']}: {error_info['error_message']}")

    def attempt_recovery(self, node_name: str, state: dict, max_attempts: int = 3):
        """尝试错误恢复"""
        attempt_key = f"{node_name}_{hash(str(state))}"
        attempts = self.recovery_attempts.get(attempt_key, 0)

        if attempts >= max_attempts:
            logger.error(f"💀 恢复失败: {node_name} 已达到最大重试次数 {max_attempts}")
            return None

        self.recovery_attempts[attempt_key] = attempts + 1
        logger.info(f"🔄 尝试恢复: {node_name} (第 {attempts + 1} 次)")

        return state

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.errors:
            return {"total_errors": 0}

        error_types = {}
        node_errors = {}

        for error in self.errors:
            error_type = error["error_type"]
            node_name = error["node_name"]

            error_types[error_type] = error_types.get(error_type, 0) + 1
            node_errors[node_name] = node_errors.get(node_name, 0) + 1

        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "node_errors": node_errors,
            "recent_errors": self.errors[-5:]  # 最近5个错误
        }

# 全局错误追踪器
error_tracker = ErrorTracker()

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: ErrorState) -> ErrorState:
            last_exception = None
            retry_count = state.get('retry_count', 0)

            for attempt in range(max_retries + 1):
                try:
                    # 执行函数
                    result = func(state)

                    # 成功时重置重试计数
                    if 'retry_count' in result:
                        result['retry_count'] = 0

                    return result

                except Exception as e:
                    last_exception = e
                    retry_count += 1

                    if attempt == max_retries:
                        logger.error(f"❌ 最终失败 (尝试 {attempt + 1} 次): {e}")

                        # 记录到错误追踪器
                        error_tracker.log_error(func.__name__, e, state)

                        # 返回错误状态而不是抛出异常
                        errors = state.get('errors', [])
                        errors.append(f"{func.__name__}: {str(e)} (重试{max_retries}次后失败)")

                        return {
                            **state,
                            'errors': errors,
                            'retry_count': retry_count,
                            'success': False
                        }

                    # 计算退避时间
                    backoff_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"🔄 重试 {attempt + 1}/{max_retries}，等待 {backoff_time:.2f}s: {e}")
                    time.sleep(backoff_time)

                    # 更新状态中的重试计数
                    state = {**state, 'retry_count': retry_count}

            raise last_exception

        return wrapper
    return decorator

def safe_execution(func):
    """安全执行装饰器"""
    @wraps(func)
    def wrapper(state: ErrorState) -> ErrorState:
        try:
            # 检查是否已有严重错误
            if state.get('errors') and any('严重' in error for error in state['errors']):
                logger.warning(f"⚠️  跳过 {func.__name__}: 存在严重错误")
                return state

            return func(state)

        except Exception as e:
            # 安全的错误处理
            logger.error(f"💥 {func.__name__} 执行出错: {e}")

            errors = state.get('errors', [])
            errors.append(f"{func.__name__}: {str(e)}")

            return {
                **state,
                'errors': errors,
                'success': False,
                'metadata': {
                    **state.get('metadata', {}),
                    f'{func.__name__}_failed': True,
                    f'{func.__name__}_error': str(e)
                }
            }

    return wrapper

@retry_with_backoff(max_retries=2, backoff_factor=0.5)
@safe_execution
def unreliable_validation_node(state: ErrorState) -> ErrorState:
    """不可靠的验证节点（用于演示重试）"""
    input_data = state.get('input_data', '')

    # 模拟随机失败
    if random.random() < 0.6:  # 60% 失败率
        raise ValueError(f"验证失败: 随机错误 (输入: {input_data[:20]}...)")

    # 模拟验证逻辑
    if not input_data.strip():
        raise ValueError("输入数据不能为空")

    if len(input_data) > 1000:
        warnings = state.get('warnings', [])
        warnings.append("输入数据较长，可能影响性能")
        return {
            **state,
            'warnings': warnings,
            'success': True
        }

    logger.info(f"✅ 验证成功: {input_data[:50]}...")
    return {
        **state,
        'success': True,
        'metadata': {
            **state.get('metadata', {}),
            'validation_passed': True,
            'input_length': len(input_data)
        }
    }

@safe_execution
def robust_processing_node(state: ErrorState) -> ErrorState:
    """健壮的处理节点"""
    if not state.get('success', False):
        logger.warning("⚠️  前置验证未通过，尝试降级处理")

    input_data = state.get('input_data', '')

    try:
        # 尝试正常处理
        if 'error' in input_data.lower():
            raise ValueError("输入包含错误关键词")

        if 'timeout' in input_data.lower():
            # 模拟超时
            time.sleep(5)

        # 正常处理逻辑
        processed_data = f"[已处理] {input_data.upper()}"

        return {
            **state,
            'processed_data': processed_data,
            'success': True,
            'metadata': {
                **state.get('metadata', {}),
                'processing_method': 'normal',
                'processing_success': True
            }
        }

    except ValueError as e:
        # 特定错误的处理
        logger.warning(f"⚠️  处理错误，使用降级方案: {e}")

        warnings = state.get('warnings', [])
        warnings.append(f"处理降级: {str(e)}")

        # 降级处理
        processed_data = f"[降级处理] {input_data}"

        return {
            **state,
            'processed_data': processed_data,
            'warnings': warnings,
            'success': True,
            'metadata': {
                **state.get('metadata', {}),
                'processing_method': 'fallback',
                'processing_success': True,
                'fallback_reason': str(e)
            }
        }

    except Exception as e:
        # 通用错误处理
        logger.error(f"❌ 处理失败: {e}")

        errors = state.get('errors', [])
        errors.append(f"处理失败: {str(e)}")

        return {
            **state,
            'errors': errors,
            'success': False,
            'processed_data': '[处理失败]',
            'metadata': {
                **state.get('metadata', {}),
                'processing_method': 'failed',
                'processing_success': False
            }
        }

@safe_execution
def error_recovery_node(state: ErrorState) -> ErrorState:
    """错误恢复节点"""
    errors = state.get('errors', [])

    if not errors:
        logger.info("✅ 无需恢复，继续正常流程")
        return state

    logger.info(f"🔧 尝试恢复 {len(errors)} 个错误")

    recovered_errors = []
    remaining_errors = []

    for error in errors:
        if '随机错误' in error:
            # 可以恢复的错误
            logger.info(f"🔄 恢复错误: {error}")
            recovered_errors.append(error)
        else:
            # 无法恢复的错误
            remaining_errors.append(error)

    if recovered_errors:
        warnings = state.get('warnings', [])
        warnings.extend([f"已恢复: {error}" for error in recovered_errors])

        return {
            **state,
            'errors': remaining_errors,
            'warnings': warnings,
            'success': len(remaining_errors) == 0,
            'metadata': {
                **state.get('metadata', {}),
                'recovered_errors': len(recovered_errors),
                'remaining_errors': len(remaining_errors)
            }
        }

    return state

def error_condition(state: ErrorState) -> str:
    """错误路由条件"""
    errors = state.get('errors', [])

    if not errors:
        return "success"

    # 检查错误类型
    for error in errors:
        if '严重' in error or '致命' in error:
            return "critical_error"

    return "recoverable_error"

def create_error_handling_graph() -> StateGraph:
    """创建错误处理图"""
    graph = StateGraph(ErrorState)

    # 添加节点
    graph.add_node("validate", unreliable_validation_node)
    graph.add_node("process", robust_processing_node)
    graph.add_node("recover", error_recovery_node)

    # 定义边
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "process")

    # 条件路由
    graph.add_conditional_edges(
        "process",
        error_condition,
        {
            "success": END,
            "recoverable_error": "recover",
            "critical_error": END
        }
    )

    graph.add_edge("recover", END)

    return graph.compile()

def test_error_scenarios():
    """测试各种错误场景"""
    print("🧪 错误处理测试")
    print("="*40)

    app = create_error_handling_graph()

    test_cases = [
        {
            "name": "正常输入",
            "input": "normal processing test"
        },
        {
            "name": "触发处理错误",
            "input": "this will cause an error"
        },
        {
            "name": "触发超时",
            "input": "timeout test case"
        },
        {
            "name": "空输入",
            "input": ""
        },
        {
            "name": "长文本输入",
            "input": "long text " * 200
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 测试 {i}: {test_case['name']}")
        print("-" * 30)

        initial_state = {
            'input_data': test_case['input'],
            'processed_data': '',
            'errors': [],
            'warnings': [],
            'retry_count': 0,
            'success': False,
            'metadata': {'test_case': test_case['name']}
        }

        try:
            result = app.invoke(initial_state)

            # 显示结果
            print(f"📊 执行结果:")
            print(f"   成功: {result['success']}")
            print(f"   错误数: {len(result['errors'])}")
            print(f"   警告数: {len(result['warnings'])}")
            print(f"   重试次数: {result['retry_count']}")

            if result['errors']:
                print(f"   错误详情: {result['errors']}")

            if result['warnings']:
                print(f"   警告详情: {result['warnings']}")

            # 显示处理结果
            processed = result['processed_data']
            if processed:
                print(f"   处理结果: {processed[:100]}...")

        except Exception as e:
            print(f"❌ 测试失败: {e}")

        print()

def demonstrate_circuit_breaker():
    """演示断路器模式"""
    print("\n⚡ 断路器模式演示")
    print("="*30)

    class CircuitBreaker:
        def __init__(self, failure_threshold: int = 3, timeout: int = 5):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        def call(self, func, *args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    print("🔄 断路器进入半开状态")
                else:
                    raise Exception("💥 断路器开启，拒绝调用")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

        def _on_success(self):
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                print("✅ 断路器恢复到关闭状态")

        def _on_failure(self):
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print(f"⚡ 断路器开启 (失败 {self.failure_count} 次)")

    def unreliable_service(data):
        """不可靠的服务"""
        if random.random() < 0.7:  # 70% 失败率
            raise Exception("服务调用失败")
        return f"服务处理结果: {data}"

    # 创建断路器
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=3)

    # 测试断路器
    for i in range(10):
        try:
            result = circuit_breaker.call(unreliable_service, f"数据{i}")
            print(f"✅ 调用 {i+1} 成功: {result}")
        except Exception as e:
            print(f"❌ 调用 {i+1} 失败: {e}")

        time.sleep(1)

def show_error_summary():
    """显示错误摘要"""
    print("\n📊 错误追踪摘要")
    print("="*25)

    summary = error_tracker.get_error_summary()

    if summary['total_errors'] == 0:
        print("✅ 无错误记录")
        return

    print(f"总错误数: {summary['total_errors']}")

    if 'error_types' in summary:
        print("\n错误类型分布:")
        for error_type, count in summary['error_types'].items():
            print(f"  {error_type}: {count}")

    if 'node_errors' in summary:
        print("\n节点错误分布:")
        for node, count in summary['node_errors'].items():
            print(f"  {node}: {count}")

    if summary.get('recent_errors'):
        print("\n最近错误:")
        for error in summary['recent_errors']:
            print(f"  [{error['timestamp']}] {error['node_name']}: {error['error_message']}")

if __name__ == "__main__":
    # 测试错误处理场景
    test_error_scenarios()

    # 演示断路器模式
    demonstrate_circuit_breaker()

    # 显示错误摘要
    show_error_summary()

    print("\n🎉 错误处理示例演示完成!")
    print("💡 关键要点:")
    print("   1. 使用重试机制处理临时错误")
    print("   2. 实现降级策略保证系统可用性")
    print("   3. 记录和追踪错误便于调试")
    print("   4. 使用断路器防止级联失败")