#!/usr/bin/env python3
"""
LangGraph 性能测试示例代码

演示性能优化技巧和性能测试方法。
"""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph import StateGraph, START, END
import time
import asyncio
import threading
import psutil
import gc
from operator import add
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json

# 性能测试状态
class PerformanceState(TypedDict):
    input_data: str
    results: Annotated[List[str], add]  # 使用 reducer 支持并行
    performance_metrics: Dict[str, Any]
    execution_count: int

@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    node_name: str
    timestamp: str

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_memory = None

    def start_monitoring(self):
        """开始监控"""
        gc.collect()  # 强制垃圾回收
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024

    def record_metrics(self, node_name: str, execution_time: float):
        """记录性能指标"""
        current_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_usage = current_memory - (self.start_memory or current_memory)

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_percent=psutil.cpu_percent(),
            node_name=node_name,
            timestamp=datetime.now().isoformat()
        )

        self.metrics.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics:
            return {}

        total_time = sum(m.execution_time for m in self.metrics)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics)
        max_memory = max(m.memory_usage_mb for m in self.metrics)

        return {
            "total_execution_time": total_time,
            "average_memory_usage_mb": avg_memory,
            "peak_memory_usage_mb": max_memory,
            "total_measurements": len(self.metrics),
            "node_performance": self._get_node_performance()
        }

    def _get_node_performance(self) -> Dict[str, Dict[str, float]]:
        """获取节点性能分析"""
        node_metrics = {}

        for metric in self.metrics:
            if metric.node_name not in node_metrics:
                node_metrics[metric.node_name] = {
                    "total_time": 0,
                    "count": 0,
                    "max_time": 0,
                    "min_time": float('inf')
                }

            node_data = node_metrics[metric.node_name]
            node_data["total_time"] += metric.execution_time
            node_data["count"] += 1
            node_data["max_time"] = max(node_data["max_time"], metric.execution_time)
            node_data["min_time"] = min(node_data["min_time"], metric.execution_time)

        # 计算平均值
        for node_name, data in node_metrics.items():
            data["avg_time"] = data["total_time"] / data["count"]

        return node_metrics

# 全局性能监控器
perf_monitor = PerformanceMonitor()

def performance_test(node_name: str):
    """性能测试装饰器"""
    def decorator(func):
        def wrapper(state):
            start_time = time.time()

            try:
                result = func(state)
                execution_time = time.time() - start_time

                # 记录性能指标
                metrics = perf_monitor.record_metrics(node_name, execution_time)

                # 添加性能数据到状态
                performance_metrics = result.get('performance_metrics', {})
                performance_metrics[node_name] = {
                    "execution_time": execution_time,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "timestamp": metrics.timestamp
                }
                result['performance_metrics'] = performance_metrics

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                perf_monitor.record_metrics(f"{node_name}_failed", execution_time)
                raise

        return wrapper
    return decorator

@performance_test("cpu_intensive")
def cpu_intensive_node(state: PerformanceState) -> PerformanceState:
    """CPU密集型节点"""
    print(f"🔄 CPU密集型处理开始...")

    # 模拟CPU密集型操作
    result = 0
    iterations = 1000000

    for i in range(iterations):
        result += i * i

    # 模拟更复杂的计算
    data = state.get('input_data', '')
    processed_result = f"CPU处理结果: {result}, 数据长度: {len(data)}"

    return {
        **state,
        "results": [processed_result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("optimized_cpu")
def optimized_cpu_node(state: PerformanceState) -> PerformanceState:
    """优化后的CPU密集型节点"""
    print(f"⚡ 优化CPU处理开始...")

    # 使用数学公式代替循环
    n = 1000000
    result = n * (n - 1) * (2 * n - 1) // 6  # 平方和公式

    data = state.get('input_data', '')
    processed_result = f"优化CPU处理结果: {result}, 数据长度: {len(data)}"

    return {
        **state,
        "results": [processed_result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("io_intensive")
def io_intensive_node(state: PerformanceState) -> PerformanceState:
    """IO密集型节点"""
    print(f"💾 IO密集型处理开始...")

    # 模拟IO操作
    time.sleep(0.5)  # 模拟网络IO

    # 模拟文件操作
    import tempfile
    import os

    temp_data = state.get('input_data', '') * 1000
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(temp_data)
        temp_file = f.name

    # 读取文件
    with open(temp_file, 'r') as f:
        file_content = f.read()

    # 清理
    os.unlink(temp_file)

    processed_result = f"IO处理结果: 处理了 {len(file_content)} 字符"

    return {
        **state,
        "results": [processed_result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("parallel_a")
def parallel_node_a(state: PerformanceState) -> PerformanceState:
    """并行节点 A"""
    thread_id = threading.current_thread().ident
    print(f"🅰️  并行节点A开始 (线程: {thread_id})")

    # 模拟处理
    time.sleep(1)

    result = f"并行A结果 (线程: {thread_id})"

    return {
        **state,
        "results": [result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("parallel_b")
def parallel_node_b(state: PerformanceState) -> PerformanceState:
    """并行节点 B"""
    thread_id = threading.current_thread().ident
    print(f"🅱️  并行节点B开始 (线程: {thread_id})")

    # 模拟不同的处理时间
    time.sleep(0.8)

    result = f"并行B结果 (线程: {thread_id})"

    return {
        **state,
        "results": [result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("memory_intensive")
def memory_intensive_node(state: PerformanceState) -> PerformanceState:
    """内存密集型节点"""
    print(f"🧠 内存密集型处理开始...")

    # 创建大量数据
    large_data = []
    for i in range(100000):
        large_data.append(f"数据项_{i}_" + "x" * 100)

    # 处理数据
    processed_count = len(large_data)
    processed_result = f"内存处理结果: 处理了 {processed_count} 个数据项"

    # 主动清理大数据（演示内存管理）
    del large_data
    gc.collect()

    return {
        **state,
        "results": [processed_result],
        "execution_count": state.get("execution_count", 0) + 1
    }

@performance_test("streaming")
def streaming_node(state: PerformanceState) -> PerformanceState:
    """流式处理节点"""
    print(f"📡 流式处理开始...")

    # 模拟流式处理
    chunks = []
    input_data = state.get('input_data', '')

    # 分块处理
    chunk_size = 10
    for i in range(0, len(input_data), chunk_size):
        chunk = input_data[i:i+chunk_size]
        processed_chunk = f"流式块_{i//chunk_size}: {chunk}"
        chunks.append(processed_chunk)

        # 模拟流式延迟
        time.sleep(0.01)

    processed_result = f"流式处理结果: {len(chunks)} 个块"

    return {
        **state,
        "results": [processed_result],
        "execution_count": state.get("execution_count", 0) + 1
    }

def create_performance_test_graphs():
    """创建性能测试图"""
    graphs = {}

    # 顺序处理图
    sequential_graph = StateGraph(PerformanceState)
    sequential_graph.add_node("cpu", cpu_intensive_node)
    sequential_graph.add_node("io", io_intensive_node)
    sequential_graph.add_node("memory", memory_intensive_node)

    sequential_graph.add_edge(START, "cpu")
    sequential_graph.add_edge("cpu", "io")
    sequential_graph.add_edge("io", "memory")
    sequential_graph.add_edge("memory", END)

    graphs["sequential"] = sequential_graph.compile()

    # 并行处理图
    parallel_graph = StateGraph(PerformanceState)
    parallel_graph.add_node("parallel_a", parallel_node_a)
    parallel_graph.add_node("parallel_b", parallel_node_b)

    parallel_graph.add_edge(START, "parallel_a")
    parallel_graph.add_edge(START, "parallel_b")
    parallel_graph.add_edge("parallel_a", END)
    parallel_graph.add_edge("parallel_b", END)

    graphs["parallel"] = parallel_graph.compile()

    # 优化对比图
    optimization_graph = StateGraph(PerformanceState)
    optimization_graph.add_node("original", cpu_intensive_node)
    optimization_graph.add_node("optimized", optimized_cpu_node)

    optimization_graph.add_edge(START, "original")
    optimization_graph.add_edge("original", "optimized")
    optimization_graph.add_edge("optimized", END)

    graphs["optimization"] = optimization_graph.compile()

    # 流式处理图
    streaming_graph = StateGraph(PerformanceState)
    streaming_graph.add_node("streaming", streaming_node)

    streaming_graph.add_edge(START, "streaming")
    streaming_graph.add_edge("streaming", END)

    graphs["streaming"] = streaming_graph.compile()

    return graphs

def run_performance_benchmarks():
    """运行性能基准测试"""
    print("🚀 性能基准测试")
    print("="*40)

    graphs = create_performance_test_graphs()

    test_data = {
        'input_data': 'performance test data ' * 100,
        'results': [],
        'performance_metrics': {},
        'execution_count': 0
    }

    benchmarks = [
        ("顺序处理", "sequential"),
        ("并行处理", "parallel"),
        ("优化对比", "optimization"),
        ("流式处理", "streaming")
    ]

    for name, graph_type in benchmarks:
        print(f"\n📊 测试: {name}")
        print("-" * 30)

        perf_monitor.start_monitoring()
        start_time = time.time()

        try:
            result = graphs[graph_type].invoke(test_data.copy())
            total_time = time.time() - start_time

            print(f"✅ 执行完成")
            print(f"   总时间: {total_time:.3f}s")
            print(f"   结果数: {len(result['results'])}")
            print(f"   执行次数: {result['execution_count']}")

            # 显示性能指标
            if result['performance_metrics']:
                print(f"   节点性能:")
                for node, metrics in result['performance_metrics'].items():
                    print(f"     {node}: {metrics['execution_time']:.3f}s")

        except Exception as e:
            print(f"❌ 测试失败: {e}")

    # 显示总体性能摘要
    print(f"\n📈 性能摘要")
    print("="*20)
    summary = perf_monitor.get_summary()

    if summary:
        print(f"总执行时间: {summary['total_execution_time']:.3f}s")
        print(f"平均内存使用: {summary['average_memory_usage_mb']:.2f}MB")
        print(f"峰值内存使用: {summary['peak_memory_usage_mb']:.2f}MB")
        print(f"总测量次数: {summary['total_measurements']}")

        print(f"\n节点性能分析:")
        for node, perf in summary['node_performance'].items():
            print(f"  {node}:")
            print(f"    平均时间: {perf['avg_time']:.3f}s")
            print(f"    最大时间: {perf['max_time']:.3f}s")
            print(f"    最小时间: {perf['min_time']:.3f}s")
            print(f"    执行次数: {perf['count']}")

def run_concurrent_stress_test():
    """运行并发压力测试"""
    print(f"\n🔥 并发压力测试")
    print("="*25)

    graphs = create_performance_test_graphs()
    app = graphs["parallel"]

    def single_execution(index: int):
        """单次执行"""
        test_data = {
            'input_data': f'concurrent test {index}',
            'results': [],
            'performance_metrics': {},
            'execution_count': 0
        }

        start_time = time.time()
        result = app.invoke(test_data)
        execution_time = time.time() - start_time

        return {
            'index': index,
            'execution_time': execution_time,
            'result_count': len(result['results'])
        }

    # 并发测试
    num_workers = 5
    num_tasks = 20

    print(f"启动 {num_workers} 个工作线程，执行 {num_tasks} 个任务")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_execution, i) for i in range(num_tasks)]
        results = [future.result() for future in as_completed(futures)]

    total_time = time.time() - start_time

    # 分析结果
    execution_times = [r['execution_time'] for r in results]
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)
    min_time = min(execution_times)

    print(f"\n📊 并发测试结果:")
    print(f"   总时间: {total_time:.3f}s")
    print(f"   平均单次时间: {avg_time:.3f}s")
    print(f"   最长单次时间: {max_time:.3f}s")
    print(f"   最短单次时间: {min_time:.3f}s")
    print(f"   吞吐量: {num_tasks/total_time:.2f} tasks/s")

def run_memory_leak_test():
    """运行内存泄漏测试"""
    print(f"\n🧪 内存泄漏测试")
    print("="*20)

    graph = create_performance_test_graphs()["streaming"]

    memory_readings = []
    num_iterations = 50

    print(f"执行 {num_iterations} 次迭代，监控内存使用...")

    for i in range(num_iterations):
        # 强制垃圾回收
        gc.collect()

        # 记录内存使用
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        memory_readings.append(memory_mb)

        # 执行图
        test_data = {
            'input_data': f'memory test iteration {i}' * 50,
            'results': [],
            'performance_metrics': {},
            'execution_count': 0
        }

        result = graph.invoke(test_data)

        if i % 10 == 0:
            print(f"   迭代 {i}: 内存使用 {memory_mb:.2f}MB")

    # 分析内存趋势
    start_memory = memory_readings[0]
    end_memory = memory_readings[-1]
    max_memory = max(memory_readings)
    memory_growth = end_memory - start_memory

    print(f"\n📈 内存分析:")
    print(f"   起始内存: {start_memory:.2f}MB")
    print(f"   结束内存: {end_memory:.2f}MB")
    print(f"   峰值内存: {max_memory:.2f}MB")
    print(f"   内存增长: {memory_growth:+.2f}MB")

    if memory_growth > 10:  # 10MB阈值
        print(f"   ⚠️  警告: 可能存在内存泄漏")
    else:
        print(f"   ✅ 内存使用正常")

async def run_async_performance_test():
    """运行异步性能测试"""
    print(f"\n⚡ 异步性能测试")
    print("="*20)

    async def async_task(index: int, delay: float):
        """异步任务"""
        await asyncio.sleep(delay)
        return f"异步任务 {index} 完成"

    # 对比同步和异步执行
    tasks = [(i, 0.1) for i in range(20)]

    # 同步执行
    print("🔄 同步执行...")
    start_time = time.time()
    sync_results = []
    for index, delay in tasks:
        time.sleep(delay)
        sync_results.append(f"同步任务 {index} 完成")
    sync_time = time.time() - start_time

    # 异步执行
    print("⚡ 异步执行...")
    start_time = time.time()
    async_tasks = [async_task(index, delay) for index, delay in tasks]
    async_results = await asyncio.gather(*async_tasks)
    async_time = time.time() - start_time

    print(f"\n📊 性能对比:")
    print(f"   同步执行时间: {sync_time:.3f}s")
    print(f"   异步执行时间: {async_time:.3f}s")
    print(f"   性能提升: {sync_time/async_time:.2f}x")

def generate_performance_report():
    """生成性能报告"""
    print(f"\n📄 生成性能报告")
    print("="*20)

    summary = perf_monitor.get_summary()

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "metrics": [
            {
                "node_name": m.node_name,
                "execution_time": m.execution_time,
                "memory_usage_mb": m.memory_usage_mb,
                "cpu_percent": m.cpu_percent,
                "timestamp": m.timestamp
            }
            for m in perf_monitor.metrics
        ]
    }

    # 保存报告
    report_file = f"performance_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📊 性能报告已保存: {report_file}")

if __name__ == "__main__":
    print("🚀 LangGraph 性能测试套件")
    print("="*50)

    # 运行各种性能测试
    run_performance_benchmarks()

    # 并发压力测试
    run_concurrent_stress_test()

    # 内存泄漏测试
    run_memory_leak_test()

    # 异步性能测试
    print("⚡ 启动异步测试...")
    asyncio.run(run_async_performance_test())

    # 生成性能报告
    generate_performance_report()

    print("\n🎉 性能测试完成!")
    print("💡 优化建议:")
    print("   1. CPU密集型任务使用算法优化")
    print("   2. IO密集型任务考虑异步处理")
    print("   3. 并行处理提升吞吐量")
    print("   4. 及时清理大对象避免内存泄漏")
    print("   5. 使用流式处理优化用户体验")