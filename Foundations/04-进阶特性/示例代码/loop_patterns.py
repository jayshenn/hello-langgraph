#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环结构示例代码
演示 LangGraph 中各种循环模式的实现
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time
import random


class LoopState(TypedDict):
    """循环状态定义"""
    task_name: str
    current_iteration: int
    max_iterations: int
    results: List[dict]
    error_count: int
    success_rate: float
    should_continue: bool
    performance_metrics: dict


def create_intelligent_retry_system():
    """创建智能重试系统"""

    def task_executor(state: LoopState) -> LoopState:
        """任务执行节点"""
        iteration = state.get("current_iteration", 0)
        task_name = state.get("task_name", "未知任务")

        print(f"🔄 执行第 {iteration + 1} 次迭代: {task_name}")

        # 模拟任务执行
        start_time = time.time()

        # 随机模拟成功/失败
        success_probability = 0.7 + (iteration * 0.1)  # 重试次数越多，成功率越高
        is_success = random.random() < success_probability

        execution_time = time.time() - start_time

        # 记录结果
        result = {
            "iteration": iteration + 1,
            "success": is_success,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "error_message": None if is_success else f"模拟错误_{iteration + 1}"
        }

        results = state.get("results", [])
        results.append(result)
        state["results"] = results

        # 更新计数器
        state["current_iteration"] = iteration + 1

        if not is_success:
            error_count = state.get("error_count", 0)
            state["error_count"] = error_count + 1

        # 更新性能指标
        metrics = state.get("performance_metrics", {})
        total_executions = len(results)
        successful_executions = sum(1 for r in results if r["success"])

        metrics.update({
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": sum(r["execution_time"] for r in results) / total_executions,
            "last_execution_time": execution_time
        })

        state["performance_metrics"] = metrics
        state["success_rate"] = metrics["success_rate"]

        return state

    def intelligent_continue_checker(state: LoopState) -> str:
        """智能继续检查器"""
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        results = state.get("results", [])
        error_count = state.get("error_count", 0)

        # 检查基本条件
        if current_iteration >= max_iterations:
            print(f"⏹️  达到最大迭代次数 {max_iterations}")
            return "max_iterations"

        # 检查最近的结果
        if results and results[-1]["success"]:
            print("✅ 任务执行成功")
            return "success"

        # 检查连续失败次数
        consecutive_failures = 0
        for result in reversed(results):
            if not result["success"]:
                consecutive_failures += 1
            else:
                break

        if consecutive_failures >= 3:
            print("❌ 连续失败次数过多，停止重试")
            return "too_many_failures"

        # 检查总体成功率
        if len(results) >= 3:  # 至少执行3次后才检查成功率
            success_rate = state.get("success_rate", 0.0)
            if success_rate < 0.2:  # 成功率低于20%
                print(f"📉 成功率过低 ({success_rate:.1%})，停止重试")
                return "low_success_rate"

        print("🔄 继续重试")
        return "continue"

    def delay_calculator(state: LoopState) -> LoopState:
        """延迟计算器（指数退避）"""
        current_iteration = state.get("current_iteration", 0)

        # 指数退避：1s, 2s, 4s, 8s...
        delay = min(2 ** (current_iteration - 1), 30)  # 最大延迟30秒

        print(f"⏰ 等待 {delay} 秒后重试...")
        time.sleep(delay)  # 在实际应用中，这里应该使用异步等待

        return state

    def success_handler(state: LoopState) -> LoopState:
        """成功处理器"""
        task_name = state.get("task_name", "任务")
        current_iteration = state.get("current_iteration", 0)
        performance_metrics = state.get("performance_metrics", {})

        print(f"🎉 {task_name} 执行成功！")
        print(f"📊 总共尝试了 {current_iteration} 次")
        print(f"📈 成功率: {performance_metrics.get('success_rate', 0):.1%}")

        state["should_continue"] = False
        return state

    def failure_handler(state: LoopState) -> LoopState:
        """失败处理器"""
        task_name = state.get("task_name", "任务")
        current_iteration = state.get("current_iteration", 0)
        error_count = state.get("error_count", 0)

        print(f"💥 {task_name} 最终执行失败")
        print(f"📊 总共尝试了 {current_iteration} 次，失败 {error_count} 次")

        state["should_continue"] = False
        return state

    # 构建智能重试图
    graph = StateGraph(LoopState)

    # 添加节点
    graph.add_node("task_executor", task_executor)
    graph.add_node("delay_calculator", delay_calculator)
    graph.add_node("success_handler", success_handler)
    graph.add_node("failure_handler", failure_handler)

    # 设置入口点
    graph.set_entry_point("task_executor")

    # 添加条件边
    graph.add_conditional_edges(
        "task_executor",
        intelligent_continue_checker,
        {
            "continue": "delay_calculator",
            "success": "success_handler",
            "max_iterations": "failure_handler",
            "too_many_failures": "failure_handler",
            "low_success_rate": "failure_handler"
        }
    )

    # 延迟后回到执行
    graph.add_edge("delay_calculator", "task_executor")

    # 处理器结束
    graph.add_edge("success_handler", END)
    graph.add_edge("failure_handler", END)

    return graph.compile()


def create_data_processing_pipeline():
    """创建数据处理管道循环"""

    class PipelineState(TypedDict):
        raw_data: List[dict]
        processed_data: List[dict]
        quality_score: float
        processing_round: int
        max_rounds: int
        quality_threshold: float

    def data_validator(state: PipelineState) -> PipelineState:
        """数据验证器"""
        raw_data = state.get("raw_data", [])
        processing_round = state.get("processing_round", 0)

        print(f"🔍 第 {processing_round + 1} 轮: 验证 {len(raw_data)} 条数据")

        # 模拟数据验证
        valid_data = []
        for item in raw_data:
            # 简单验证逻辑
            if len(str(item)) > 5 and "error" not in str(item).lower():
                valid_data.append(item)

        validation_rate = len(valid_data) / len(raw_data) if raw_data else 0
        print(f"   ✅ 数据验证率: {validation_rate:.1%}")

        state["raw_data"] = valid_data
        return state

    def data_processor(state: PipelineState) -> PipelineState:
        """数据处理器"""
        raw_data = state.get("raw_data", [])

        print(f"⚙️  处理 {len(raw_data)} 条数据")

        # 模拟数据处理
        processed_data = []
        for item in raw_data:
            processed_item = {
                "original": item,
                "processed": f"processed_{item}",
                "timestamp": time.time(),
                "processing_round": state.get("processing_round", 0) + 1
            }
            processed_data.append(processed_item)

        state["processed_data"] = processed_data
        return state

    def quality_assessor(state: PipelineState) -> PipelineState:
        """质量评估器"""
        processed_data = state.get("processed_data", [])
        processing_round = state.get("processing_round", 0)

        # 模拟质量评估
        quality_scores = []
        for item in processed_data:
            # 随机质量分数，随着处理轮次增加而提高
            base_score = random.uniform(0.6, 0.9)
            round_bonus = processing_round * 0.05
            item_score = min(base_score + round_bonus, 1.0)
            quality_scores.append(item_score)

        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        print(f"📊 数据质量评分: {overall_quality:.2f}")

        state["quality_score"] = overall_quality
        state["processing_round"] = processing_round + 1

        return state

    def quality_checker(state: PipelineState) -> str:
        """质量检查器"""
        quality_score = state.get("quality_score", 0.0)
        quality_threshold = state.get("quality_threshold", 0.85)
        processing_round = state.get("processing_round", 0)
        max_rounds = state.get("max_rounds", 3)

        if quality_score >= quality_threshold:
            print(f"🎯 质量达标 ({quality_score:.2f} >= {quality_threshold})")
            return "quality_met"
        elif processing_round >= max_rounds:
            print(f"⏰ 达到最大处理轮次 ({processing_round}/{max_rounds})")
            return "max_rounds"
        else:
            print(f"🔄 质量未达标，继续处理 ({quality_score:.2f} < {quality_threshold})")
            return "continue_processing"

    def pipeline_complete(state: PipelineState) -> PipelineState:
        """管道完成处理器"""
        processed_data = state.get("processed_data", [])
        quality_score = state.get("quality_score", 0.0)
        processing_round = state.get("processing_round", 0)

        print(f"✅ 数据处理管道完成！")
        print(f"📦 处理了 {len(processed_data)} 条数据")
        print(f"📊 最终质量分数: {quality_score:.2f}")
        print(f"🔄 总共处理轮次: {processing_round}")

        return state

    # 构建数据处理管道
    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("validator", data_validator)
    graph.add_node("processor", data_processor)
    graph.add_node("assessor", quality_assessor)
    graph.add_node("complete", pipeline_complete)

    # 设置入口点
    graph.set_entry_point("validator")

    # 线性处理流程
    graph.add_edge("validator", "processor")
    graph.add_edge("processor", "assessor")

    # 质量检查的条件边
    graph.add_conditional_edges(
        "assessor",
        quality_checker,
        {
            "continue_processing": "validator",  # 回到验证器形成循环
            "quality_met": "complete",
            "max_rounds": "complete"
        }
    )

    graph.add_edge("complete", END)

    return graph.compile()


def demo_loop_patterns():
    """演示循环模式"""
    print("🚀 循环模式演示开始")
    print("=" * 60)

    # 演示1: 智能重试系统
    print("\n🎯 演示1: 智能重试系统")
    print("-" * 40)

    retry_app = create_intelligent_retry_system()

    retry_state = {
        "task_name": "网络API调用",
        "current_iteration": 0,
        "max_iterations": 5,
        "results": [],
        "error_count": 0,
        "success_rate": 0.0,
        "should_continue": True,
        "performance_metrics": {}
    }

    result1 = retry_app.invoke(retry_state)

    # 演示2: 数据处理管道
    print("\n\n🎯 演示2: 数据处理管道循环")
    print("-" * 40)

    pipeline_app = create_data_processing_pipeline()

    # 模拟原始数据
    raw_data = [
        {"id": 1, "value": "data_001"},
        {"id": 2, "value": "data_002"},
        {"id": 3, "value": "error_data"},  # 这个会被过滤
        {"id": 4, "value": "data_004"},
        {"id": 5, "value": "data_005"}
    ]

    pipeline_state = {
        "raw_data": raw_data,
        "processed_data": [],
        "quality_score": 0.0,
        "processing_round": 0,
        "max_rounds": 3,
        "quality_threshold": 0.85
    }

    result2 = pipeline_app.invoke(pipeline_state)

    print("\n🎉 所有循环模式演示完成！")


if __name__ == "__main__":
    demo_loop_patterns()