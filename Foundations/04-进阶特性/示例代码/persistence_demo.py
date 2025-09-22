#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistence持久化示例代码
演示 LangGraph 中状态持久化和恢复的实现
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import time
import json
import sqlite3
import pickle
import os


class PersistentState(TypedDict):
    """持久化状态定义"""
    session_id: str
    user_id: str
    task_progress: dict
    execution_history: List[dict]
    checkpoint_metadata: dict
    recovery_info: dict


def create_fault_tolerant_system():
    """创建容错系统"""

    def checkpoint_manager(state: PersistentState) -> PersistentState:
        """检查点管理器"""
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")

        print(f"💾 检查点管理器: 会话 {session_id}, 用户 {user_id}")

        # 更新检查点元数据
        metadata = state.get("checkpoint_metadata", {})
        metadata.update({
            "last_checkpoint": time.time(),
            "checkpoint_count": metadata.get("checkpoint_count", 0) + 1,
            "node": "checkpoint_manager",
            "state_size": len(str(state))
        })

        state["checkpoint_metadata"] = metadata

        print(f"   📊 检查点 #{metadata['checkpoint_count']} 已创建")

        return state

    def task_processor(state: PersistentState) -> PersistentState:
        """任务处理器"""
        task_progress = state.get("task_progress", {})
        execution_history = state.get("execution_history", [])

        current_step = task_progress.get("current_step", 0)
        total_steps = task_progress.get("total_steps", 5)

        print(f"⚙️  处理任务步骤 {current_step + 1}/{total_steps}")

        # 模拟任务处理
        import random
        processing_time = random.uniform(0.5, 2.0)
        success = random.random() > 0.2  # 80% 成功率

        if success:
            # 成功处理
            step_result = {
                "step": current_step + 1,
                "status": "completed",
                "processing_time": processing_time,
                "timestamp": time.time(),
                "result": f"步骤{current_step + 1}完成"
            }

            task_progress.update({
                "current_step": current_step + 1,
                "last_success_step": current_step + 1,
                "last_update": time.time()
            })

            print(f"   ✅ 步骤 {current_step + 1} 完成 ({processing_time:.2f}s)")

        else:
            # 处理失败
            step_result = {
                "step": current_step + 1,
                "status": "failed",
                "processing_time": processing_time,
                "timestamp": time.time(),
                "error": f"步骤{current_step + 1}处理失败"
            }

            # 不更新进度，保持在当前步骤
            print(f"   ❌ 步骤 {current_step + 1} 失败 ({processing_time:.2f}s)")

        execution_history.append(step_result)
        state["task_progress"] = task_progress
        state["execution_history"] = execution_history

        return state

    def recovery_handler(state: PersistentState) -> PersistentState:
        """恢复处理器"""
        execution_history = state.get("execution_history", [])
        task_progress = state.get("task_progress", {})

        print("🔄 检查是否需要恢复...")

        # 检查最近的执行结果
        if execution_history:
            last_execution = execution_history[-1]

            if last_execution.get("status") == "failed":
                print("   🚨 检测到上次执行失败，启动恢复流程")

                recovery_info = {
                    "recovery_triggered": True,
                    "failed_step": last_execution.get("step"),
                    "recovery_timestamp": time.time(),
                    "recovery_strategy": "retry_from_last_checkpoint"
                }

                state["recovery_info"] = recovery_info

                # 重置到上一个成功的步骤
                last_success_step = task_progress.get("last_success_step", 0)
                task_progress["current_step"] = last_success_step

                print(f"   🔧 恢复到步骤 {last_success_step}")

        return state

    def completion_checker(state: PersistentState) -> str:
        """完成检查器"""
        task_progress = state.get("task_progress", {})
        execution_history = state.get("execution_history", [])

        current_step = task_progress.get("current_step", 0)
        total_steps = task_progress.get("total_steps", 5)

        # 检查是否有失败需要恢复
        if execution_history and execution_history[-1].get("status") == "failed":
            return "recovery"

        # 检查是否完成所有步骤
        if current_step >= total_steps:
            return "completed"

        # 继续处理
        return "continue"

    def completion_handler(state: PersistentState) -> PersistentState:
        """完成处理器"""
        task_progress = state.get("task_progress", {})
        execution_history = state.get("execution_history", [])
        checkpoint_metadata = state.get("checkpoint_metadata", {})

        print(f"🎉 所有任务步骤完成！")

        # 生成完成报告
        successful_steps = [h for h in execution_history if h.get("status") == "completed"]
        failed_steps = [h for h in execution_history if h.get("status") == "failed"]

        completion_report = {
            "total_steps": task_progress.get("total_steps", 0),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "total_checkpoints": checkpoint_metadata.get("checkpoint_count", 0),
            "completion_time": time.time()
        }

        print(f"📊 完成报告:")
        print(f"   ✅ 成功步骤: {len(successful_steps)}")
        print(f"   ❌ 失败步骤: {len(failed_steps)}")
        print(f"   💾 总检查点: {checkpoint_metadata.get('checkpoint_count', 0)}")

        state["completion_report"] = completion_report

        return state

    # 构建容错系统图
    graph = StateGraph(PersistentState)

    graph.add_node("checkpoint_manager", checkpoint_manager)
    graph.add_node("task_processor", task_processor)
    graph.add_node("recovery_handler", recovery_handler)
    graph.add_node("completion_handler", completion_handler)

    graph.set_entry_point("checkpoint_manager")

    graph.add_edge("checkpoint_manager", "task_processor")

    graph.add_conditional_edges(
        "task_processor",
        completion_checker,
        {
            "continue": "checkpoint_manager",  # 回到检查点管理器形成循环
            "recovery": "recovery_handler",
            "completed": "completion_handler"
        }
    )

    graph.add_edge("recovery_handler", "checkpoint_manager")
    graph.add_edge("completion_handler", END)

    # 配置内存检查点保存器
    memory_saver = MemorySaver()
    app = graph.compile(checkpointer=memory_saver)

    return app


def create_state_recovery_system():
    """创建状态恢复系统"""

    class RecoveryState(TypedDict):
        operation_id: str
        data_backup: dict
        operation_log: List[dict]
        recovery_points: List[dict]
        current_operation: str

    def backup_creator(state: RecoveryState) -> RecoveryState:
        """备份创建器"""
        operation_id = state.get("operation_id", "")
        data_backup = state.get("data_backup", {})

        print(f"💾 创建数据备份: {operation_id}")

        # 模拟重要数据备份
        backup_data = {
            "timestamp": time.time(),
            "operation_id": operation_id,
            "system_state": {
                "user_count": 1000,
                "data_version": "v1.2.3",
                "last_update": time.time()
            },
            "configuration": {
                "feature_flags": {"new_ui": True, "beta_features": False},
                "settings": {"theme": "dark", "language": "zh-CN"}
            }
        }

        data_backup[operation_id] = backup_data

        # 记录恢复点
        recovery_points = state.get("recovery_points", [])
        recovery_point = {
            "point_id": len(recovery_points) + 1,
            "operation_id": operation_id,
            "timestamp": time.time(),
            "description": f"操作 {operation_id} 前的备份点"
        }
        recovery_points.append(recovery_point)

        state["data_backup"] = data_backup
        state["recovery_points"] = recovery_points

        print(f"   ✅ 备份点 #{recovery_point['point_id']} 已创建")

        return state

    def operation_executor(state: RecoveryState) -> RecoveryState:
        """操作执行器"""
        operation_id = state.get("operation_id", "")
        operation_log = state.get("operation_log", [])

        print(f"⚙️  执行操作: {operation_id}")

        # 模拟操作执行
        import random
        operation_success = random.random() > 0.3  # 70% 成功率

        operation_record = {
            "operation_id": operation_id,
            "timestamp": time.time(),
            "status": "success" if operation_success else "failed",
            "details": f"操作 {operation_id} {'成功' if operation_success else '失败'}"
        }

        if not operation_success:
            operation_record["error"] = f"操作 {operation_id} 执行过程中出现错误"

        operation_log.append(operation_record)
        state["operation_log"] = operation_log
        state["current_operation"] = operation_id

        print(f"   {'✅' if operation_success else '❌'} 操作{'成功' if operation_success else '失败'}")

        return state

    def recovery_manager(state: RecoveryState) -> RecoveryState:
        """恢复管理器"""
        operation_log = state.get("operation_log", [])
        data_backup = state.get("data_backup", {})
        recovery_points = state.get("recovery_points", [])

        print("🔄 启动恢复流程...")

        # 找到最近的失败操作
        failed_operations = [op for op in operation_log if op.get("status") == "failed"]

        if failed_operations:
            latest_failure = failed_operations[-1]
            failed_operation_id = latest_failure["operation_id"]

            print(f"   🚨 检测到失败操作: {failed_operation_id}")

            # 查找对应的备份
            if failed_operation_id in data_backup:
                backup_data = data_backup[failed_operation_id]

                print(f"   💾 找到备份数据，准备恢复...")
                print(f"   📅 备份时间: {time.ctime(backup_data['timestamp'])}")

                # 执行恢复操作
                recovery_record = {
                    "recovery_id": f"recovery_{int(time.time())}",
                    "source_backup": failed_operation_id,
                    "recovery_timestamp": time.time(),
                    "status": "completed",
                    "restored_state": backup_data["system_state"]
                }

                state["last_recovery"] = recovery_record

                print(f"   ✅ 恢复完成: {recovery_record['recovery_id']}")

        return state

    def operation_status_checker(state: RecoveryState) -> str:
        """操作状态检查器"""
        operation_log = state.get("operation_log", [])

        if not operation_log:
            return "continue"

        latest_operation = operation_log[-1]
        if latest_operation.get("status") == "failed":
            return "recovery_needed"

        return "success"

    # 构建状态恢复图
    graph = StateGraph(RecoveryState)

    graph.add_node("backup_creator", backup_creator)
    graph.add_node("operation_executor", operation_executor)
    graph.add_node("recovery_manager", recovery_manager)

    graph.set_entry_point("backup_creator")

    graph.add_edge("backup_creator", "operation_executor")

    graph.add_conditional_edges(
        "operation_executor",
        operation_status_checker,
        {
            "recovery_needed": "recovery_manager",
            "success": END,
            "continue": END
        }
    )

    graph.add_edge("recovery_manager", END)

    return graph.compile()


def create_time_travel_debugger():
    """创建时间旅行调试器"""

    class TimelineState(TypedDict):
        execution_id: str
        timeline: List[dict]
        current_checkpoint: int
        debug_mode: bool
        state_snapshots: dict

    def state_recorder(state: TimelineState) -> TimelineState:
        """状态记录器"""
        execution_id = state.get("execution_id", "")
        timeline = state.get("timeline", [])
        state_snapshots = state.get("state_snapshots", {})

        checkpoint_id = len(timeline)

        print(f"📸 记录状态快照: 检查点 #{checkpoint_id}")

        # 创建状态快照
        snapshot = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "execution_id": execution_id,
            "state_data": {
                "variables": {"x": checkpoint_id * 10, "y": checkpoint_id * 20},
                "step_name": f"step_{checkpoint_id}",
                "processing_time": checkpoint_id * 0.1
            }
        }

        timeline.append({
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "description": f"执行步骤 {checkpoint_id}",
            "snapshot_ref": checkpoint_id
        })

        state_snapshots[str(checkpoint_id)] = snapshot

        state["timeline"] = timeline
        state["state_snapshots"] = state_snapshots
        state["current_checkpoint"] = checkpoint_id

        print(f"   💾 快照已保存: {snapshot['state_data']}")

        return state

    def time_travel_controller(state: TimelineState) -> TimelineState:
        """时间旅行控制器"""
        current_checkpoint = state.get("current_checkpoint", 0)
        timeline = state.get("timeline", [])
        debug_mode = state.get("debug_mode", False)

        print(f"⏰ 时间旅行控制器: 当前检查点 #{current_checkpoint}")

        if debug_mode and len(timeline) > 3:
            # 演示时间旅行：回到检查点2
            target_checkpoint = 2
            print(f"   🔄 时间旅行到检查点 #{target_checkpoint}")

            # 从快照恢复状态
            state_snapshots = state.get("state_snapshots", {})
            if str(target_checkpoint) in state_snapshots:
                target_snapshot = state_snapshots[str(target_checkpoint)]

                print(f"   📥 恢复状态: {target_snapshot['state_data']}")

                # 创建分支时间线
                branch_timeline = timeline[:target_checkpoint + 1]
                branch_timeline.append({
                    "checkpoint_id": len(branch_timeline),
                    "timestamp": time.time(),
                    "description": f"从检查点 #{target_checkpoint} 创建分支",
                    "branch_info": {"source_checkpoint": target_checkpoint}
                })

                state["timeline"] = branch_timeline
                state["current_checkpoint"] = len(branch_timeline) - 1

        return state

    def timeline_analyzer(state: TimelineState) -> TimelineState:
        """时间线分析器"""
        timeline = state.get("timeline", [])
        state_snapshots = state.get("state_snapshots", {})

        print(f"📊 分析执行时间线...")

        # 分析时间线统计
        analysis = {
            "total_checkpoints": len(timeline),
            "execution_duration": timeline[-1]["timestamp"] - timeline[0]["timestamp"] if timeline else 0,
            "branches_detected": len([entry for entry in timeline if "branch_info" in entry]),
            "analysis_timestamp": time.time()
        }

        # 性能分析
        if state_snapshots:
            processing_times = []
            for snapshot_id, snapshot in state_snapshots.items():
                processing_time = snapshot["state_data"].get("processing_time", 0)
                processing_times.append(processing_time)

            if processing_times:
                analysis["average_processing_time"] = sum(processing_times) / len(processing_times)
                analysis["max_processing_time"] = max(processing_times)

        state["timeline_analysis"] = analysis

        print(f"   📈 分析结果: {json.dumps(analysis, indent=2)}")

        return state

    def should_continue_timeline(state: TimelineState) -> str:
        """时间线继续判断"""
        current_checkpoint = state.get("current_checkpoint", 0)
        max_checkpoints = 5

        if current_checkpoint >= max_checkpoints:
            return "analyze"

        return "continue"

    # 构建时间旅行调试图
    graph = StateGraph(TimelineState)

    graph.add_node("state_recorder", state_recorder)
    graph.add_node("time_travel_controller", time_travel_controller)
    graph.add_node("timeline_analyzer", timeline_analyzer)

    graph.set_entry_point("state_recorder")

    graph.add_edge("state_recorder", "time_travel_controller")

    graph.add_conditional_edges(
        "time_travel_controller",
        should_continue_timeline,
        {
            "continue": "state_recorder",
            "analyze": "timeline_analyzer"
        }
    )

    graph.add_edge("timeline_analyzer", END)

    return graph.compile()


def demo_persistence_features():
    """演示持久化特性"""
    print("🚀 Persistence持久化演示开始")
    print("=" * 60)

    # 演示1: 容错系统
    print("\n🎯 演示1: 容错系统")
    print("-" * 40)

    fault_tolerant_app = create_fault_tolerant_system()

    config = {"configurable": {"thread_id": "fault_tolerant_session_001"}}

    initial_state = {
        "session_id": "session_001",
        "user_id": "user_123",
        "task_progress": {"current_step": 0, "total_steps": 5},
        "execution_history": [],
        "checkpoint_metadata": {},
        "recovery_info": {}
    }

    result1 = fault_tolerant_app.invoke(initial_state, config=config)

    # 演示2: 状态恢复系统
    print("\n\n🎯 演示2: 状态恢复系统")
    print("-" * 40)

    recovery_app = create_state_recovery_system()

    # 模拟多个操作
    operations = ["update_user_profile", "process_payment", "send_notification"]

    for operation in operations:
        state = {
            "operation_id": operation,
            "data_backup": {},
            "operation_log": [],
            "recovery_points": [],
            "current_operation": ""
        }

        print(f"\n🔧 执行操作: {operation}")
        result = recovery_app.invoke(state)

    # 演示3: 时间旅行调试器
    print("\n\n🎯 演示3: 时间旅行调试器")
    print("-" * 40)

    time_travel_app = create_time_travel_debugger()

    debug_state = {
        "execution_id": "debug_session_001",
        "timeline": [],
        "current_checkpoint": 0,
        "debug_mode": True,
        "state_snapshots": {}
    }

    result3 = time_travel_app.invoke(debug_state)

    print("\n🎉 Persistence持久化演示完成！")

    # 显示最终状态摘要
    print("\n📊 演示结果摘要:")
    print(f"   ✅ 容错系统: 处理了 {len(result1.get('execution_history', []))} 个步骤")

    if "timeline_analysis" in result3:
        analysis = result3["timeline_analysis"]
        print(f"   ⏰ 时间旅行: 记录了 {analysis.get('total_checkpoints', 0)} 个检查点")


if __name__ == "__main__":
    demo_persistence_features()