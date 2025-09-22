"""
监督者模式多Agent系统

这个模块实现了监督者模式的多Agent系统，其中：
- 一个中央监督者Agent负责任务分配和协调
- 多个专业化Agent执行具体任务
- 监督者负责整合结果和质量控制

适合需要严格控制和管理的场景。
"""

import asyncio
import datetime
import json
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, asdict
from enum import Enum

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# =============================================================================
# 数据模型和枚举
# =============================================================================

class TaskType(Enum):
    """任务类型"""
    RESEARCH = "research"
    CODING = "coding"
    REVIEW = "review"
    INTEGRATION = "integration"
    TESTING = "testing"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Agent类型"""
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"


@dataclass
class Task:
    """任务数据类"""
    id: str
    type: TaskType
    description: str
    requirements: List[str]
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    deadline: Optional[datetime.datetime] = None
    dependencies: List[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime.datetime = None
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


@dataclass
class Agent:
    """Agent数据类"""
    id: str
    type: AgentType
    name: str
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    current_tasks: List[str] = None
    status: Literal["idle", "busy", "offline"] = "idle"
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.current_tasks is None:
            self.current_tasks = []
        if self.performance_metrics is None:
            self.performance_metrics = {
                "success_rate": 1.0,
                "avg_completion_time": 0.0,
                "task_count": 0
            }

    def is_available(self) -> bool:
        """检查Agent是否可用"""
        return (self.status == "idle" and
                len(self.current_tasks) < self.max_concurrent_tasks)

    def can_handle_task(self, task_type: TaskType) -> bool:
        """检查Agent是否能处理指定类型的任务"""
        task_capability_map = {
            TaskType.RESEARCH: ["research", "analysis", "information_gathering"],
            TaskType.CODING: ["programming", "software_development", "implementation"],
            TaskType.REVIEW: ["code_review", "quality_assurance", "testing"],
            TaskType.INTEGRATION: ["system_integration", "coordination"],
            TaskType.TESTING: ["testing", "quality_assurance", "validation"]
        }

        required_capabilities = task_capability_map.get(task_type, [])
        return any(cap in self.capabilities for cap in required_capabilities)


class SupervisorState(TypedDict):
    """监督者系统状态"""
    # 任务管理
    tasks: List[Task]
    completed_tasks: List[Task]
    failed_tasks: List[Task]

    # Agent管理
    agents: List[Agent]
    agent_assignments: Dict[str, List[str]]  # agent_id -> task_ids

    # 当前处理状态
    current_task_id: Optional[str]
    current_step: str

    # 系统状态
    is_active: bool
    total_tasks: int
    completed_count: int
    failed_count: int

    # 结果和日志
    final_result: Optional[Dict[str, Any]]
    execution_log: List[Dict[str, Any]]
    error_log: List[str]


# =============================================================================
# Agent实现
# =============================================================================

class BaseAgentWorker:
    """Agent工作器基类"""

    def __init__(self, agent_info: Agent):
        self.agent_info = agent_info
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行任务（子类需要实现）"""
        raise NotImplementedError

    def _create_system_prompt(self, task: Task) -> str:
        """创建系统提示词"""
        return f"""你是{self.agent_info.name}，专长于{', '.join(self.agent_info.capabilities)}。

当前任务：
- 类型：{task.type.value}
- 描述：{task.description}
- 要求：{', '.join(task.requirements)}

请根据你的专业能力完成这个任务，提供详细和有用的结果。"""


class ResearcherAgent(BaseAgentWorker):
    """研究员Agent"""

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行研究任务"""
        system_prompt = self._create_system_prompt(task)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请开始执行研究任务：{task.description}")
        ]

        try:
            response = await self.llm.ainvoke(messages)

            # 模拟研究过程
            research_result = {
                "summary": response.content,
                "key_findings": [
                    "找到了相关的技术方案",
                    "分析了市场现状和趋势",
                    "识别了潜在的风险和机会"
                ],
                "recommendations": [
                    "建议采用现代化的技术栈",
                    "需要考虑用户体验设计",
                    "建议分阶段实施"
                ],
                "sources": [
                    "行业报告",
                    "技术文档",
                    "专家访谈"
                ]
            }

            return {
                "success": True,
                "result": research_result,
                "execution_time": 2.5,
                "quality_score": 0.85
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }


class CoderAgent(BaseAgentWorker):
    """编程Agent"""

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行编程任务"""
        system_prompt = self._create_system_prompt(task)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请实现以下功能：{task.description}")
        ]

        try:
            response = await self.llm.ainvoke(messages)

            # 模拟编程过程
            code_result = {
                "code": response.content,
                "files_created": [
                    "main.py",
                    "utils.py",
                    "tests/test_main.py"
                ],
                "features_implemented": task.requirements,
                "test_coverage": 85,
                "code_quality": {
                    "complexity": "medium",
                    "maintainability": "high",
                    "security": "good"
                }
            }

            return {
                "success": True,
                "result": code_result,
                "execution_time": 5.2,
                "quality_score": 0.88
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }


class ReviewerAgent(BaseAgentWorker):
    """代码审查Agent"""

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行代码审查任务"""
        system_prompt = self._create_system_prompt(task)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请审查以下代码：{task.description}")
        ]

        try:
            response = await self.llm.ainvoke(messages)

            # 模拟代码审查过程
            review_result = {
                "review_summary": response.content,
                "issues_found": [
                    {
                        "type": "warning",
                        "description": "建议添加更多的错误处理",
                        "file": "main.py",
                        "line": 25
                    },
                    {
                        "type": "suggestion",
                        "description": "可以优化算法性能",
                        "file": "utils.py",
                        "line": 15
                    }
                ],
                "strengths": [
                    "代码结构清晰",
                    "测试覆盖率良好",
                    "文档完整"
                ],
                "overall_score": 8.5,
                "approval_status": "approved_with_suggestions"
            }

            return {
                "success": True,
                "result": review_result,
                "execution_time": 3.1,
                "quality_score": 0.90
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }


# =============================================================================
# 监督者系统节点
# =============================================================================

def task_analysis_node(state: SupervisorState) -> SupervisorState:
    """任务分析节点"""
    # 分析任务依赖关系和优先级
    tasks = state["tasks"]

    # 按优先级和依赖关系排序
    sorted_tasks = sorted(tasks,
                         key=lambda t: (t.priority, len(t.dependencies)),
                         reverse=True)

    # 记录分析结果
    log_entry = {
        "step": "task_analysis",
        "timestamp": datetime.datetime.now().isoformat(),
        "message": f"分析了{len(tasks)}个任务，按优先级排序完成",
        "details": {
            "total_tasks": len(tasks),
            "high_priority": len([t for t in tasks if t.priority >= 3]),
            "has_dependencies": len([t for t in tasks if t.dependencies])
        }
    }

    return {
        **state,
        "tasks": sorted_tasks,
        "current_step": "task_analysis",
        "execution_log": state["execution_log"] + [log_entry]
    }


def agent_assignment_node(state: SupervisorState) -> SupervisorState:
    """Agent分配节点"""
    tasks = state["tasks"]
    agents = state["agents"]
    assignments = state["agent_assignments"]

    # 找到下一个可执行的任务
    next_task = None
    for task in tasks:
        if task.status == TaskStatus.PENDING:
            # 检查依赖是否完成
            deps_completed = all(
                any(ct.id == dep_id and ct.status == TaskStatus.COMPLETED
                    for ct in state["completed_tasks"])
                for dep_id in task.dependencies
            )

            if deps_completed:
                next_task = task
                break

    if not next_task:
        return {
            **state,
            "current_step": "no_available_tasks"
        }

    # 找到最适合的Agent
    suitable_agents = [
        agent for agent in agents
        if agent.is_available() and agent.can_handle_task(next_task.type)
    ]

    if not suitable_agents:
        return {
            **state,
            "current_step": "no_available_agents",
            "error_log": state["error_log"] + [f"没有可用的Agent处理任务: {next_task.id}"]
        }

    # 选择性能最好的Agent
    best_agent = max(suitable_agents,
                    key=lambda a: a.performance_metrics["success_rate"])

    # 分配任务
    next_task.assigned_agent = best_agent.id
    next_task.status = TaskStatus.ASSIGNED

    # 更新Agent状态
    best_agent.current_tasks.append(next_task.id)
    if best_agent.id not in assignments:
        assignments[best_agent.id] = []
    assignments[best_agent.id].append(next_task.id)

    log_entry = {
        "step": "agent_assignment",
        "timestamp": datetime.datetime.now().isoformat(),
        "message": f"任务 {next_task.id} 分配给 {best_agent.name}",
        "details": {
            "task_id": next_task.id,
            "task_type": next_task.type.value,
            "agent_id": best_agent.id,
            "agent_name": best_agent.name
        }
    }

    return {
        **state,
        "current_task_id": next_task.id,
        "current_step": "task_assigned",
        "execution_log": state["execution_log"] + [log_entry]
    }


async def task_execution_node(state: SupervisorState) -> SupervisorState:
    """任务执行节点"""
    current_task_id = state["current_task_id"]
    if not current_task_id:
        return state

    # 找到当前任务
    current_task = None
    for task in state["tasks"]:
        if task.id == current_task_id:
            current_task = task
            break

    if not current_task:
        return {
            **state,
            "error_log": state["error_log"] + [f"找不到任务: {current_task_id}"]
        }

    # 找到分配的Agent
    assigned_agent = None
    for agent in state["agents"]:
        if agent.id == current_task.assigned_agent:
            assigned_agent = agent
            break

    if not assigned_agent:
        return {
            **state,
            "error_log": state["error_log"] + [f"找不到分配的Agent: {current_task.assigned_agent}"]
        }

    # 创建对应的Agent工作器
    agent_workers = {
        AgentType.RESEARCHER: ResearcherAgent,
        AgentType.CODER: CoderAgent,
        AgentType.REVIEWER: ReviewerAgent
    }

    worker_class = agent_workers.get(assigned_agent.type)
    if not worker_class:
        return {
            **state,
            "error_log": state["error_log"] + [f"不支持的Agent类型: {assigned_agent.type}"]
        }

    # 执行任务
    current_task.status = TaskStatus.IN_PROGRESS
    current_task.started_at = datetime.datetime.now()

    try:
        worker = worker_class(assigned_agent)
        execution_result = await worker.execute_task(current_task)

        if execution_result["success"]:
            current_task.status = TaskStatus.COMPLETED
            current_task.completed_at = datetime.datetime.now()
            current_task.result = execution_result["result"]

            # 更新Agent性能指标
            assigned_agent.performance_metrics["task_count"] += 1
            completion_time = execution_result["execution_time"]
            current_avg = assigned_agent.performance_metrics["avg_completion_time"]
            task_count = assigned_agent.performance_metrics["task_count"]
            assigned_agent.performance_metrics["avg_completion_time"] = (
                (current_avg * (task_count - 1) + completion_time) / task_count
            )

            # 移动到完成列表
            completed_tasks = state["completed_tasks"] + [current_task]
            remaining_tasks = [t for t in state["tasks"] if t.id != current_task_id]

            log_entry = {
                "step": "task_completed",
                "timestamp": datetime.datetime.now().isoformat(),
                "message": f"任务 {current_task.id} 完成",
                "details": {
                    "task_id": current_task.id,
                    "execution_time": completion_time,
                    "quality_score": execution_result.get("quality_score", 0.0)
                }
            }

            return {
                **state,
                "tasks": remaining_tasks,
                "completed_tasks": completed_tasks,
                "completed_count": state["completed_count"] + 1,
                "current_task_id": None,
                "current_step": "task_completed",
                "execution_log": state["execution_log"] + [log_entry]
            }
        else:
            current_task.status = TaskStatus.FAILED
            current_task.error_message = execution_result["error"]

            # 移动到失败列表
            failed_tasks = state["failed_tasks"] + [current_task]
            remaining_tasks = [t for t in state["tasks"] if t.id != current_task_id]

            return {
                **state,
                "tasks": remaining_tasks,
                "failed_tasks": failed_tasks,
                "failed_count": state["failed_count"] + 1,
                "current_task_id": None,
                "current_step": "task_failed",
                "error_log": state["error_log"] + [execution_result["error"]]
            }

    except Exception as e:
        current_task.status = TaskStatus.FAILED
        current_task.error_message = str(e)

        return {
            **state,
            "current_step": "execution_error",
            "error_log": state["error_log"] + [str(e)]
        }


def result_integration_node(state: SupervisorState) -> SupervisorState:
    """结果整合节点"""
    completed_tasks = state["completed_tasks"]

    if not completed_tasks:
        return {
            **state,
            "final_result": {"message": "没有完成的任务"},
            "current_step": "integration_completed"
        }

    # 整合所有结果
    integrated_result = {
        "summary": {
            "total_tasks": state["total_tasks"],
            "completed": state["completed_count"],
            "failed": state["failed_count"],
            "success_rate": state["completed_count"] / state["total_tasks"] if state["total_tasks"] > 0 else 0
        },
        "results_by_type": {},
        "timeline": [],
        "quality_metrics": {
            "avg_quality_score": 0.0,
            "total_execution_time": 0.0
        }
    }

    # 按任务类型分组结果
    total_quality = 0.0
    total_time = 0.0

    for task in completed_tasks:
        task_type = task.type.value
        if task_type not in integrated_result["results_by_type"]:
            integrated_result["results_by_type"][task_type] = []

        integrated_result["results_by_type"][task_type].append({
            "task_id": task.id,
            "description": task.description,
            "result": task.result,
            "completion_time": task.completed_at.isoformat() if task.completed_at else None
        })

        # 时间线
        integrated_result["timeline"].append({
            "task_id": task.id,
            "type": task_type,
            "started": task.started_at.isoformat() if task.started_at else None,
            "completed": task.completed_at.isoformat() if task.completed_at else None
        })

    # 计算平均质量分数
    if completed_tasks:
        integrated_result["quality_metrics"]["avg_quality_score"] = total_quality / len(completed_tasks)
        integrated_result["quality_metrics"]["total_execution_time"] = total_time

    log_entry = {
        "step": "result_integration",
        "timestamp": datetime.datetime.now().isoformat(),
        "message": "所有任务结果已整合完成",
        "details": integrated_result["summary"]
    }

    return {
        **state,
        "final_result": integrated_result,
        "current_step": "integration_completed",
        "is_active": False,
        "execution_log": state["execution_log"] + [log_entry]
    }


# =============================================================================
# 路由函数
# =============================================================================

def supervisor_router(state: SupervisorState) -> str:
    """监督者路由函数"""
    current_step = state["current_step"]

    # 如果还有待处理任务，继续分配
    if state["tasks"] and current_step in ["task_analysis", "task_completed"]:
        return "assign_agent"

    # 如果有分配的任务，执行任务
    if current_step == "task_assigned":
        return "execute_task"

    # 如果任务失败但还有其他任务，继续处理
    if current_step == "task_failed" and state["tasks"]:
        return "assign_agent"

    # 没有更多任务或所有Agent都不可用，整合结果
    if current_step in ["no_available_tasks", "no_available_agents"] or not state["tasks"]:
        return "integrate_results"

    # 默认结束
    return "end"


# =============================================================================
# 监督者系统
# =============================================================================

def create_supervisor_graph() -> StateGraph:
    """创建监督者系统图"""
    graph = StateGraph(SupervisorState)

    # 添加节点
    graph.add_node("analyze_tasks", task_analysis_node)
    graph.add_node("assign_agent", agent_assignment_node)
    graph.add_node("execute_task", task_execution_node)
    graph.add_node("integrate_results", result_integration_node)

    # 设置入口点
    graph.set_entry_point("analyze_tasks")

    # 添加条件路由
    graph.add_conditional_edges(
        "analyze_tasks",
        supervisor_router,
        {
            "assign_agent": "assign_agent",
            "integrate_results": "integrate_results",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "assign_agent",
        supervisor_router,
        {
            "execute_task": "execute_task",
            "integrate_results": "integrate_results",
            "end": END
        }
    )

    graph.add_conditional_edges(
        "execute_task",
        supervisor_router,
        {
            "assign_agent": "assign_agent",
            "integrate_results": "integrate_results",
            "end": END
        }
    )

    graph.add_edge("integrate_results", END)

    return graph


class SupervisorSystem:
    """监督者系统主类"""

    def __init__(self):
        """初始化监督者系统"""
        self.graph = create_supervisor_graph()
        self.memory_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory_saver)

        # 初始化Agent
        self.agents = self._create_default_agents()

        print("🤖 监督者系统已启动")
        print(f"📊 已注册 {len(self.agents)} 个Agent")

    def _create_default_agents(self) -> List[Agent]:
        """创建默认的Agent"""
        return [
            Agent(
                id="researcher_001",
                type=AgentType.RESEARCHER,
                name="高级研究员",
                capabilities=["research", "analysis", "information_gathering", "market_research"],
                max_concurrent_tasks=2
            ),
            Agent(
                id="coder_001",
                type=AgentType.CODER,
                name="全栈开发工程师",
                capabilities=["programming", "software_development", "implementation", "architecture"],
                max_concurrent_tasks=1
            ),
            Agent(
                id="reviewer_001",
                type=AgentType.REVIEWER,
                name="高级代码审查员",
                capabilities=["code_review", "quality_assurance", "testing", "security_audit"],
                max_concurrent_tasks=3
            )
        ]

    def create_initial_state(self, tasks: List[Dict[str, Any]]) -> SupervisorState:
        """创建初始状态"""
        task_objects = []
        for i, task_data in enumerate(tasks):
            task = Task(
                id=f"task_{i+1:03d}",
                type=TaskType(task_data["type"]),
                description=task_data["description"],
                requirements=task_data.get("requirements", []),
                priority=task_data.get("priority", 1),
                dependencies=task_data.get("dependencies", [])
            )
            task_objects.append(task)

        return {
            "tasks": task_objects,
            "completed_tasks": [],
            "failed_tasks": [],
            "agents": self.agents,
            "agent_assignments": {},
            "current_task_id": None,
            "current_step": "initialized",
            "is_active": True,
            "total_tasks": len(task_objects),
            "completed_count": 0,
            "failed_count": 0,
            "final_result": None,
            "execution_log": [],
            "error_log": []
        }

    async def execute_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行任务列表"""
        # 创建初始状态
        initial_state = self.create_initial_state(tasks)

        # 配置
        config = {"configurable": {"thread_id": f"supervisor_{uuid.uuid4()}"}}

        try:
            print(f"🚀 开始执行 {len(tasks)} 个任务...")

            # 执行图
            result = await self.app.ainvoke(initial_state, config)

            print("✅ 任务执行完成")

            return {
                "success": True,
                "result": result["final_result"],
                "execution_log": result["execution_log"],
                "error_log": result["error_log"],
                "summary": {
                    "total_tasks": result["total_tasks"],
                    "completed": result["completed_count"],
                    "failed": result["failed_count"],
                    "success_rate": result["completed_count"] / result["total_tasks"] if result["total_tasks"] > 0 else 0
                }
            }

        except Exception as e:
            print(f"❌ 任务执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """获取所有Agent的状态"""
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "status": agent.status,
                "current_tasks": len(agent.current_tasks),
                "max_tasks": agent.max_concurrent_tasks,
                "capabilities": agent.capabilities,
                "performance": agent.performance_metrics
            }
            for agent in self.agents
        ]


# =============================================================================
# 演示和测试
# =============================================================================

async def demo_simple_workflow():
    """演示简单工作流"""
    print("=== 简单工作流演示 ===\n")

    system = SupervisorSystem()

    # 定义任务
    tasks = [
        {
            "type": "research",
            "description": "研究待办事项应用的市场需求和技术方案",
            "requirements": [
                "分析用户需求",
                "调研竞品功能",
                "推荐技术栈"
            ],
            "priority": 3
        },
        {
            "type": "coding",
            "description": "实现待办事项应用的核心功能",
            "requirements": [
                "用户注册登录",
                "任务增删改查",
                "任务分类和标签"
            ],
            "priority": 2,
            "dependencies": ["task_001"]
        },
        {
            "type": "review",
            "description": "审查待办事项应用的代码质量",
            "requirements": [
                "代码规范检查",
                "安全性审查",
                "性能优化建议"
            ],
            "priority": 1,
            "dependencies": ["task_002"]
        }
    ]

    # 执行任务
    result = await system.execute_tasks(tasks)

    # 显示结果
    if result["success"]:
        print("📊 执行结果:")
        print(f"- 总任务数: {result['summary']['total_tasks']}")
        print(f"- 完成任务: {result['summary']['completed']}")
        print(f"- 失败任务: {result['summary']['failed']}")
        print(f"- 成功率: {result['summary']['success_rate']:.2%}")

        print("\n📋 详细结果:")
        for task_type, results in result["result"]["results_by_type"].items():
            print(f"\n{task_type.upper()} 任务:")
            for task_result in results:
                print(f"  - {task_result['description']}")
    else:
        print(f"❌ 执行失败: {result['error']}")


async def demo_complex_workflow():
    """演示复杂工作流"""
    print("=== 复杂工作流演示 ===\n")

    system = SupervisorSystem()

    # 复杂的产品开发任务
    tasks = [
        {
            "type": "research",
            "description": "电商平台市场调研和竞品分析",
            "requirements": ["用户行为分析", "技术趋势研究", "商业模式分析"],
            "priority": 3
        },
        {
            "type": "research",
            "description": "技术架构调研和选型",
            "requirements": ["微服务架构", "数据库选型", "缓存策略"],
            "priority": 3
        },
        {
            "type": "coding",
            "description": "用户管理模块开发",
            "requirements": ["注册登录", "权限管理", "个人信息"],
            "priority": 2,
            "dependencies": ["task_001", "task_002"]
        },
        {
            "type": "coding",
            "description": "商品管理模块开发",
            "requirements": ["商品展示", "分类管理", "搜索功能"],
            "priority": 2,
            "dependencies": ["task_002"]
        },
        {
            "type": "review",
            "description": "用户管理模块代码审查",
            "requirements": ["安全性检查", "性能测试", "代码规范"],
            "priority": 1,
            "dependencies": ["task_003"]
        },
        {
            "type": "review",
            "description": "商品管理模块代码审查",
            "requirements": ["功能测试", "集成测试", "用户体验"],
            "priority": 1,
            "dependencies": ["task_004"]
        }
    ]

    # 执行任务
    result = await system.execute_tasks(tasks)

    # 显示结果
    if result["success"]:
        print("📊 执行摘要:")
        summary = result["summary"]
        print(f"  总任务: {summary['total_tasks']}")
        print(f"  已完成: {summary['completed']}")
        print(f"  失败: {summary['failed']}")
        print(f"  成功率: {summary['success_rate']:.1%}")

        print("\n🔍 Agent状态:")
        for agent_status in system.get_agent_status():
            print(f"  {agent_status['name']}: {agent_status['performance']['task_count']} 个任务")


# =============================================================================
# 主程序
# =============================================================================

async def main():
    """主程序"""
    print("🚀 监督者模式多Agent系统")
    print("=" * 50)

    while True:
        print("\n选择演示模式:")
        print("1. 简单工作流演示")
        print("2. 复杂工作流演示")
        print("3. 查看Agent状态")
        print("4. 自定义任务")
        print("5. 退出")

        choice = input("\n请选择 (1-5): ").strip()

        if choice == "1":
            await demo_simple_workflow()

        elif choice == "2":
            await demo_complex_workflow()

        elif choice == "3":
            system = SupervisorSystem()
            print("\n📊 Agent状态:")
            for agent in system.get_agent_status():
                print(f"  {agent['name']} ({agent['type']}): {agent['status']}")
                print(f"    能力: {', '.join(agent['capabilities'])}")
                print(f"    任务容量: {agent['current_tasks']}/{agent['max_tasks']}")

        elif choice == "4":
            print("\n自定义任务功能开发中...")

        elif choice == "5":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选择，请重试。")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# 学习指南和扩展建议
# =============================================================================

"""
🎯 监督者模式学习要点:

1. **中央控制**:
   - 监督者负责全局协调和决策
   - 清晰的任务分配和执行流程
   - 统一的状态管理和监控

2. **专业化Agent**:
   - 每个Agent专注于特定领域
   - 基于能力的任务分配
   - 性能指标跟踪和优化

3. **依赖管理**:
   - 任务依赖关系的解析
   - 按依赖顺序执行任务
   - 并行任务的识别和处理

4. **错误处理**:
   - 任务失败的检测和处理
   - Agent异常的恢复机制
   - 系统容错和降级策略

5. **结果整合**:
   - 多Agent结果的统一格式
   - 质量评估和报告生成
   - 性能指标的计算和分析

🔧 扩展建议:

1. **动态Agent管理**:
   - 运行时添加/移除Agent
   - Agent负载均衡
   - 智能任务分配算法

2. **高级协作模式**:
   - Agent间直接通信
   - 知识共享机制
   - 协作学习功能

3. **监控和可视化**:
   - 实时执行状态监控
   - 性能指标仪表板
   - 任务执行时间线

4. **持久化和恢复**:
   - 任务状态持久化
   - 系统故障恢复
   - 长时间运行的任务支持

这个监督者模式为构建可控、可监控的多Agent系统提供了坚实基础！
"""