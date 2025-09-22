# 05-Human-in-the-Loop

## 🎯 学习目标

通过本章学习，你将掌握：
- Human-in-the-Loop (HIL) 的核心概念和应用场景
- `interrupt()` 函数的使用方法
- 设计人工审批和干预流程
- 实现断点调试和动态控制机制

## 📚 Human-in-the-Loop 基础概念

### 什么是 Human-in-the-Loop？

Human-in-the-Loop (HIL) 是一种设计模式，允许在自动化流程中的关键点暂停执行，等待人工输入、审查或决策。这种模式结合了 AI 的效率和人类的判断力。

### HIL 的核心价值

1. **质量保证**：在关键决策点引入人工审查
2. **风险控制**：防止AI做出可能有害的决策
3. **学习反馈**：通过人工干预改进AI性能
4. **合规要求**：满足某些行业的人工监督要求

### 应用场景

```
┌─────────────────────────────────────┐
│            HIL 应用场景             │
├─────────────────────────────────────┤
│ • 内容审核（敏感内容检测）          │
│ • 金融决策（贷款审批）              │
│ • 医疗诊断（治疗方案确认）          │
│ • 法律文档（合同审查）              │
│ • 客服升级（复杂问题转人工）        │
│ • 代码审查（安全检查）              │
│ • 创意输出（设计方案确认）          │
└─────────────────────────────────────┘
```

## 🛑 interrupt() 函数详解

### 基础用法

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import human_in_the_loop
from typing import TypedDict

class HILState(TypedDict):
    """Human-in-the-Loop 状态"""
    user_request: str
    ai_analysis: str
    human_approval: bool
    final_action: str
    approval_required: bool

def create_basic_hil_graph():
    """创建基础HIL图"""

    def analyze_request(state: HILState) -> HILState:
        """分析用户请求"""
        request = state.get("user_request", "")

        # AI分析逻辑
        ai_analysis = f"AI分析结果：对于请求'{request}'，建议采取相应行动。"

        # 判断是否需要人工审批
        sensitive_keywords = ["删除", "转账", "重要", "机密"]
        needs_approval = any(keyword in request for keyword in sensitive_keywords)

        state["ai_analysis"] = ai_analysis
        state["approval_required"] = needs_approval

        return state

    def human_approval_node(state: HILState) -> HILState:
        """人工审批节点"""
        # 使用 interrupt 暂停执行，等待人工输入
        from langgraph.prebuilt import interrupt

        # 准备审批信息
        approval_info = {
            "request": state.get("user_request"),
            "ai_analysis": state.get("ai_analysis"),
            "action": "请审批是否执行此操作"
        }

        # 中断执行，等待人工决策
        human_decision = interrupt(approval_info)

        # 处理人工决策
        if human_decision is not None:
            state["human_approval"] = human_decision.get("approved", False)
            state["approval_reason"] = human_decision.get("reason", "")

        return state

    def execute_action(state: HILState) -> HILState:
        """执行最终行动"""
        if state.get("human_approval", False) or not state.get("approval_required", False):
            state["final_action"] = "操作已执行"
        else:
            state["final_action"] = "操作被拒绝或未获得批准"

        return state

    def approval_router(state: HILState) -> str:
        """审批路由器"""
        if state.get("approval_required", False):
            return "human_approval"
        else:
            return "execute"

    # 构建图
    graph = StateGraph(HILState)
    graph.add_node("analyze", analyze_request)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("execute", execute_action)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze",
        approval_router,
        {
            "human_approval": "human_approval",
            "execute": "execute"
        }
    )

    graph.add_edge("human_approval", "execute")
    graph.add_edge("execute", END)

    return graph.compile()

# 使用示例
def use_hil_graph():
    """使用HIL图的示例"""
    app = create_basic_hil_graph()

    # 第一次调用 - 需要审批的请求
    initial_state = {
        "user_request": "删除所有用户数据",
        "ai_analysis": "",
        "human_approval": False,
        "final_action": "",
        "approval_required": False
    }

    # 执行到中断点
    result = app.invoke(initial_state)
    print("执行到中断点：", result)

    # 模拟人工决策
    human_input = {"approved": False, "reason": "风险太高，拒绝执行"}

    # 从中断点继续执行
    final_result = app.invoke(human_input)
    print("最终结果：", final_result)
```

### 高级 interrupt 模式

```python
def advanced_interrupt_patterns():
    """高级中断模式"""

    class AdvancedHILState(TypedDict):
        task_queue: list
        current_task: dict
        human_feedback: dict
        retry_count: int
        quality_score: float

    def quality_check_node(state: AdvancedHILState) -> AdvancedHILState:
        """质量检查节点"""
        current_task = state.get("current_task", {})
        quality_score = state.get("quality_score", 0.0)

        if quality_score < 0.7:  # 质量分数低于阈值
            # 准备人工审查信息
            review_info = {
                "task": current_task,
                "quality_score": quality_score,
                "issues": ["质量不达标", "需要人工审查"],
                "options": ["批准", "修改", "拒绝"],
                "default_action": "修改"
            }

            # 中断并等待人工审查
            human_feedback = interrupt(review_info)

            if human_feedback:
                state["human_feedback"] = human_feedback
                action = human_feedback.get("action", "modify")

                if action == "approve":
                    state["quality_score"] = 1.0  # 人工批准
                elif action == "reject":
                    state["quality_score"] = 0.0  # 人工拒绝
                else:  # modify
                    # 根据人工反馈修改任务
                    modifications = human_feedback.get("modifications", {})
                    current_task.update(modifications)
                    state["current_task"] = current_task

        return state

    def conditional_interrupt_node(state: AdvancedHILState) -> AdvancedHILState:
        """条件性中断节点"""
        retry_count = state.get("retry_count", 0)
        current_task = state.get("current_task", {})

        # 只在特定条件下中断
        should_interrupt = (
            retry_count >= 3 or  # 重试次数过多
            current_task.get("priority") == "high" or  # 高优先级任务
            current_task.get("risk_level") == "critical"  # 关键风险任务
        )

        if should_interrupt:
            interrupt_info = {
                "reason": "条件触发人工干预",
                "retry_count": retry_count,
                "task_priority": current_task.get("priority"),
                "risk_level": current_task.get("risk_level"),
                "required_action": "review_and_decide"
            }

            human_decision = interrupt(interrupt_info)

            if human_decision:
                action = human_decision.get("action")
                if action == "continue":
                    state["retry_count"] = 0  # 重置重试计数
                elif action == "abort":
                    state["current_task"]["status"] = "aborted"
                elif action == "escalate":
                    state["current_task"]["escalated"] = True

        return state

    return graph
```

## 🔧 实际应用模式

### 1. 内容审核系统

```python
def create_content_moderation_system():
    """创建内容审核系统"""

    class ModerationState(TypedDict):
        content: str
        ai_analysis: dict
        human_review: dict
        final_decision: str
        confidence_score: float

    def ai_content_analysis(state: ModerationState) -> ModerationState:
        """AI内容分析"""
        content = state.get("content", "")

        # 模拟AI内容分析
        analysis = {
            "toxicity_score": 0.3,
            "spam_probability": 0.1,
            "inappropriate_content": False,
            "detected_issues": [],
            "confidence": 0.85
        }

        # 检测潜在问题
        if "垃圾" in content:
            analysis["spam_probability"] = 0.8
            analysis["detected_issues"].append("可能的垃圾内容")

        if any(word in content for word in ["暴力", "仇恨"]):
            analysis["toxicity_score"] = 0.9
            analysis["inappropriate_content"] = True
            analysis["detected_issues"].append("可能包含不当内容")

        state["ai_analysis"] = analysis
        state["confidence_score"] = analysis["confidence"]

        return state

    def human_moderation_node(state: ModerationState) -> ModerationState:
        """人工审核节点"""
        ai_analysis = state.get("ai_analysis", {})

        # 准备审核信息
        moderation_request = {
            "content": state.get("content"),
            "ai_analysis": ai_analysis,
            "review_type": "content_moderation",
            "urgency": "high" if ai_analysis.get("toxicity_score", 0) > 0.8 else "normal",
            "instructions": "请审查此内容是否违反社区准则"
        }

        # 中断等待人工审核
        human_review = interrupt(moderation_request)

        if human_review:
            state["human_review"] = human_review

            # 处理人工审核结果
            if human_review.get("action") == "approve":
                state["final_decision"] = "approved"
            elif human_review.get("action") == "reject":
                state["final_decision"] = "rejected"
                state["rejection_reason"] = human_review.get("reason", "")
            elif human_review.get("action") == "flag":
                state["final_decision"] = "flagged_for_review"

        return state

    def moderation_router(state: ModerationState) -> str:
        """审核路由器"""
        ai_analysis = state.get("ai_analysis", {})
        confidence = state.get("confidence_score", 0.0)

        # 高置信度且低风险 - 自动通过
        if (confidence > 0.9 and
            ai_analysis.get("toxicity_score", 0) < 0.3 and
            not ai_analysis.get("inappropriate_content", False)):
            return "auto_approve"

        # 高风险或低置信度 - 人工审核
        if (ai_analysis.get("toxicity_score", 0) > 0.7 or
            ai_analysis.get("inappropriate_content", False) or
            confidence < 0.6):
            return "human_review"

        # 中等风险 - 可配置策略
        return "human_review"  # 保守策略

    def auto_approve_node(state: ModerationState) -> ModerationState:
        """自动批准节点"""
        state["final_decision"] = "auto_approved"
        return state

    # 构建审核图
    graph = StateGraph(ModerationState)
    graph.add_node("ai_analysis", ai_content_analysis)
    graph.add_node("human_review", human_moderation_node)
    graph.add_node("auto_approve", auto_approve_node)

    graph.set_entry_point("ai_analysis")

    graph.add_conditional_edges(
        "ai_analysis",
        moderation_router,
        {
            "human_review": "human_review",
            "auto_approve": "auto_approve"
        }
    )

    graph.add_edge("human_review", END)
    graph.add_edge("auto_approve", END)

    return graph.compile()
```

### 2. 金融审批系统

```python
def create_financial_approval_system():
    """创建金融审批系统"""

    class FinancialState(TypedDict):
        application: dict
        risk_assessment: dict
        approval_history: list
        human_decision: dict
        final_result: str

    def risk_assessment_node(state: FinancialState) -> FinancialState:
        """风险评估节点"""
        application = state.get("application", {})

        # 模拟风险评估
        risk_score = 0.0
        risk_factors = []

        amount = application.get("amount", 0)
        credit_score = application.get("credit_score", 750)
        income = application.get("income", 0)

        # 计算风险分数
        if amount > 100000:
            risk_score += 0.3
            risk_factors.append("高金额申请")

        if credit_score < 650:
            risk_score += 0.4
            risk_factors.append("信用评分较低")

        if income < amount * 0.3:  # 收入低于贷款金额30%
            risk_score += 0.5
            risk_factors.append("收入债务比例不理想")

        risk_assessment = {
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "risk_factors": risk_factors,
            "recommendation": "approve" if risk_score < 0.4 else "review" if risk_score < 0.8 else "reject"
        }

        state["risk_assessment"] = risk_assessment
        return state

    def senior_approval_node(state: FinancialState) -> FinancialState:
        """高级审批节点"""
        application = state.get("application", {})
        risk_assessment = state.get("risk_assessment", {})

        # 准备审批包
        approval_package = {
            "applicant": {
                "name": application.get("name"),
                "income": application.get("income"),
                "credit_score": application.get("credit_score"),
                "employment_history": application.get("employment_history")
            },
            "request": {
                "amount": application.get("amount"),
                "purpose": application.get("purpose"),
                "term": application.get("term")
            },
            "risk_analysis": risk_assessment,
            "required_decision": ["approve", "reject", "request_more_info"],
            "approval_authority": "senior_underwriter",
            "deadline": "24_hours"
        }

        # 中断等待高级审批
        human_decision = interrupt(approval_package)

        if human_decision:
            state["human_decision"] = human_decision

            decision = human_decision.get("decision")
            reason = human_decision.get("reason", "")

            if decision == "approve":
                state["final_result"] = f"贷款批准，原因：{reason}"
            elif decision == "reject":
                state["final_result"] = f"贷款拒绝，原因：{reason}"
            else:  # request_more_info
                state["final_result"] = f"需要更多信息：{reason}"
                state["additional_info_required"] = human_decision.get("info_needed", [])

        return state

    def approval_router(state: FinancialState) -> str:
        """审批路由器"""
        risk_assessment = state.get("risk_assessment", {})
        risk_level = risk_assessment.get("risk_level", "medium")
        amount = state.get("application", {}).get("amount", 0)

        # 自动批准条件：低风险且金额较小
        if risk_level == "low" and amount < 50000:
            return "auto_approve"

        # 需要人工审批的条件
        if (risk_level in ["medium", "high"] or
            amount > 50000):
            return "senior_approval"

        return "auto_approve"

    def auto_approve_node(state: FinancialState) -> FinancialState:
        """自动批准节点"""
        state["final_result"] = "贷款自动批准（满足自动批准条件）"
        return state

    # 构建金融审批图
    graph = StateGraph(FinancialState)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("senior_approval", senior_approval_node)
    graph.add_node("auto_approve", auto_approve_node)

    graph.set_entry_point("risk_assessment")

    graph.add_conditional_edges(
        "risk_assessment",
        approval_router,
        {
            "senior_approval": "senior_approval",
            "auto_approve": "auto_approve"
        }
    )

    graph.add_edge("senior_approval", END)
    graph.add_edge("auto_approve", END)

    return graph.compile()
```

## 🎛️ 断点调试系统

### 1. 动态断点

```python
class DebugBreakpointSystem:
    """调试断点系统"""

    def __init__(self):
        self.breakpoints = {}
        self.debug_mode = False

    def set_breakpoint(self, node_name: str, condition: callable = None):
        """设置断点"""
        self.breakpoints[node_name] = {
            "enabled": True,
            "condition": condition,
            "hit_count": 0
        }

    def clear_breakpoint(self, node_name: str):
        """清除断点"""
        if node_name in self.breakpoints:
            del self.breakpoints[node_name]

    def should_break(self, node_name: str, state: dict) -> bool:
        """检查是否应该在此节点中断"""
        if not self.debug_mode or node_name not in self.breakpoints:
            return False

        breakpoint = self.breakpoints[node_name]
        if not breakpoint["enabled"]:
            return False

        breakpoint["hit_count"] += 1

        # 检查断点条件
        condition = breakpoint.get("condition")
        if condition and not condition(state):
            return False

        return True

    def create_debug_node(self, original_node_func, node_name: str):
        """创建带调试功能的节点"""
        def debug_wrapper(state):
            # 检查是否需要中断
            if self.should_break(node_name, state):
                debug_info = {
                    "node_name": node_name,
                    "current_state": state,
                    "breakpoint_info": self.breakpoints[node_name],
                    "debug_options": ["continue", "step", "inspect", "modify_state"],
                    "message": f"Breakpoint hit at node: {node_name}"
                }

                # 中断执行
                debug_command = interrupt(debug_info)

                if debug_command:
                    command = debug_command.get("command", "continue")

                    if command == "modify_state":
                        modifications = debug_command.get("state_modifications", {})
                        state.update(modifications)
                    elif command == "inspect":
                        # 返回当前状态供检查
                        return state

            # 执行原始节点功能
            return original_node_func(state)

        return debug_wrapper

def create_debuggable_graph():
    """创建可调试的图"""
    debug_system = DebugBreakpointSystem()

    class DebugState(TypedDict):
        data: str
        processed_data: str
        step_count: int
        debug_info: dict

    def process_step1(state: DebugState) -> DebugState:
        """处理步骤1"""
        data = state.get("data", "")
        processed = f"Step1: {data.upper()}"

        state["processed_data"] = processed
        state["step_count"] = state.get("step_count", 0) + 1

        return state

    def process_step2(state: DebugState) -> DebugState:
        """处理步骤2"""
        processed_data = state.get("processed_data", "")
        final_processed = f"Step2: {processed_data} -> FINAL"

        state["processed_data"] = final_processed
        state["step_count"] = state.get("step_count", 0) + 1

        return state

    # 设置调试断点
    debug_system.debug_mode = True
    debug_system.set_breakpoint("step1", condition=lambda state: "test" in state.get("data", ""))
    debug_system.set_breakpoint("step2")

    # 创建带调试功能的节点
    debug_step1 = debug_system.create_debug_node(process_step1, "step1")
    debug_step2 = debug_system.create_debug_node(process_step2, "step2")

    # 构建图
    graph = StateGraph(DebugState)
    graph.add_node("step1", debug_step1)
    graph.add_node("step2", debug_step2)

    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    return graph.compile(), debug_system
```

### 2. 运行时控制

```python
class RuntimeController:
    """运行时控制器"""

    def __init__(self):
        self.execution_mode = "normal"  # normal, step, pause
        self.execution_log = []

    def create_controllable_node(self, node_func, node_name: str):
        """创建可控制的节点"""
        def controlled_wrapper(state):
            # 记录执行
            self.execution_log.append({
                "node": node_name,
                "timestamp": time.time(),
                "state_snapshot": dict(state)
            })

            # 检查执行模式
            if self.execution_mode == "pause":
                control_info = {
                    "status": "paused",
                    "node": node_name,
                    "state": state,
                    "controls": ["resume", "step", "abort"],
                    "message": f"Execution paused at {node_name}"
                }

                control_command = interrupt(control_info)

                if control_command:
                    command = control_command.get("command", "resume")
                    if command == "abort":
                        raise Exception("Execution aborted by user")
                    elif command == "step":
                        self.execution_mode = "step"

            elif self.execution_mode == "step":
                step_info = {
                    "status": "stepping",
                    "node": node_name,
                    "state": state,
                    "controls": ["continue", "pause", "abort"],
                    "message": f"Step execution at {node_name}"
                }

                step_command = interrupt(step_info)

                if step_command:
                    command = step_command.get("command", "continue")
                    if command == "pause":
                        self.execution_mode = "pause"
                    elif command == "continue":
                        self.execution_mode = "normal"
                    elif command == "abort":
                        raise Exception("Execution aborted by user")

            # 执行实际节点
            return node_func(state)

        return controlled_wrapper
```

## 🎮 实践练习

### 练习 1：智能客服升级系统

创建一个智能客服系统，当遇到复杂问题时自动升级到人工客服：

```python
def intelligent_customer_service():
    """
    练习：智能客服升级系统

    要求：
    1. AI首先尝试处理用户问题
    2. 如果置信度低或问题复杂，升级到人工
    3. 人工客服可以选择接管或提供指导
    4. 记录升级原因和处理结果
    """
    # TODO: 实现你的客服升级系统
    pass
```

### 练习 2：代码审查工作流

设计一个代码审查工作流，包含自动检查和人工审查：

```python
def code_review_workflow():
    """
    练习：代码审查工作流

    要求：
    1. 自动进行代码质量检查
    2. 高风险或复杂变更需要人工审查
    3. 支持审查者添加评论和建议
    4. 根据审查结果决定是否合并
    """
    # TODO: 实现你的代码审查工作流
    pass
```

## 🚀 最佳实践

### 1. HIL 设计原则

- **明确触发条件**：清楚定义何时需要人工干预
- **提供足够上下文**：给人工审查者提供完整信息
- **设计简洁界面**：让人工决策过程尽可能简单
- **支持快速决策**：提供默认选项和快捷操作

### 2. 性能优化

- **异步处理**：避免阻塞其他任务
- **批量审查**：将相似任务组合处理
- **智能路由**：将任务分配给最合适的审查者
- **缓存决策**：对相似情况复用之前的决策

### 3. 安全考虑

- **权限验证**：确保只有授权人员可以做决策
- **审计日志**：记录所有人工干预的详细信息
- **超时处理**：设置决策超时机制
- **备用方案**：准备人工不可用时的处理策略

## 📚 推荐阅读

- [LangGraph 官方文档 - Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Interrupts 使用指南](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/)
- 相关概念：Breakpoints、Time Travel

---

**下一章预告**：学习完 Human-in-the-Loop 后，我们将深入探讨 Persistence 机制，了解如何实现状态持久化和故障恢复。