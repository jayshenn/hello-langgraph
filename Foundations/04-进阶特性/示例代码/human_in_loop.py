#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-in-the-Loop示例代码
演示 LangGraph 中人机协作的实现
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time
import json


class HILState(TypedDict):
    """Human-in-the-Loop状态定义"""
    task_description: str
    ai_analysis: dict
    human_review_required: bool
    human_feedback: dict
    final_decision: str
    approval_chain: List[dict]
    risk_level: str


def create_approval_workflow():
    """创建审批工作流"""

    def risk_assessor(state: HILState) -> HILState:
        """风险评估器"""
        task_description = state.get("task_description", "")

        print(f"🔍 AI风险评估: {task_description}")

        # 模拟AI风险评估
        risk_factors = []
        risk_score = 0.0

        # 检查高风险关键词
        high_risk_keywords = ["删除", "转账", "重要", "机密", "永久", "不可逆"]
        medium_risk_keywords = ["修改", "更新", "发送", "公开"]

        for keyword in high_risk_keywords:
            if keyword in task_description:
                risk_factors.append(f"包含高风险关键词: {keyword}")
                risk_score += 0.3

        for keyword in medium_risk_keywords:
            if keyword in task_description:
                risk_factors.append(f"包含中风险关键词: {keyword}")
                risk_score += 0.1

        # 检查任务复杂度
        if len(task_description) > 100:
            risk_factors.append("任务描述复杂")
            risk_score += 0.1

        # 确定风险等级
        if risk_score >= 0.7:
            risk_level = "高风险"
        elif risk_score >= 0.4:
            risk_level = "中风险"
        else:
            risk_level = "低风险"

        ai_analysis = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "ai_recommendation": "需要人工审批" if risk_score >= 0.4 else "可以自动执行",
            "analysis_timestamp": time.time()
        }

        state["ai_analysis"] = ai_analysis
        state["risk_level"] = risk_level
        state["human_review_required"] = risk_score >= 0.4

        print(f"   📊 风险等级: {risk_level} (评分: {risk_score:.2f})")
        print(f"   ⚠️  风险因素: {', '.join(risk_factors) if risk_factors else '无'}")

        return state

    def human_approval_node(state: HILState) -> HILState:
        """人工审批节点"""
        task_description = state.get("task_description", "")
        ai_analysis = state.get("ai_analysis", {})

        print(f"\n🙋 人工审批请求")
        print(f"📋 任务: {task_description}")
        print(f"🤖 AI建议: {ai_analysis.get('ai_recommendation', '无')}")
        print(f"⚠️  风险因素: {', '.join(ai_analysis.get('risk_factors', []))}")

        # 模拟人工审批界面
        approval_request = {
            "task": task_description,
            "ai_analysis": ai_analysis,
            "review_options": ["批准", "拒绝", "需要更多信息", "修改后批准"],
            "reviewer_guidelines": "请根据风险评估结果做出决策"
        }

        # 这里在实际应用中会使用 interrupt() 函数暂停执行
        # human_feedback = interrupt(approval_request)

        # 为演示目的，我们模拟人工决策
        human_feedback = simulate_human_decision(ai_analysis)

        state["human_feedback"] = human_feedback

        # 记录审批链
        approval_chain = state.get("approval_chain", [])
        approval_entry = {
            "reviewer": human_feedback.get("reviewer", "未知审批者"),
            "decision": human_feedback.get("decision", "未决定"),
            "reason": human_feedback.get("reason", ""),
            "timestamp": time.time()
        }
        approval_chain.append(approval_entry)
        state["approval_chain"] = approval_chain

        print(f"✅ 人工决策: {human_feedback.get('decision', '未决定')}")
        print(f"💭 理由: {human_feedback.get('reason', '无')}")

        return state

    def decision_processor(state: HILState) -> HILState:
        """决策处理器"""
        human_feedback = state.get("human_feedback", {})
        ai_analysis = state.get("ai_analysis", {})
        human_review_required = state.get("human_review_required", False)

        if human_review_required:
            # 基于人工反馈做最终决策
            decision = human_feedback.get("decision", "拒绝")
            reason = human_feedback.get("reason", "人工审批")
        else:
            # 基于AI分析自动决策
            decision = "自动批准"
            reason = "低风险任务，AI自动批准"

        final_decision = f"{decision}: {reason}"
        state["final_decision"] = final_decision

        print(f"\n🎯 最终决策: {final_decision}")

        return state

    def approval_router(state: HILState) -> str:
        """审批路由器"""
        human_review_required = state.get("human_review_required", False)

        if human_review_required:
            return "human_approval"
        else:
            return "auto_decision"

    # 构建审批工作流图
    graph = StateGraph(HILState)

    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("auto_decision", decision_processor)
    graph.add_node("final_processor", decision_processor)

    graph.set_entry_point("risk_assessor")

    # 风险评估后的路由
    graph.add_conditional_edges(
        "risk_assessor",
        approval_router,
        {
            "human_approval": "human_approval",
            "auto_decision": "auto_decision"
        }
    )

    # 人工审批后的处理
    graph.add_edge("human_approval", "final_processor")
    graph.add_edge("auto_decision", END)
    graph.add_edge("final_processor", END)

    return graph.compile()


def simulate_human_decision(ai_analysis: dict) -> dict:
    """模拟人工决策（实际应用中由真人决策）"""
    risk_score = ai_analysis.get("risk_score", 0.0)
    risk_factors = ai_analysis.get("risk_factors", [])

    # 模拟不同的审批者决策倾向
    import random

    if risk_score >= 0.8:
        # 高风险，通常拒绝
        decisions = ["拒绝", "需要更多信息"]
        decision = random.choice(decisions)
        reason = "风险过高，不建议执行"
    elif risk_score >= 0.6:
        # 中高风险，可能批准
        decisions = ["修改后批准", "需要更多信息", "拒绝"]
        decision = random.choice(decisions)
        reason = "存在一定风险，需要谨慎处理"
    else:
        # 中低风险，通常批准
        decisions = ["批准", "修改后批准"]
        decision = random.choice(decisions)
        reason = "风险可控，可以执行"

    return {
        "reviewer": "高级审批员",
        "decision": decision,
        "reason": reason,
        "review_time": time.time(),
        "considered_factors": risk_factors
    }


def create_content_moderation_system():
    """创建内容审核系统"""

    class ModerationState(TypedDict):
        content: str
        content_type: str
        ai_moderation: dict
        human_review: dict
        moderation_result: str

    def ai_content_analyzer(state: ModerationState) -> ModerationState:
        """AI内容分析器"""
        content = state.get("content", "")
        content_type = state.get("content_type", "text")

        print(f"🔍 AI内容分析: {content_type}")
        print(f"📝 内容预览: {content[:50]}{'...' if len(content) > 50 else ''}")

        # 模拟AI内容分析
        analysis_results = {
            "toxicity_score": 0.0,
            "spam_probability": 0.0,
            "adult_content": False,
            "hate_speech": False,
            "misinformation": False,
            "detected_issues": []
        }

        # 检查有害内容
        harmful_keywords = ["暴力", "仇恨", "歧视", "欺诈"]
        for keyword in harmful_keywords:
            if keyword in content:
                analysis_results["toxicity_score"] += 0.3
                analysis_results["detected_issues"].append(f"包含有害关键词: {keyword}")

        # 检查垃圾内容
        spam_indicators = ["免费", "立即", "点击", "限时"]
        spam_count = sum(1 for indicator in spam_indicators if indicator in content)
        if spam_count >= 2:
            analysis_results["spam_probability"] = min(spam_count * 0.3, 1.0)
            analysis_results["detected_issues"].append("疑似垃圾内容")

        # 检查成人内容
        adult_keywords = ["成人", "限制级"]
        if any(keyword in content for keyword in adult_keywords):
            analysis_results["adult_content"] = True
            analysis_results["detected_issues"].append("包含成人内容")

        # 确定是否需要人工审核
        needs_human_review = (
            analysis_results["toxicity_score"] > 0.5 or
            analysis_results["spam_probability"] > 0.6 or
            analysis_results["adult_content"] or
            len(analysis_results["detected_issues"]) > 0
        )

        ai_moderation = {
            **analysis_results,
            "confidence": 0.85,
            "needs_human_review": needs_human_review,
            "ai_recommendation": "需要人工审核" if needs_human_review else "可以自动通过"
        }

        state["ai_moderation"] = ai_moderation

        print(f"   🤖 AI建议: {ai_moderation['ai_recommendation']}")
        if analysis_results["detected_issues"]:
            print(f"   ⚠️  检测到的问题: {', '.join(analysis_results['detected_issues'])}")

        return state

    def human_moderation_node(state: ModerationState) -> ModerationState:
        """人工审核节点"""
        content = state.get("content", "")
        ai_moderation = state.get("ai_moderation", {})

        print(f"\n👥 人工内容审核")
        print(f"📄 审核内容: {content}")
        print(f"🤖 AI检测结果: {ai_moderation.get('detected_issues', [])}")

        # 模拟人工审核决策
        human_review = simulate_human_moderation(ai_moderation, content)

        state["human_review"] = human_review

        print(f"✅ 人工审核结果: {human_review.get('decision', '未决定')}")
        print(f"💭 审核理由: {human_review.get('reason', '无')}")

        return state

    def moderation_finalizer(state: ModerationState) -> ModerationState:
        """审核结果终结器"""
        ai_moderation = state.get("ai_moderation", {})
        human_review = state.get("human_review", {})

        if ai_moderation.get("needs_human_review", False):
            # 基于人工审核结果
            decision = human_review.get("decision", "拒绝")
            reason = human_review.get("reason", "人工审核")
        else:
            # 基于AI分析自动通过
            decision = "通过"
            reason = "AI自动审核通过"

        moderation_result = f"{decision}: {reason}"
        state["moderation_result"] = moderation_result

        print(f"\n🎯 最终审核结果: {moderation_result}")

        return state

    def moderation_router(state: ModerationState) -> str:
        """审核路由器"""
        ai_moderation = state.get("ai_moderation", {})
        needs_human_review = ai_moderation.get("needs_human_review", False)

        return "human_moderation" if needs_human_review else "auto_approve"

    # 构建内容审核图
    graph = StateGraph(ModerationState)

    graph.add_node("ai_analyzer", ai_content_analyzer)
    graph.add_node("human_moderation", human_moderation_node)
    graph.add_node("auto_approve", moderation_finalizer)
    graph.add_node("finalize", moderation_finalizer)

    graph.set_entry_point("ai_analyzer")

    graph.add_conditional_edges(
        "ai_analyzer",
        moderation_router,
        {
            "human_moderation": "human_moderation",
            "auto_approve": "auto_approve"
        }
    )

    graph.add_edge("human_moderation", "finalize")
    graph.add_edge("auto_approve", END)
    graph.add_edge("finalize", END)

    return graph.compile()


def simulate_human_moderation(ai_moderation: dict, content: str) -> dict:
    """模拟人工审核决策"""
    detected_issues = ai_moderation.get("detected_issues", [])
    toxicity_score = ai_moderation.get("toxicity_score", 0.0)

    import random

    if toxicity_score > 0.7 or len(detected_issues) > 2:
        # 高风险内容，通常拒绝
        decision = "拒绝"
        reason = "内容违反社区准则"
    elif toxicity_score > 0.3 or len(detected_issues) > 0:
        # 中风险内容，可能需要修改
        decisions = ["需要修改", "警告通过", "拒绝"]
        decision = random.choice(decisions)
        reason = "内容存在问题，需要处理"
    else:
        # 低风险内容，通常通过
        decision = "通过"
        reason = "内容符合社区准则"

    return {
        "moderator": "内容审核员",
        "decision": decision,
        "reason": reason,
        "review_timestamp": time.time(),
        "flagged_sections": detected_issues
    }


def demo_human_in_loop():
    """演示Human-in-the-Loop系统"""
    print("🚀 Human-in-the-Loop演示开始")
    print("=" * 60)

    # 演示1: 审批工作流
    print("\n🎯 演示1: 智能审批工作流")
    print("-" * 40)

    approval_app = create_approval_workflow()

    # 测试不同风险级别的任务
    test_tasks = [
        "更新用户信息",
        "删除重要数据库记录",
        "发送营销邮件给所有用户",
        "备份系统数据",
        "永久删除所有机密文件"
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n📋 任务 {i}: {task}")
        print("-" * 30)

        state = {
            "task_description": task,
            "ai_analysis": {},
            "human_review_required": False,
            "human_feedback": {},
            "final_decision": "",
            "approval_chain": [],
            "risk_level": ""
        }

        result = approval_app.invoke(state)

    # 演示2: 内容审核系统
    print("\n\n🎯 演示2: 内容审核系统")
    print("-" * 40)

    moderation_app = create_content_moderation_system()

    # 测试不同类型的内容
    test_contents = [
        {"content": "这是一条正常的消息", "type": "text"},
        {"content": "免费获取！立即点击！限时优惠！", "type": "text"},
        {"content": "我对某个群体表示仇恨和歧视", "type": "text"},
        {"content": "分享一些技术知识和经验", "type": "text"},
        {"content": "这里包含一些成人和暴力内容", "type": "text"}
    ]

    for i, content_item in enumerate(test_contents, 1):
        print(f"\n📄 内容 {i}: {content_item['content']}")
        print("-" * 30)

        state = {
            "content": content_item["content"],
            "content_type": content_item["type"],
            "ai_moderation": {},
            "human_review": {},
            "moderation_result": ""
        }

        result = moderation_app.invoke(state)

    print("\n🎉 Human-in-the-Loop演示完成！")


if __name__ == "__main__":
    demo_human_in_loop()