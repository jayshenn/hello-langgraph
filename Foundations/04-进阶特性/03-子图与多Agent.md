# 03-子图与多Agent

## 🎯 学习目标

通过本章学习，你将掌握：
- 子图（Subgraph）的概念和应用场景
- 多 Agent 系统的设计模式
- 父子图之间的通信机制
- Supervisor 和 Swarm 架构模式

## 📚 基础概念

### 什么是子图？

子图是 LangGraph 中的一个强大特性，允许你将复杂的图分解为更小、更易管理的组件。子图可以：

- **封装复杂逻辑**：将相关的节点组合成一个逻辑单元
- **促进代码复用**：同一个子图可以在多个地方使用
- **简化调试**：独立测试和调试子图功能
- **支持团队协作**：不同团队可以独立开发不同的子图

### 什么是多 Agent 系统？

多 Agent 系统是由多个独立的 AI Agent 协作完成复杂任务的架构。每个 Agent 专注于特定的领域或任务类型。

## 🏗️ 子图架构模式

### 1. 基础子图实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 定义子图状态
class SubgraphState(TypedDict):
    input_data: str
    processing_result: str
    sub_task_complete: bool

def create_data_processing_subgraph():
    """创建数据处理子图"""

    def validate_input(state: SubgraphState) -> SubgraphState:
        """验证输入数据"""
        input_data = state.get("input_data", "")

        if len(input_data) < 10:
            state["validation_error"] = "Input too short"
            state["is_valid"] = False
        else:
            state["is_valid"] = True

        return state

    def process_data(state: SubgraphState) -> SubgraphState:
        """处理数据"""
        if state.get("is_valid", False):
            processed = f"Processed: {state['input_data']}"
            state["processing_result"] = processed
            state["sub_task_complete"] = True
        else:
            state["processing_result"] = "Processing failed due to validation error"
            state["sub_task_complete"] = False

        return state

    def validation_router(state: SubgraphState) -> str:
        """验证路由器"""
        return "process" if state.get("is_valid", False) else END

    # 创建子图
    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("validate", validate_input)
    subgraph.add_node("process", process_data)

    subgraph.set_entry_point("validate")
    subgraph.add_conditional_edges(
        "validate",
        validation_router,
        {
            "process": "process",
            END: END
        }
    )
    subgraph.add_edge("process", END)

    return subgraph.compile()

# 在主图中使用子图
def main_graph_with_subgraph():
    """包含子图的主图"""

    class MainState(TypedDict):
        user_request: str
        sub_results: list
        final_output: str

    def prepare_subgraph_input(state: MainState) -> SubgraphState:
        """准备子图输入"""
        return {"input_data": state["user_request"]}

    def integrate_subgraph_result(state: MainState, sub_result: SubgraphState) -> MainState:
        """整合子图结果"""
        state["sub_results"] = state.get("sub_results", [])
        state["sub_results"].append(sub_result)
        return state

    def use_subgraph_node(state: MainState) -> MainState:
        """使用子图的节点"""
        # 准备子图输入
        sub_input = prepare_subgraph_input(state)

        # 获取已编译的子图
        subgraph = create_data_processing_subgraph()

        # 执行子图
        sub_result = subgraph.invoke(sub_input)

        # 整合结果
        return integrate_subgraph_result(state, sub_result)

    # 创建主图
    main_graph = StateGraph(MainState)
    main_graph.add_node("subgraph_processor", use_subgraph_node)
    main_graph.set_entry_point("subgraph_processor")
    main_graph.add_edge("subgraph_processor", END)

    return main_graph.compile()
```

### 2. 状态映射和转换

```python
def advanced_subgraph_integration():
    """高级子图集成示例"""

    # 主图状态
    class MainState(TypedDict):
        user_query: str
        search_results: list
        analysis_results: list
        final_answer: str

    # 搜索子图状态
    class SearchState(TypedDict):
        query: str
        search_results: list
        search_success: bool

    # 分析子图状态
    class AnalysisState(TypedDict):
        data_to_analyze: list
        analysis_method: str
        analysis_results: dict
        confidence_score: float

    def create_search_subgraph():
        """创建搜索子图"""
        def web_search(state: SearchState) -> SearchState:
            query = state.get("query", "")
            # 模拟搜索
            results = [f"Result {i} for '{query}'" for i in range(3)]
            state["search_results"] = results
            state["search_success"] = len(results) > 0
            return state

        subgraph = StateGraph(SearchState)
        subgraph.add_node("search", web_search)
        subgraph.set_entry_point("search")
        subgraph.add_edge("search", END)
        return subgraph.compile()

    def create_analysis_subgraph():
        """创建分析子图"""
        def analyze_data(state: AnalysisState) -> AnalysisState:
            data = state.get("data_to_analyze", [])
            method = state.get("analysis_method", "basic")

            # 模拟分析
            analysis = {
                "summary": f"Analyzed {len(data)} items using {method}",
                "key_points": data[:2] if data else [],
                "method_used": method
            }

            state["analysis_results"] = analysis
            state["confidence_score"] = 0.85 if len(data) > 2 else 0.6
            return state

        subgraph = StateGraph(AnalysisState)
        subgraph.add_node("analyze", analyze_data)
        subgraph.set_entry_point("analyze")
        subgraph.add_edge("analyze", END)
        return subgraph.compile()

    # 状态转换函数
    def main_to_search_state(main_state: MainState) -> SearchState:
        """主状态转换为搜索状态"""
        return {
            "query": main_state.get("user_query", ""),
            "search_results": [],
            "search_success": False
        }

    def search_to_main_state(main_state: MainState, search_result: SearchState) -> MainState:
        """搜索结果合并到主状态"""
        main_state["search_results"] = search_result.get("search_results", [])
        return main_state

    def main_to_analysis_state(main_state: MainState) -> AnalysisState:
        """主状态转换为分析状态"""
        return {
            "data_to_analyze": main_state.get("search_results", []),
            "analysis_method": "comprehensive",
            "analysis_results": {},
            "confidence_score": 0.0
        }

    def analysis_to_main_state(main_state: MainState, analysis_result: AnalysisState) -> MainState:
        """分析结果合并到主状态"""
        main_state["analysis_results"] = analysis_result.get("analysis_results", {})
        return main_state

    # 主图节点
    def search_node(state: MainState) -> MainState:
        """搜索节点"""
        search_subgraph = create_search_subgraph()
        search_input = main_to_search_state(state)
        search_output = search_subgraph.invoke(search_input)
        return search_to_main_state(state, search_output)

    def analysis_node(state: MainState) -> MainState:
        """分析节点"""
        analysis_subgraph = create_analysis_subgraph()
        analysis_input = main_to_analysis_state(state)
        analysis_output = analysis_subgraph.invoke(analysis_input)
        return analysis_to_main_state(state, analysis_output)

    def synthesis_node(state: MainState) -> MainState:
        """综合节点"""
        search_results = state.get("search_results", [])
        analysis_results = state.get("analysis_results", {})

        final_answer = f"""
        Based on search results: {len(search_results)} items found.
        Analysis summary: {analysis_results.get('summary', 'No analysis available')}
        Key points: {', '.join(analysis_results.get('key_points', []))}
        """

        state["final_answer"] = final_answer.strip()
        return state

    # 构建主图
    main_graph = StateGraph(MainState)
    main_graph.add_node("search", search_node)
    main_graph.add_node("analysis", analysis_node)
    main_graph.add_node("synthesis", synthesis_node)

    main_graph.set_entry_point("search")
    main_graph.add_edge("search", "analysis")
    main_graph.add_edge("analysis", "synthesis")
    main_graph.add_edge("synthesis", END)

    return main_graph.compile()
```

## 🤖 多 Agent 系统架构

### 1. Supervisor 架构

Supervisor 模式中，有一个中央协调者负责任务分配和结果整合：

```python
def create_supervisor_system():
    """创建 Supervisor 多 Agent 系统"""

    class SupervisorState(TypedDict):
        user_request: str
        task_assignments: list
        agent_results: dict
        next_agent: str
        final_result: str

    def create_research_agent():
        """创建研究 Agent"""
        class ResearchState(TypedDict):
            research_query: str
            research_results: list

        def research_node(state: ResearchState) -> ResearchState:
            query = state.get("research_query", "")
            # 模拟研究过程
            results = [f"Research finding {i} for '{query}'" for i in range(3)]
            state["research_results"] = results
            return state

        graph = StateGraph(ResearchState)
        graph.add_node("research", research_node)
        graph.set_entry_point("research")
        graph.add_edge("research", END)
        return graph.compile()

    def create_analysis_agent():
        """创建分析 Agent"""
        class AnalysisState(TypedDict):
            data_to_analyze: list
            analysis_type: str
            analysis_output: dict

        def analysis_node(state: AnalysisState) -> AnalysisState:
            data = state.get("data_to_analyze", [])
            analysis_type = state.get("analysis_type", "general")

            # 模拟分析过程
            output = {
                "insights": [f"Insight from {item}" for item in data[:2]],
                "trends": ["Trend 1", "Trend 2"],
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            }

            state["analysis_output"] = output
            return state

        graph = StateGraph(AnalysisState)
        graph.add_node("analyze", analysis_node)
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    def create_writing_agent():
        """创建写作 Agent"""
        class WritingState(TypedDict):
            content_brief: dict
            writing_style: str
            final_content: str

        def writing_node(state: WritingState) -> WritingState:
            brief = state.get("content_brief", {})
            style = state.get("writing_style", "professional")

            # 模拟写作过程
            content = f"""
            # Report ({style} style)

            ## Insights
            {', '.join(brief.get('insights', []))}

            ## Trends
            {', '.join(brief.get('trends', []))}

            ## Recommendations
            {', '.join(brief.get('recommendations', []))}
            """

            state["final_content"] = content.strip()
            return state

        graph = StateGraph(WritingState)
        graph.add_node("write", writing_node)
        graph.set_entry_point("write")
        graph.add_edge("write", END)
        return graph.compile()

    # Supervisor 逻辑
    def supervisor_node(state: SupervisorState) -> SupervisorState:
        """Supervisor 决策节点"""
        user_request = state.get("user_request", "")
        agent_results = state.get("agent_results", {})

        # 决定下一个要调用的 Agent
        if "research_agent" not in agent_results:
            state["next_agent"] = "research_agent"
        elif "analysis_agent" not in agent_results:
            state["next_agent"] = "analysis_agent"
        elif "writing_agent" not in agent_results:
            state["next_agent"] = "writing_agent"
        else:
            state["next_agent"] = "complete"

        return state

    def research_agent_node(state: SupervisorState) -> SupervisorState:
        """研究 Agent 节点"""
        research_agent = create_research_agent()

        research_input = {
            "research_query": state.get("user_request", ""),
            "research_results": []
        }

        result = research_agent.invoke(research_input)

        agent_results = state.get("agent_results", {})
        agent_results["research_agent"] = result
        state["agent_results"] = agent_results

        return state

    def analysis_agent_node(state: SupervisorState) -> SupervisorState:
        """分析 Agent 节点"""
        analysis_agent = create_analysis_agent()

        research_results = state.get("agent_results", {}).get("research_agent", {})

        analysis_input = {
            "data_to_analyze": research_results.get("research_results", []),
            "analysis_type": "comprehensive",
            "analysis_output": {}
        }

        result = analysis_agent.invoke(analysis_input)

        agent_results = state.get("agent_results", {})
        agent_results["analysis_agent"] = result
        state["agent_results"] = agent_results

        return state

    def writing_agent_node(state: SupervisorState) -> SupervisorState:
        """写作 Agent 节点"""
        writing_agent = create_writing_agent()

        analysis_results = state.get("agent_results", {}).get("analysis_agent", {})

        writing_input = {
            "content_brief": analysis_results.get("analysis_output", {}),
            "writing_style": "professional",
            "final_content": ""
        }

        result = writing_agent.invoke(writing_input)

        agent_results = state.get("agent_results", {})
        agent_results["writing_agent"] = result
        state["agent_results"] = agent_results

        return state

    def completion_node(state: SupervisorState) -> SupervisorState:
        """完成节点"""
        writing_result = state.get("agent_results", {}).get("writing_agent", {})
        state["final_result"] = writing_result.get("final_content", "No content generated")
        return state

    # 路由函数
    def supervisor_router(state: SupervisorState) -> str:
        """Supervisor 路由函数"""
        next_agent = state.get("next_agent", "complete")
        return next_agent

    # 构建 Supervisor 图
    supervisor_graph = StateGraph(SupervisorState)

    supervisor_graph.add_node("supervisor", supervisor_node)
    supervisor_graph.add_node("research_agent", research_agent_node)
    supervisor_graph.add_node("analysis_agent", analysis_agent_node)
    supervisor_graph.add_node("writing_agent", writing_agent_node)
    supervisor_graph.add_node("complete", completion_node)

    supervisor_graph.set_entry_point("supervisor")

    supervisor_graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "research_agent": "research_agent",
            "analysis_agent": "analysis_agent",
            "writing_agent": "writing_agent",
            "complete": "complete"
        }
    )

    # 每个 Agent 完成后回到 Supervisor
    supervisor_graph.add_edge("research_agent", "supervisor")
    supervisor_graph.add_edge("analysis_agent", "supervisor")
    supervisor_graph.add_edge("writing_agent", "supervisor")
    supervisor_graph.add_edge("complete", END)

    return supervisor_graph.compile()
```

### 2. Swarm 架构

Swarm 模式中，Agent 之间可以直接通信和协作：

```python
def create_swarm_system():
    """创建 Swarm 多 Agent 系统"""

    class SwarmState(TypedDict):
        task_queue: list
        active_agents: list
        shared_knowledge: dict
        task_results: dict
        next_action: str

    class AgentMessage(TypedDict):
        from_agent: str
        to_agent: str
        message_type: str
        content: dict
        timestamp: float

    def create_data_collector_agent():
        """数据收集 Agent"""
        def collect_data(state: SwarmState) -> SwarmState:
            # 从任务队列获取任务
            task_queue = state.get("task_queue", [])
            if not task_queue:
                return state

            current_task = task_queue[0]

            # 模拟数据收集
            collected_data = {
                "source": current_task.get("source", "unknown"),
                "data": [f"Data point {i}" for i in range(5)],
                "collection_time": time.time()
            }

            # 更新共享知识库
            shared_knowledge = state.get("shared_knowledge", {})
            shared_knowledge["collected_data"] = collected_data
            state["shared_knowledge"] = shared_knowledge

            # 发送消息给分析 Agent
            message = {
                "from_agent": "data_collector",
                "to_agent": "data_analyzer",
                "message_type": "data_ready",
                "content": {"data_location": "shared_knowledge.collected_data"},
                "timestamp": time.time()
            }

            messages = state.get("agent_messages", [])
            messages.append(message)
            state["agent_messages"] = messages

            return state

        return collect_data

    def create_analyzer_agent():
        """数据分析 Agent"""
        def analyze_data(state: SwarmState) -> SwarmState:
            shared_knowledge = state.get("shared_knowledge", {})
            collected_data = shared_knowledge.get("collected_data")

            if not collected_data:
                return state

            # 模拟数据分析
            analysis_result = {
                "patterns": ["Pattern A", "Pattern B"],
                "anomalies": ["Anomaly 1"],
                "confidence": 0.85,
                "analysis_time": time.time()
            }

            # 更新共享知识库
            shared_knowledge["analysis_result"] = analysis_result
            state["shared_knowledge"] = shared_knowledge

            # 发送消息给报告 Agent
            message = {
                "from_agent": "data_analyzer",
                "to_agent": "report_generator",
                "message_type": "analysis_complete",
                "content": {"analysis_location": "shared_knowledge.analysis_result"},
                "timestamp": time.time()
            }

            messages = state.get("agent_messages", [])
            messages.append(message)
            state["agent_messages"] = messages

            return state

        return analyze_data

    def create_report_generator_agent():
        """报告生成 Agent"""
        def generate_report(state: SwarmState) -> SwarmState:
            shared_knowledge = state.get("shared_knowledge", {})
            analysis_result = shared_knowledge.get("analysis_result")

            if not analysis_result:
                return state

            # 生成报告
            report = f"""
            # Data Analysis Report

            ## Patterns Detected
            {', '.join(analysis_result.get('patterns', []))}

            ## Anomalies Found
            {', '.join(analysis_result.get('anomalies', []))}

            ## Confidence Level
            {analysis_result.get('confidence', 0):.2%}

            Generated at: {time.ctime()}
            """

            # 保存最终结果
            task_results = state.get("task_results", {})
            task_results["final_report"] = report.strip()
            state["task_results"] = task_results

            # 通知所有 Agent 任务完成
            completion_message = {
                "from_agent": "report_generator",
                "to_agent": "all",
                "message_type": "task_complete",
                "content": {"report_location": "task_results.final_report"},
                "timestamp": time.time()
            }

            messages = state.get("agent_messages", [])
            messages.append(completion_message)
            state["agent_messages"] = messages

            state["next_action"] = "complete"

            return state

        return generate_report

    # 消息路由器
    def message_router(state: SwarmState) -> str:
        """基于消息的路由"""
        messages = state.get("agent_messages", [])

        if not messages:
            return "data_collector"

        latest_message = messages[-1]
        message_type = latest_message.get("message_type")

        if message_type == "data_ready":
            return "data_analyzer"
        elif message_type == "analysis_complete":
            return "report_generator"
        elif message_type == "task_complete":
            return "complete"
        else:
            return "data_collector"

    # 构建 Swarm 图
    swarm_graph = StateGraph(SwarmState)

    swarm_graph.add_node("data_collector", create_data_collector_agent())
    swarm_graph.add_node("data_analyzer", create_analyzer_agent())
    swarm_graph.add_node("report_generator", create_report_generator_agent())
    swarm_graph.add_node("complete", lambda state: state)

    swarm_graph.set_entry_point("data_collector")

    swarm_graph.add_conditional_edges(
        "data_collector",
        message_router,
        {
            "data_collector": "data_collector",
            "data_analyzer": "data_analyzer",
            "report_generator": "report_generator",
            "complete": "complete"
        }
    )

    swarm_graph.add_conditional_edges(
        "data_analyzer",
        message_router,
        {
            "data_collector": "data_collector",
            "data_analyzer": "data_analyzer",
            "report_generator": "report_generator",
            "complete": "complete"
        }
    )

    swarm_graph.add_conditional_edges(
        "report_generator",
        message_router,
        {
            "data_collector": "data_collector",
            "data_analyzer": "data_analyzer",
            "report_generator": "report_generator",
            "complete": "complete"
        }
    )

    swarm_graph.add_edge("complete", END)

    return swarm_graph.compile()
```

## 🔄 Agent 间通信机制

### 1. 直接状态共享

```python
class SharedState(TypedDict):
    """共享状态定义"""
    global_context: dict
    agent_outputs: dict
    communication_log: list
    current_phase: str

def agent_communication_example():
    """Agent 通信示例"""

    def agent_a(state: SharedState) -> SharedState:
        """Agent A"""
        # 处理自己的任务
        result_a = {"task": "completed", "data": [1, 2, 3]}

        # 更新共享状态
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["agent_a"] = result_a
        state["agent_outputs"] = agent_outputs

        # 记录通信日志
        log_entry = {
            "timestamp": time.time(),
            "agent": "agent_a",
            "action": "completed_task",
            "message": "Task completed, data available for Agent B"
        }

        comm_log = state.get("communication_log", [])
        comm_log.append(log_entry)
        state["communication_log"] = comm_log

        return state

    def agent_b(state: SharedState) -> SharedState:
        """Agent B"""
        # 检查 Agent A 的输出
        agent_outputs = state.get("agent_outputs", {})
        agent_a_result = agent_outputs.get("agent_a")

        if agent_a_result:
            # 基于 Agent A 的结果进行处理
            agent_a_data = agent_a_result.get("data", [])
            processed_data = [x * 2 for x in agent_a_data]

            result_b = {"processed_data": processed_data, "status": "success"}
        else:
            result_b = {"error": "Agent A data not available", "status": "failed"}

        agent_outputs["agent_b"] = result_b
        state["agent_outputs"] = agent_outputs

        # 记录通信日志
        log_entry = {
            "timestamp": time.time(),
            "agent": "agent_b",
            "action": "processed_agent_a_data",
            "message": f"Processed {len(agent_a_data)} items from Agent A"
        }

        comm_log = state.get("communication_log", [])
        comm_log.append(log_entry)
        state["communication_log"] = comm_log

        return state
```

### 2. 消息队列通信

```python
def message_queue_communication():
    """消息队列通信机制"""

    class MessageQueueState(TypedDict):
        message_queue: list
        agent_inboxes: dict
        processed_messages: list

    def send_message(state: MessageQueueState, message: dict) -> MessageQueueState:
        """发送消息"""
        message_queue = state.get("message_queue", [])
        message_queue.append(message)
        state["message_queue"] = message_queue
        return state

    def deliver_messages(state: MessageQueueState) -> MessageQueueState:
        """分发消息到各 Agent 收件箱"""
        message_queue = state.get("message_queue", [])
        agent_inboxes = state.get("agent_inboxes", {})
        processed_messages = state.get("processed_messages", [])

        for message in message_queue:
            to_agent = message.get("to_agent")
            if to_agent:
                if to_agent not in agent_inboxes:
                    agent_inboxes[to_agent] = []
                agent_inboxes[to_agent].append(message)
                processed_messages.append(message)

        # 清空消息队列
        state["message_queue"] = []
        state["agent_inboxes"] = agent_inboxes
        state["processed_messages"] = processed_messages

        return state

    def agent_with_messaging(agent_name: str):
        """带消息功能的 Agent"""
        def agent_node(state: MessageQueueState) -> MessageQueueState:
            # 检查收件箱
            agent_inboxes = state.get("agent_inboxes", {})
            inbox = agent_inboxes.get(agent_name, [])

            # 处理收到的消息
            for message in inbox:
                message_type = message.get("type")
                content = message.get("content")

                if message_type == "task_request":
                    # 处理任务请求
                    result = f"Task completed by {agent_name}: {content}"

                    # 发送回复消息
                    reply = {
                        "from_agent": agent_name,
                        "to_agent": message.get("from_agent"),
                        "type": "task_result",
                        "content": result,
                        "timestamp": time.time()
                    }

                    state = send_message(state, reply)

            # 清空收件箱
            agent_inboxes[agent_name] = []
            state["agent_inboxes"] = agent_inboxes

            return state

        return agent_node
```

## 🎮 实践练习

### 练习 1：构建客服子图系统

创建一个客服系统，包含以下子图：
- 意图识别子图
- FAQ 查询子图
- 人工转接子图

```python
def customer_service_subgraph_system():
    """
    练习：构建客服子图系统

    要求：
    1. 意图识别子图：分类用户问题类型
    2. FAQ 子图：处理常见问题
    3. 人工转接子图：复杂问题转人工
    4. 主图：协调各个子图的执行
    """
    # TODO: 实现你的客服系统
    pass
```

### 练习 2：设计多 Agent 数据分析系统

创建一个数据分析系统，包含：
- 数据收集 Agent
- 数据清洗 Agent
- 数据分析 Agent
- 报告生成 Agent

```python
def data_analysis_multi_agent():
    """
    练习：设计多 Agent 数据分析系统

    要求：
    1. 使用 Supervisor 模式协调
    2. Agent 间通过共享状态通信
    3. 支持任务失败重试
    4. 生成详细的执行报告
    """
    # TODO: 实现你的多 Agent 系统
    pass
```

## 🔗 最佳实践

### 1. 子图设计原则

- **单一职责**：每个子图专注一个明确的功能
- **接口清晰**：明确定义输入输出状态格式
- **可测试性**：子图应该能够独立测试
- **可复用性**：设计通用的子图组件

### 2. 多 Agent 协调策略

- **避免循环依赖**：设计清晰的 Agent 依赖关系
- **错误传播**：处理 Agent 失败对整个系统的影响
- **负载均衡**：合理分配任务避免某个 Agent 过载
- **监控和日志**：记录 Agent 间的通信和执行状态

## 📚 推荐阅读

- [LangGraph 官方文档 - Subgraphs](https://langchain-ai.github.io/langgraph/concepts/subgraphs/)
- [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- 项目示例：`Agents/` 目录中的多 Agent 实现

---

**下一章预告**：学习完子图和多 Agent 系统后，我们将深入探讨 Memory 机制，了解如何为 Agent 添加记忆能力。