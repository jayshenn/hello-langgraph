# Nodes 和 Edges：LangGraph 的构建要素

> 🎯 **学习目标**：掌握节点和边的使用方法，理解图的构建规则和数据流模式

## 🔵 Nodes（节点）：图的处理单元

### 节点的本质

在 LangGraph 中，**节点（Node）** 是图的基本处理单元，每个节点：
- 接收当前状态作为输入
- 执行特定的业务逻辑
- 返回更新后的状态

```python
from typing import TypedDict

class ChatState(TypedDict):
    user_input: str
    response: str
    confidence: float

# 标准节点函数签名
def my_node(state: ChatState) -> ChatState:
    """
    节点函数的标准格式：
    - 输入：当前状态
    - 输出：更新后的状态
    """
    # 业务逻辑处理
    processed_input = process_input(state["user_input"])

    # 返回更新后的状态
    return {
        **state,  # 保持其他字段不变
        "response": processed_input,
        "confidence": 0.85
    }
```

### 节点函数的规范

#### 1. 纯函数特性

```python
# ✅ 好的节点：纯函数，可预测
def good_node(state: ChatState) -> ChatState:
    # 不依赖外部状态
    result = analyze_text(state["user_input"])

    return {
        **state,
        "analysis_result": result
    }

# ❌ 不好的节点：有副作用
global_counter = 0

def bad_node(state: ChatState) -> ChatState:
    global global_counter
    global_counter += 1  # 副作用：修改全局状态

    # 依赖外部可变状态
    result = f"处理次数: {global_counter}"

    return {**state, "result": result}
```

#### 2. 错误处理

```python
def robust_node(state: ChatState) -> ChatState:
    """健壮的节点实现"""
    try:
        # 主要业务逻辑
        result = risky_operation(state["user_input"])

        return {
            **state,
            "result": result,
            "error": None,
            "status": "success"
        }

    except ValueError as e:
        # 业务逻辑错误
        return {
            **state,
            "error": f"输入错误: {str(e)}",
            "status": "input_error"
        }

    except Exception as e:
        # 系统错误
        return {
            **state,
            "error": f"系统错误: {str(e)}",
            "status": "system_error"
        }
```

#### 3. 条件处理

```python
def conditional_node(state: ChatState) -> ChatState:
    """根据状态条件进行不同处理"""
    user_input = state["user_input"]

    if not user_input or not user_input.strip():
        return {
            **state,
            "error": "输入不能为空",
            "needs_input": True
        }

    if len(user_input) > 1000:
        # 输入过长，需要截断
        return {
            **state,
            "user_input": user_input[:1000],
            "truncated": True,
            "original_length": len(user_input)
        }

    # 正常处理
    return {
        **state,
        "processed": True,
        "processing_time": time.time()
    }
```

### 特殊节点类型

#### 1. 工具调用节点

```python
from langchain.tools import Tool

def tool_calling_node(state: ChatState) -> ChatState:
    """调用外部工具的节点"""
    user_query = state["user_input"]

    # 判断是否需要使用工具
    if "天气" in user_query:
        # 调用天气工具
        weather_result = weather_tool.run(user_query)
        return {
            **state,
            "tool_result": weather_result,
            "tool_used": "weather",
            "needs_response_generation": True
        }

    elif "搜索" in user_query:
        # 调用搜索工具
        search_result = search_tool.run(user_query)
        return {
            **state,
            "tool_result": search_result,
            "tool_used": "search",
            "needs_response_generation": True
        }

    else:
        # 不需要工具
        return {
            **state,
            "tool_result": None,
            "tool_used": None,
            "needs_response_generation": True
        }
```

#### 2. LLM 调用节点

```python
from langchain_openai import ChatOpenAI

def llm_node(state: ChatState) -> ChatState:
    """调用大语言模型的节点"""
    llm = ChatOpenAI(model="gpt-4")

    # 构建提示
    messages = build_messages(state)

    # 调用 LLM
    response = llm.invoke(messages)

    return {
        **state,
        "llm_response": response.content,
        "tokens_used": response.usage_metadata["total_tokens"] if response.usage_metadata else 0
    }

def build_messages(state: ChatState) -> list:
    """构建消息列表"""
    messages = [
        {"role": "system", "content": "你是一个有用的助手"}
    ]

    # 添加历史对话
    for msg in state.get("conversation_history", []):
        messages.append(msg)

    # 添加当前用户输入
    messages.append({"role": "user", "content": state["user_input"]})

    return messages
```

#### 3. 数据处理节点

```python
def data_processing_node(state: ChatState) -> ChatState:
    """数据处理和转换节点"""
    raw_data = state.get("raw_data", [])

    # 数据清洗
    cleaned_data = [
        item for item in raw_data
        if item and isinstance(item, str) and len(item.strip()) > 0
    ]

    # 数据转换
    processed_data = [
        {
            "original": item,
            "cleaned": item.strip().lower(),
            "length": len(item),
            "word_count": len(item.split())
        }
        for item in cleaned_data
    ]

    # 数据统计
    statistics = {
        "total_items": len(processed_data),
        "avg_length": sum(item["length"] for item in processed_data) / len(processed_data) if processed_data else 0,
        "total_words": sum(item["word_count"] for item in processed_data)
    }

    return {
        **state,
        "processed_data": processed_data,
        "data_statistics": statistics,
        "processing_completed": True
    }
```

## 🔗 Edges（边）：图的控制流

### 边的类型

#### 1. 普通边（Regular Edges）

```python
from langgraph.graph import StateGraph

# 创建图
graph = StateGraph(ChatState)

# 添加节点
graph.add_node("input_processing", process_input_node)
graph.add_node("llm_call", llm_node)
graph.add_node("output_formatting", format_output_node)

# 添加普通边：固定的执行顺序
graph.add_edge("input_processing", "llm_call")      # input_processing -> llm_call
graph.add_edge("llm_call", "output_formatting")    # llm_call -> output_formatting
```

#### 2. 条件边（Conditional Edges）

```python
def route_by_input_type(state: ChatState) -> str:
    """路由函数：根据状态决定下一个节点"""
    user_input = state["user_input"].lower()

    if "搜索" in user_input or "查找" in user_input:
        return "search_node"
    elif "天气" in user_input:
        return "weather_node"
    elif "计算" in user_input or "数学" in user_input:
        return "calculation_node"
    else:
        return "general_chat_node"

# 添加条件边
graph.add_conditional_edges(
    "input_analysis",           # 源节点
    route_by_input_type,        # 路由函数
    {
        "search_node": "search_node",
        "weather_node": "weather_node",
        "calculation_node": "calculation_node",
        "general_chat_node": "general_chat_node"
    }
)
```

#### 3. 动态边（Send API）

```python
from langgraph.constants import Send

def fan_out_processing(state: ChatState) -> list[Send]:
    """动态创建多个并行任务"""
    queries = state.get("parallel_queries", [])
    sends = []

    for i, query in enumerate(queries):
        # 为每个查询创建一个独立的处理任务
        sends.append(
            Send("process_single_query", {
                "query": query,
                "query_id": i,
                "timestamp": time.time()
            })
        )

    return sends

def process_single_query(state: dict) -> ChatState:
    """处理单个查询"""
    query = state["query"]
    query_id = state["query_id"]

    result = process_query(query)

    return {
        "parallel_results": [{
            "query_id": query_id,
            "query": query,
            "result": result,
            "processed_at": time.time()
        }]
    }

# 添加动态边
graph.add_conditional_edges("fan_out", fan_out_processing)
graph.add_node("process_single_query", process_single_query)
```

### 复杂路由模式

#### 1. 多条件路由

```python
def complex_router(state: ChatState) -> str:
    """复杂的路由逻辑"""
    user_input = state["user_input"]
    user_history = state.get("conversation_history", [])
    user_preferences = state.get("user_preferences", {})

    # 条件 1：紧急情况检测
    if any(keyword in user_input.lower() for keyword in ["紧急", "急", "help", "sos"]):
        return "emergency_handler"

    # 条件 2：基于历史对话的上下文
    if len(user_history) > 0 and "继续" in user_input:
        last_topic = extract_topic(user_history[-1])
        if last_topic:
            return f"continue_{last_topic}"

    # 条件 3：基于用户偏好
    preferred_style = user_preferences.get("response_style", "standard")
    if preferred_style == "detailed":
        return "detailed_response"
    elif preferred_style == "concise":
        return "concise_response"

    # 条件 4：内容类型检测
    content_type = detect_content_type(user_input)
    if content_type == "question":
        return "qa_handler"
    elif content_type == "request":
        return "request_handler"

    # 默认路由
    return "general_handler"

graph.add_conditional_edges(
    "input_analysis",
    complex_router,
    {
        "emergency_handler": "emergency_handler",
        "continue_research": "research_continuation",
        "continue_coding": "coding_continuation",
        "detailed_response": "detailed_processor",
        "concise_response": "concise_processor",
        "qa_handler": "question_answering",
        "request_handler": "request_processing",
        "general_handler": "general_processing"
    }
)
```

#### 2. 循环和重试逻辑

```python
def retry_logic(state: ChatState) -> str:
    """重试逻辑路由"""
    error_count = state.get("error_count", 0)
    max_retries = state.get("max_retries", 3)
    last_error = state.get("last_error")

    # 如果没有错误，继续正常流程
    if not last_error:
        return "next_step"

    # 如果重试次数已达上限，进入错误处理
    if error_count >= max_retries:
        return "error_handler"

    # 根据错误类型决定重试策略
    if "network" in last_error.lower():
        return "network_retry"
    elif "rate_limit" in last_error.lower():
        return "rate_limit_retry"
    else:
        return "general_retry"

def processing_node_with_retry(state: ChatState) -> ChatState:
    """带重试机制的处理节点"""
    try:
        result = risky_operation(state["user_input"])
        return {
            **state,
            "result": result,
            "error_count": 0,  # 重置错误计数
            "last_error": None
        }
    except Exception as e:
        return {
            **state,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": str(e)
        }

# 添加重试循环
graph.add_conditional_edges(
    "processing_node",
    retry_logic,
    {
        "next_step": "success_handler",
        "error_handler": "final_error_handler",
        "network_retry": "processing_node",     # 循环回自己
        "rate_limit_retry": "wait_and_retry",
        "general_retry": "processing_node"
    }
)
```

## 🚀 START 和 END 节点

### START 节点

```python
from langgraph.graph import START, END

# 方式 1：使用 set_entry_point
graph.set_entry_point("first_node")

# 方式 2：使用 START 常量（推荐）
graph.add_edge(START, "first_node")

# 方式 3：多个入口点
def entry_router(state: ChatState) -> str:
    """入口路由"""
    if state.get("is_admin"):
        return "admin_flow"
    else:
        return "user_flow"

graph.add_conditional_edges(
    START,
    entry_router,
    {
        "admin_flow": "admin_node",
        "user_flow": "user_node"
    }
)
```

### END 节点

```python
# 方式 1：使用 set_finish_point
graph.set_finish_point("final_node")

# 方式 2：使用 END 常量（推荐）
graph.add_edge("final_node", END)

# 方式 3：条件结束
def should_continue(state: ChatState) -> str:
    """判断是否应该继续"""
    if state.get("is_complete"):
        return "end"
    elif state.get("needs_human_review"):
        return "human_review"
    else:
        return "continue_processing"

graph.add_conditional_edges(
    "decision_node",
    should_continue,
    {
        "end": END,
        "human_review": "human_review_node",
        "continue_processing": "next_processing_node"
    }
)
```

## 🔄 数据流模式

### 1. 线性流程

```python
"""
用户输入 → 预处理 → LLM调用 → 后处理 → 输出
"""

graph.add_edge(START, "preprocessing")
graph.add_edge("preprocessing", "llm_call")
graph.add_edge("llm_call", "postprocessing")
graph.add_edge("postprocessing", END)
```

### 2. 分支流程

```python
"""
              → 搜索处理 →
用户输入 → 分析 → 天气处理 → 结果汇总 → 输出
              → 计算处理 →
"""

def route_by_type(state: ChatState) -> str:
    intent = state["intent"]
    return f"{intent}_handler"

graph.add_edge(START, "intent_analysis")
graph.add_conditional_edges(
    "intent_analysis",
    route_by_type,
    {
        "search_handler": "search_handler",
        "weather_handler": "weather_handler",
        "calculation_handler": "calculation_handler"
    }
)

# 所有分支汇总到同一个节点
graph.add_edge("search_handler", "result_aggregation")
graph.add_edge("weather_handler", "result_aggregation")
graph.add_edge("calculation_handler", "result_aggregation")
graph.add_edge("result_aggregation", END)
```

### 3. 并行流程

```python
"""
用户输入 → 分发 → 搜索API ↘
                → 数据库查询 → 结果合并 → 输出
                → 缓存查找 ↗
"""

def parallel_dispatch(state: ChatState) -> list[Send]:
    """并行分发到多个处理节点"""
    query = state["user_input"]

    return [
        Send("search_api", {"query": query, "source": "api"}),
        Send("search_database", {"query": query, "source": "database"}),
        Send("search_cache", {"query": query, "source": "cache"})
    ]

graph.add_edge(START, "input_validation")
graph.add_conditional_edges("input_validation", parallel_dispatch)

# 并行节点自动汇总结果（通过 reducer）
graph.add_edge("search_api", "result_merger")
graph.add_edge("search_database", "result_merger")
graph.add_edge("search_cache", "result_merger")
graph.add_edge("result_merger", END)
```

### 4. 循环流程

```python
"""
用户输入 → 处理 → 验证 → [失败] → 重新处理
                    ↓
                  [成功] → 输出
"""

def validation_router(state: ChatState) -> str:
    """验证结果路由"""
    if state.get("validation_passed"):
        return "success"
    elif state.get("retry_count", 0) < 3:
        return "retry"
    else:
        return "failed"

graph.add_edge(START, "initial_processing")
graph.add_edge("initial_processing", "validation")

graph.add_conditional_edges(
    "validation",
    validation_router,
    {
        "success": "output_formatting",
        "retry": "retry_processing",      # 循环回到处理
        "failed": "error_handling"
    }
)

graph.add_edge("retry_processing", "validation")  # 重试后重新验证
graph.add_edge("output_formatting", END)
graph.add_edge("error_handling", END)
```

## 🎪 实战案例：智能客服系统

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

class CustomerServiceState(TypedDict):
    user_input: str
    intent: Literal["order", "technical", "billing", "general"]
    confidence: float
    user_id: str
    session_id: str

    # 处理结果
    query_result: Optional[dict]
    response: str
    escalate_to_human: bool

    # 上下文信息
    conversation_history: List[dict]
    user_profile: Optional[dict]

def analyze_intent(state: CustomerServiceState) -> CustomerServiceState:
    """分析用户意图"""
    intent, confidence = intent_classifier(state["user_input"])

    return {
        **state,
        "intent": intent,
        "confidence": confidence
    }

def handle_order_inquiry(state: CustomerServiceState) -> CustomerServiceState:
    """处理订单查询"""
    user_id = state["user_id"]
    query = state["user_input"]

    order_info = order_system.query(user_id, query)

    return {
        **state,
        "query_result": order_info,
        "response": format_order_response(order_info)
    }

def handle_technical_support(state: CustomerServiceState) -> CustomerServiceState:
    """处理技术支持"""
    query = state["user_input"]
    kb_result = knowledge_base.search(query)

    if kb_result["confidence"] > 0.8:
        response = generate_technical_response(kb_result)
        escalate = False
    else:
        response = "让我为您转接技术专家..."
        escalate = True

    return {
        **state,
        "query_result": kb_result,
        "response": response,
        "escalate_to_human": escalate
    }

def route_by_intent(state: CustomerServiceState) -> str:
    """根据意图路由"""
    intent = state["intent"]
    confidence = state["confidence"]

    # 低置信度需要澄清
    if confidence < 0.6:
        return "clarification"

    # 根据意图路由
    if intent == "order":
        return "order_handling"
    elif intent == "technical":
        return "technical_handling"
    elif intent == "billing":
        return "billing_handling"
    else:
        return "general_handling"

def escalation_check(state: CustomerServiceState) -> str:
    """检查是否需要人工介入"""
    if state.get("escalate_to_human"):
        return "human_handoff"
    else:
        return "response_delivery"

# 构建客服图
customer_service_graph = StateGraph(CustomerServiceState)

# 添加节点
customer_service_graph.add_node("intent_analysis", analyze_intent)
customer_service_graph.add_node("clarification", ask_clarification)
customer_service_graph.add_node("order_handling", handle_order_inquiry)
customer_service_graph.add_node("technical_handling", handle_technical_support)
customer_service_graph.add_node("billing_handling", handle_billing_inquiry)
customer_service_graph.add_node("general_handling", handle_general_query)
customer_service_graph.add_node("human_handoff", escalate_to_human)
customer_service_graph.add_node("response_delivery", deliver_response)

# 添加边
customer_service_graph.add_edge(START, "intent_analysis")

customer_service_graph.add_conditional_edges(
    "intent_analysis",
    route_by_intent,
    {
        "clarification": "clarification",
        "order_handling": "order_handling",
        "technical_handling": "technical_handling",
        "billing_handling": "billing_handling",
        "general_handling": "general_handling"
    }
)

# 所有处理节点都需要检查是否升级
for node in ["order_handling", "technical_handling", "billing_handling", "general_handling"]:
    customer_service_graph.add_conditional_edges(
        node,
        escalation_check,
        {
            "human_handoff": "human_handoff",
            "response_delivery": "response_delivery"
        }
    )

customer_service_graph.add_edge("clarification", "intent_analysis")  # 澄清后重新分析
customer_service_graph.add_edge("human_handoff", END)
customer_service_graph.add_edge("response_delivery", END)

# 编译图
customer_service_app = customer_service_graph.compile()
```

## 💡 最佳实践

### ✅ 节点设计最佳实践

1. **单一职责**：每个节点只负责一个明确的功能
2. **纯函数**：避免副作用，便于测试和调试
3. **错误处理**：优雅地处理异常情况
4. **状态完整性**：确保返回的状态包含必要的字段
5. **文档化**：为节点添加清晰的文档说明

### ✅ 边设计最佳实践

1. **清晰的路由逻辑**：路由函数应该简单明了
2. **完整的路由覆盖**：确保所有可能的状态都有对应的路由
3. **合理的默认路由**：提供合理的默认处理路径
4. **避免死循环**：在循环边中包含退出条件
5. **性能考虑**：避免过于复杂的路由计算

### ❌ 常见陷阱

1. **节点间的紧耦合**：避免节点直接依赖其他节点的内部实现
2. **状态污染**：避免在节点中添加不相关的状态字段
3. **过度复杂的路由**：保持路由逻辑的简单性
4. **缺少错误处理**：忽略异常情况的处理
5. **状态不一致**：返回的状态结构与定义不符

## 🚀 下一步

掌握了节点和边的使用后，接下来学习：
- `04-Graph编译与执行.md` - 了解图的运行机制
- `示例代码/simple_graph.py` - 查看完整的实现示例

---

*节点和边是构建 LangGraph 应用的基础，理解它们的使用模式是成功的关键！* 🔧