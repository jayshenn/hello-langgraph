# LangGraph 常见问题解答 (FAQ)

> 🔍 **使用指南**：这里收集了学习和使用 LangGraph 过程中最常见的问题和解答

## 🏗️ 基础概念问题

### Q1: LangGraph 和 LangChain 有什么区别？

**A:**
- **LangChain**：适合线性的、预定义的处理流程，如简单的 RAG 应用
- **LangGraph**：适合复杂的、动态的工作流，支持条件分支、循环和状态管理

```python
# LangChain 风格（链式）
chain = prompt | llm | parser

# LangGraph 风格（图式）
graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_node)
graph.add_conditional_edges("analyze", router, {"route1": "node1", "route2": "node2"})
```

### Q2: 什么时候应该使用 LangGraph？

**A:** 在以下情况下考虑使用 LangGraph：
- ✅ 需要复杂的条件逻辑
- ✅ 需要工具调用和多步推理
- ✅ 需要人机协作（Human-in-the-Loop）
- ✅ 需要状态管理和记忆
- ✅ 需要错误恢复和重试机制

### Q3: AgentState 必须是 TypedDict 吗？

**A:** 是的，LangGraph 要求状态是 TypedDict。这提供了：
- 类型安全
- 更好的 IDE 支持
- 清晰的状态结构定义

```python
# ✅ 正确
class MyState(TypedDict):
    message: str
    count: int

# ❌ 错误
class MyState:
    def __init__(self):
        self.message = ""
        self.count = 0
```

## 🔧 技术实现问题

### Q4: 如何在节点间传递复杂数据？

**A:** 将复杂数据作为状态的一部分：

```python
class ComplexState(TypedDict):
    user_input: str
    search_results: List[Dict[str, Any]]  # 复杂数据
    metadata: Dict[str, str]

def search_node(state: ComplexState) -> ComplexState:
    results = search_api.search(state["user_input"])
    return {
        **state,
        "search_results": results,
        "metadata": {"last_search": "2024-01-01"}
    }
```

### Q5: 如何实现循环和重试逻辑？

**A:** 使用条件边让节点指向自己或其他节点：

```python
def should_retry(state: MyState) -> str:
    if state["error_count"] < 3 and state["has_error"]:
        return "retry_node"  # 重试
    elif state["has_error"]:
        return "error_handler"  # 最终错误处理
    else:
        return "success_node"  # 成功

graph.add_conditional_edges("process_node", should_retry, {
    "retry_node": "process_node",  # 循环
    "error_handler": "error_handler",
    "success_node": "success_node"
})
```

### Q6: 如何处理节点中的异常？

**A:** 在节点函数中使用 try-except，并在状态中记录错误信息：

```python
def safe_node(state: MyState) -> MyState:
    try:
        result = risky_operation(state["input"])
        return {**state, "result": result, "error": None}
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "error_count": state.get("error_count", 0) + 1
        }
```

## 🚀 性能和优化问题

### Q7: 如何优化 LangGraph 的性能？

**A:** 几个优化建议：

1. **减少状态大小**：只在状态中保存必要的数据
2. **使用流式处理**：对于长时间运行的操作
3. **并行化**：利用 LangGraph 的并行执行能力
4. **缓存**：缓存昂贵的计算结果

```python
# 并行执行示例
graph.add_node("parallel1", node1)
graph.add_node("parallel2", node2)
graph.add_edge("start", "parallel1")
graph.add_edge("start", "parallel2")
graph.add_edge(["parallel1", "parallel2"], "merge")
```

### Q8: 状态太大怎么办？

**A:** 几种处理方式：

1. **状态分离**：将大数据存储在外部，状态中只保存引用
2. **数据压缩**：使用压缩算法
3. **分页处理**：对于列表数据使用分页

```python
class OptimizedState(TypedDict):
    user_input: str
    large_data_id: str  # 只存储 ID，不存储实际数据

def get_large_data(data_id: str):
    # 从外部存储获取数据
    return external_storage.get(data_id)
```

## 🐛 调试和错误问题

### Q9: 如何调试 LangGraph 的执行流程？

**A:** 几种调试方法：

1. **状态日志**：在每个节点打印状态
2. **图可视化**：查看图的结构
3. **断点调试**：使用 IDE 的断点功能

```python
def debug_node(state: MyState) -> MyState:
    print(f"🔍 当前状态: {state}")
    # 实际处理逻辑
    result = process(state)
    print(f"🔍 处理结果: {result}")
    return result
```

### Q10: 图执行陷入无限循环怎么办？

**A:** 预防和解决方法：

1. **设置最大步数**：在状态中添加步数计数器
2. **循环检测**：记录访问过的节点
3. **超时机制**：设置执行超时

```python
class SafeState(TypedDict):
    step_count: int
    max_steps: int
    visited_nodes: List[str]

def safe_router(state: SafeState) -> str:
    if state["step_count"] >= state["max_steps"]:
        return "end"  # 强制结束
    # 正常路由逻辑
```

## 📚 最佳实践问题

### Q11: 如何设计好的状态结构？

**A:** 状态设计原则：

1. **最小化**：只包含必要的信息
2. **类型化**：使用具体的类型注解
3. **可扩展**：为未来的功能留出空间
4. **语义化**：使用有意义的字段名

```python
class WellDesignedState(TypedDict):
    # 输入数据
    user_input: str

    # 处理状态
    current_step: Literal["analyzing", "processing", "generating"]

    # 结果数据
    analysis_result: Optional[Dict[str, Any]]

    # 元数据
    timestamp: str
    session_id: str
```

### Q12: 如何组织大型 LangGraph 项目？

**A:** 项目组织建议：

```
project/
├── graphs/
│   ├── main_graph.py      # 主图定义
│   └── subgraphs/         # 子图模块
├── nodes/
│   ├── analysis.py        # 分析节点
│   ├── generation.py      # 生成节点
│   └── tools.py          # 工具节点
├── states/
│   ├── base.py           # 基础状态
│   └── specific.py       # 特定状态
├── utils/
│   ├── validators.py     # 状态验证
│   └── helpers.py        # 辅助函数
└── config/
    ├── settings.py       # 配置
    └── prompts.py        # 提示词
```

## 🔧 工具和集成问题

### Q13: 如何在 LangGraph 中使用 LangChain 的工具？

**A:** 直接在节点中调用 LangChain 工具：

```python
from langchain.tools import DuckDuckGoSearchRun

def search_node(state: MyState) -> MyState:
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.run(state["query"])

    return {**state, "search_results": results}
```

### Q14: 如何实现 Human-in-the-Loop？

**A:** 使用 LangGraph 的 interrupt 功能：

```python
from langgraph.prebuilt import interrupt

def review_node(state: MyState) -> MyState:
    if needs_human_review(state):
        human_input = interrupt("请审核这个内容...")
        return {**state, "human_feedback": human_input}
    return state
```

## 🚀 部署和生产问题

### Q15: 如何将 LangGraph 部署到生产环境？

**A:** 几种部署方式：

1. **LangGraph Platform**：官方云平台
2. **自托管**：使用 Docker 容器化
3. **无服务器**：部署到 AWS Lambda 等

```python
# Docker 部署示例
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Q16: 如何监控 LangGraph 应用的性能？

**A:** 监控建议：

1. **日志记录**：记录每个节点的执行时间
2. **指标收集**：统计成功率、错误率
3. **链路追踪**：使用 LangSmith 或其他工具

```python
import time
import logging

def monitored_node(state: MyState) -> MyState:
    start_time = time.time()
    try:
        result = process(state)
        logging.info(f"节点执行成功，耗时: {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        logging.error(f"节点执行失败: {e}")
        raise
```

## 💡 学习建议

### Q17: 学习 LangGraph 的最佳路径是什么？

**A:** 推荐学习路径：

1. **基础概念**：理解图、状态、节点的概念
2. **简单示例**：从 Hello World 开始
3. **逐步复杂**：添加条件、循环、工具
4. **实际项目**：构建完整的应用
5. **高级特性**：学习 HIL、流式处理等

### Q18: 有哪些学习资源推荐？

**A:** 学习资源：

- 📚 **官方文档**：最权威的参考
- 🎥 **视频教程**：YouTube 上的实战案例
- 💻 **开源项目**：GitHub 上的实际应用
- 🤝 **社区论坛**：Discord、Reddit 讨论
- 📖 **本 Foundations 文件夹**：系统化的学习材料

## 🏗️ 架构选择问题

### Q19: 什么时候用 Graph API，什么时候用 Functional API？

**A:** 选择标准：

**Graph API 适用场景**：
- ✅ 复杂的条件分支和循环逻辑
- ✅ 需要可视化工作流程
- ✅ 多团队协作开发
- ✅ 需要动态修改图结构

**Functional API 适用场景**：
- ✅ 简单的线性工作流
- ✅ 更接近传统函数编程
- ✅ 快速原型开发
- ✅ 数据科学和分析任务

```python
# Graph API 示例
graph = StateGraph(MyState)
graph.add_node("process", process_node)
graph.add_conditional_edges("process", router, {...})

# Functional API 示例
@entrypoint()
def my_workflow(input_data: str) -> str:
    cleaned = clean_data(input_data)
    processed = process_data(cleaned)
    return generate_result(processed)
```

### Q20: LangGraph vs LangChain vs CrewAI vs AutoGen？

**A:** 框架对比：

| 特性 | LangGraph | LangChain | CrewAI | AutoGen |
|------|-----------|-----------|---------|---------|
| **复杂性** | 中等 | 简单 | 中等 | 复杂 |
| **状态管理** | ✅ 强大 | ❌ 基础 | ✅ 好 | ✅ 好 |
| **可视化** | ✅ 优秀 | ❌ 无 | ✅ 好 | ✅ 好 |
| **多智能体** | ✅ 支持 | ❌ 基础 | ✅ 专门 | ✅ 专门 |
| **学习曲线** | 中等 | 简单 | 中等 | 陡峭 |

**选择建议**：
- **LangGraph**：需要复杂工作流和状态管理
- **LangChain**：简单的链式处理
- **CrewAI**：专注多智能体协作
- **AutoGen**：研究级别的对话系统

---

## ⚡ 性能优化问题

### Q21: 如何实现并行执行优化？

**A:** 并行优化策略：

```python
# 1. 节点并行执行
graph.add_edge(START, "node1")
graph.add_edge(START, "node2")  # node1 和 node2 并行执行
graph.add_edge(["node1", "node2"], "merge_node")

# 2. 使用 Send API 进行动态并行
from langgraph.graph import Send

def fan_out(state: MyState) -> List[Send]:
    return [
        Send("process_item", {"item": item, "batch_id": i})
        for i, item in enumerate(state["items"])
    ]

# 3. 异步节点优化
async def async_node(state: MyState) -> MyState:
    # 并发执行多个异步任务
    tasks = [
        fetch_data_a(state["input"]),
        fetch_data_b(state["input"]),
        fetch_data_c(state["input"])
    ]
    results = await asyncio.gather(*tasks)
    return {**state, "results": results}
```

### Q22: 如何优化流式处理性能？

**A:** 流式优化技巧：

```python
# 1. 选择合适的流模式
# 对于实时反馈：使用 "updates" 模式
for chunk in app.stream(input, stream_mode="updates"):
    print(f"节点更新: {chunk}")

# 对于状态监控：使用 "values" 模式
for chunk in app.stream(input, stream_mode="values"):
    print(f"当前状态: {chunk}")

# 2. 缓冲和批处理
class BufferedState(TypedDict):
    buffer: List[str]
    batch_size: int

def buffered_processing_node(state: BufferedState) -> BufferedState:
    buffer = state.get("buffer", [])
    batch_size = state.get("batch_size", 10)

    if len(buffer) >= batch_size:
        # 批量处理
        processed = batch_process(buffer)
        return {**state, "buffer": [], "processed": processed}

    return state
```

### Q23: 如何管理大状态对象？

**A:** 大状态优化策略：

```python
# 1. 状态分片存储
class OptimizedState(TypedDict):
    # 只保存引用
    large_data_ref: str
    metadata: Dict[str, Any]

# 外部存储管理器
class StateStorage:
    def __init__(self):
        self.storage = {}

    def store(self, key: str, data: Any) -> str:
        ref_id = f"ref_{hash(key)}"
        self.storage[ref_id] = data
        return ref_id

    def retrieve(self, ref_id: str) -> Any:
        return self.storage.get(ref_id)

# 2. 延迟加载
def lazy_loading_node(state: OptimizedState) -> OptimizedState:
    # 只在需要时加载大数据
    if state.get("need_large_data"):
        large_data = storage.retrieve(state["large_data_ref"])
        # 处理大数据...

    return state
```

---

## 🚀 部署相关问题

### Q24: 如何选择部署方式？

**A:** 部署方式对比：

| 部署方式 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **Cloud SaaS** | 快速上线、小团队 | 零运维、快速扩展 | 成本较高、定制限制 |
| **自托管数据面** | 数据安全要求高 | 数据控制、成本优化 | 需要运维技能 |
| **完全自托管** | 企业级应用 | 完全控制、高度定制 | 运维复杂度高 |
| **Standalone Container** | 边缘计算、离线部署 | 独立运行、无依赖 | 功能有限 |

```python
# 不同部署环境的配置示例

# 1. Cloud SaaS 配置
{
    "deployment": "cloud",
    "scaling": "auto",
    "monitoring": "enabled"
}

# 2. 自托管配置
{
    "deployment": "self-hosted",
    "database": "postgresql://localhost:5432/langgraph",
    "redis": "redis://localhost:6379",
    "scaling": {
        "min_replicas": 2,
        "max_replicas": 10
    }
}
```

### Q25: 如何处理高并发访问？

**A:** 高并发优化策略：

```python
# 1. 连接池配置
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(
    "sqlite:///checkpoints.db",
    pool_size=20,
    max_overflow=30
)

# 2. 缓存策略
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            cache_key = f"result_{hash(str(state))}"
            cached = redis_client.get(cache_key)

            if cached:
                return json.loads(cached)

            result = func(state)
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result)
            )
            return result
        return wrapper
    return decorator

# 3. 负载均衡配置
# nginx.conf
upstream langgraph_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://langgraph_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔧 调试和开发问题

### Q26: 如何使用 LangGraph Studio 调试？

**A:** Studio 调试技巧：

```python
# 1. 启用详细调试信息
app = graph.compile(debug=True)

# 2. 设置断点
app = graph.compile(
    interrupt_before=["problematic_node"],
    interrupt_after=["checkpoint_node"]
)

# 3. 状态检查
config = {"configurable": {"thread_id": "debug_session"}}

# 执行到断点
result = app.invoke(input_data, config=config)

# 检查状态
state = app.get_state(config)
print(f"当前状态: {state.values}")
print(f"下一步: {state.next}")

# 4. 单步执行
# 修改状态
app.update_state(config, {"custom_field": "debug_value"})

# 继续执行
result = app.invoke(None, config=config)
```

### Q27: 如何集成 LangSmith 监控？

**A:** LangSmith 集成方法：

```python
import os
from langsmith import traceable

# 设置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"

# 自动追踪节点
@traceable
def traced_node(state: MyState) -> MyState:
    # 这个节点会自动被 LangSmith 追踪
    return process_data(state)

# 自定义追踪信息
@traceable(
    name="custom_processing",
    tags=["processing", "v1.0"],
    metadata={"version": "1.0", "environment": "production"}
)
def custom_traced_node(state: MyState) -> MyState:
    return advanced_process(state)

# 运行时标签
config = {
    "tags": ["user_session", "experimental"],
    "metadata": {"user_id": "123", "session_id": "abc"}
}

result = app.invoke(input_data, config=config)
```

---

## 🔐 安全和认证问题

### Q28: 如何实现自定义认证？

**A:** 认证实现方案：

```python
# 1. JWT 认证
import jwt
from datetime import datetime, timedelta

class JWTAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, user_id: str) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token 已过期")
        except jwt.InvalidTokenError:
            raise ValueError("无效的 Token")

# 2. 认证节点
def auth_node(state: MyState) -> MyState:
    auth_header = state.get("auth_header", "")

    if not auth_header.startswith("Bearer "):
        return {**state, "auth_error": "缺少认证信息"}

    token = auth_header[7:]  # 移除 "Bearer " 前缀

    try:
        auth = JWTAuth("your-secret-key")
        payload = auth.verify_token(token)
        return {
            **state,
            "user_id": payload["user_id"],
            "authenticated": True
        }
    except ValueError as e:
        return {**state, "auth_error": str(e), "authenticated": False}

# 3. 权限检查
def permission_check(required_permission: str):
    def check_node(state: MyState) -> MyState:
        user_permissions = get_user_permissions(state["user_id"])

        if required_permission not in user_permissions:
            return {**state, "permission_error": "权限不足"}

        return {**state, "permission_granted": True}

    return check_node
```

### Q29: 如何保护敏感数据？

**A:** 数据保护策略：

```python
import hashlib
from cryptography.fernet import Fernet

class DataProtection:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)

    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def hash_pii(self, pii: str) -> str:
        """对 PII 进行哈希处理"""
        return hashlib.sha256(pii.encode()).hexdigest()

# 数据清理节点
def sanitize_data_node(state: MyState) -> MyState:
    """清理和保护敏感数据"""
    protection = DataProtection(Fernet.generate_key())

    # 识别敏感字段
    sensitive_fields = ["email", "phone", "ssn", "credit_card"]

    sanitized_data = {}
    for key, value in state.items():
        if key in sensitive_fields:
            # 加密敏感数据
            sanitized_data[f"{key}_encrypted"] = protection.encrypt_sensitive_data(value)
            # 创建哈希版本用于匹配
            sanitized_data[f"{key}_hash"] = protection.hash_pii(value)
        else:
            sanitized_data[key] = value

    return sanitized_data
```

---

## 🔄 版本兼容问题

### Q30: LangGraph 版本升级指南？

**A:** 版本升级策略：

```python
# 1. 检查版本兼容性
import langgraph
import langchain

print(f"LangGraph 版本: {langgraph.__version__}")
print(f"LangChain 版本: {langchain.__version__}")

# 版本兼容性矩阵
VERSION_COMPATIBILITY = {
    "langgraph-0.1.x": "langchain-0.1.x",
    "langgraph-0.2.x": "langchain-0.2.x",
    # ... 其他版本映射
}

# 2. 渐进式迁移
def migrate_state_v1_to_v2(old_state: dict) -> dict:
    """从 v1 状态格式迁移到 v2"""
    new_state = {}

    # 字段重命名
    if "old_field_name" in old_state:
        new_state["new_field_name"] = old_state["old_field_name"]

    # 数据格式转换
    if "messages" in old_state:
        # v2 使用新的消息格式
        new_state["messages"] = convert_message_format(old_state["messages"])

    return new_state

# 3. 兼容性包装器
class CompatibilityWrapper:
    def __init__(self, graph_v1):
        self.graph_v1 = graph_v1

    def invoke(self, input_data: dict) -> dict:
        # 转换输入格式
        converted_input = migrate_state_v1_to_v2(input_data)

        # 调用新版本 API
        result = self.graph_v1.invoke(converted_input)

        # 转换输出格式（如果需要）
        return result
```

---

## 💡 高级使用技巧

### Q31: 如何实现智能体的学习和改进？

**A:** 学习机制实现：

```python
# 1. 反馈收集系统
class FeedbackSystem:
    def __init__(self):
        self.feedback_db = {}

    def collect_feedback(self, session_id: str, rating: int, comments: str):
        """收集用户反馈"""
        self.feedback_db[session_id] = {
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_feedback(self) -> Dict[str, Any]:
        """分析反馈数据"""
        ratings = [f["rating"] for f in self.feedback_db.values()]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            "average_rating": avg_rating,
            "total_feedback": len(ratings),
            "improvement_needed": avg_rating < 3.5
        }

# 2. 自适应提示词调整
def adaptive_prompt_node(state: MyState) -> MyState:
    """根据历史表现调整提示词"""
    feedback_system = FeedbackSystem()
    analysis = feedback_system.analyze_feedback()

    if analysis["improvement_needed"]:
        # 使用更详细的提示词
        prompt = "请详细分析并提供step-by-step的解决方案..."
    else:
        # 使用简洁的提示词
        prompt = "请简洁地回答..."

    return {**state, "adaptive_prompt": prompt}

# 3. 性能监控和优化
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record_metric(self, node_name: str, execution_time: float, success: bool):
        """记录节点性能指标"""
        self.metrics[node_name].append({
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now()
        })

    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []

        for node_name, metrics in self.metrics.items():
            avg_time = sum(m["execution_time"] for m in metrics) / len(metrics)
            success_rate = sum(m["success"] for m in metrics) / len(metrics)

            if avg_time > 5.0:  # 超过5秒
                suggestions.append(f"{node_name}: 考虑异步处理或缓存")

            if success_rate < 0.9:  # 成功率低于90%
                suggestions.append(f"{node_name}: 需要改进错误处理")

        return suggestions
```

### Q32: 如何实现复杂的多模态处理？

**A:** 多模态处理模式：

```python
from typing import Union, List
import base64

class MultiModalState(TypedDict):
    text_input: Optional[str]
    image_input: Optional[str]  # base64 encoded
    audio_input: Optional[str]  # file path or base64
    document_input: Optional[str]
    processing_results: Dict[str, Any]

def detect_input_type(state: MultiModalState) -> str:
    """检测输入类型"""
    if state.get("image_input"):
        return "image_processing"
    elif state.get("audio_input"):
        return "audio_processing"
    elif state.get("document_input"):
        return "document_processing"
    else:
        return "text_processing"

def image_processing_node(state: MultiModalState) -> MultiModalState:
    """图像处理节点"""
    image_data = state["image_input"]

    # 这里集成视觉模型 (如 GPT-4V, Claude Vision)
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4-vision-preview")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }
    ]

    response = model.invoke(messages)

    return {
        **state,
        "processing_results": {
            "image_description": response.content,
            "processed_type": "image"
        }
    }

def audio_processing_node(state: MultiModalState) -> MultiModalState:
    """音频处理节点"""
    # 集成语音识别和处理
    # 这里可以使用 OpenAI Whisper 或其他语音处理服务

    audio_path = state["audio_input"]

    # 模拟语音转文字
    transcribed_text = "这是转录的文字内容"

    return {
        **state,
        "processing_results": {
            "transcribed_text": transcribed_text,
            "processed_type": "audio"
        }
    }

# 构建多模态图
def create_multimodal_graph():
    graph = StateGraph(MultiModalState)

    graph.add_node("detect_type", lambda s: {"input_type": detect_input_type(s)})
    graph.add_node("image_process", image_processing_node)
    graph.add_node("audio_process", audio_processing_node)
    graph.add_node("text_process", text_processing_node)

    graph.add_edge(START, "detect_type")

    graph.add_conditional_edges(
        "detect_type",
        lambda s: s["input_type"],
        {
            "image_processing": "image_process",
            "audio_processing": "audio_process",
            "text_processing": "text_process",
            "document_processing": "document_process"
        }
    )

    return graph.compile()
```

---

## 🔗 相关资源

- 📖 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- 🐛 [错误码对照表](./错误码对照.md)
- 🚀 [API 速查表](./API速查.md)
- 💡 [代码片段库](./代码片段.md)
- 💻 [项目示例](../08-项目案例/)
- 🛠️ [开发实践指南](../06-开发实践/)
- 🏢 [部署运维指南](../07-部署与运维/)

---

*这个 FAQ 会持续更新，如果你有其他问题，欢迎补充！* 💬