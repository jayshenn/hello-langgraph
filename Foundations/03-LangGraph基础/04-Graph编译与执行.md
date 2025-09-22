# Graph 编译与执行：让图运行起来

> 🎯 **学习目标**：掌握图的编译过程和执行方式，理解不同执行模式的适用场景

## ⚙️ compile()：图的编译过程

### 编译的作用

在 LangGraph 中，`compile()` 方法将图的定义转换为可执行的应用：

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class ChatState(TypedDict):
    user_input: str
    response: str

def process_node(state: ChatState) -> ChatState:
    return {**state, "response": f"处理结果: {state['user_input']}"}

# 1. 定义图结构
graph = StateGraph(ChatState)
graph.add_node("process", process_node)
graph.add_edge("START", "process")
graph.add_edge("process", "END")

# 2. 编译成可执行应用
app = graph.compile()

# 3. 现在可以运行了
result = app.invoke({"user_input": "Hello World"})
print(result)  # {"user_input": "Hello World", "response": "处理结果: Hello World"}
```

### 编译时验证

```python
# 编译过程中会进行多项验证
try:
    app = graph.compile()
except ValueError as e:
    print(f"图结构错误: {e}")

# 常见的编译错误：
# 1. 缺少入口点
# 2. 存在孤立节点（无法到达的节点）
# 3. 缺少结束路径
# 4. 循环路径没有退出条件
```

## 🔧 编译配置选项

### Checkpointer：状态持久化

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 内存检查点（适用于开发和测试）
memory_checkpointer = MemorySaver()
app_with_memory = graph.compile(checkpointer=memory_checkpointer)

# 2. SQLite 检查点（适用于生产环境）
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app_with_persistence = graph.compile(checkpointer=sqlite_checkpointer)

# 使用检查点的好处：
# - 状态持久化
# - 支持时间旅行
# - 人机协作
# - 错误恢复
```

### Interrupt 配置

```python
# 配置中断点
app = graph.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=["human_review"],  # 在指定节点前中断
    interrupt_after=["risky_operation"]  # 在指定节点后中断
)

# 动态设置中断点
app = graph.compile(checkpointer=memory_checkpointer)
config = {
    "configurable": {
        "thread_id": "conversation_1"
    },
    "interrupt_before": ["sensitive_operation"]
}

result = app.invoke(initial_state, config=config)
```

### 调试配置

```python
# 启用调试模式
app = graph.compile(
    debug=True,  # 启用调试信息
    checkpointer=memory_checkpointer
)

# 自定义编译器配置
from langgraph.pregel import Pregel

custom_app = graph.compile(
    checkpointer=memory_checkpointer,
    compiler=Pregel,  # 自定义编译器
    debug=True
)
```

## 🏃 执行模式

### 1. invoke()：同步执行

```python
# 基本同步执行
result = app.invoke({"user_input": "Hello"})
print(result)

# 带配置的同步执行
config = {
    "configurable": {
        "thread_id": "user_123",
        "model_name": "gpt-4"
    }
}

result = app.invoke(
    {"user_input": "Hello"},
    config=config
)
```

### 2. ainvoke()：异步执行

```python
import asyncio

async def run_async():
    # 异步执行
    result = await app.ainvoke({"user_input": "Hello"})
    print(result)

    # 并发执行多个请求
    tasks = [
        app.ainvoke({"user_input": f"Query {i}"})
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)
    return results

# 运行异步代码
asyncio.run(run_async())
```

### 3. stream()：流式执行

```python
# 同步流式执行
for chunk in app.stream({"user_input": "Hello"}):
    print(f"收到更新: {chunk}")

# 异步流式执行
async def stream_async():
    async for chunk in app.astream({"user_input": "Hello"}):
        print(f"异步更新: {chunk}")

asyncio.run(stream_async())
```

## 📊 流式执行详解

### 流式模式

```python
# 1. 流式节点更新（默认）
for chunk in app.stream({"user_input": "Hello"}):
    print(chunk)
    # 输出格式: {"node_name": {"updated_state": ...}}

# 2. 流式值更新
for chunk in app.stream({"user_input": "Hello"}, stream_mode="values"):
    print(chunk)
    # 输出格式: {"user_input": "Hello", "response": "..."}

# 3. 流式调试信息
for chunk in app.stream({"user_input": "Hello"}, stream_mode="debug"):
    print(chunk)
    # 输出格式: {"type": "task", "timestamp": ..., "step": ...}
```

### 多种流式模式组合

```python
# 同时流式多种类型的数据
for chunk in app.stream(
    {"user_input": "Hello"},
    stream_mode=["values", "updates", "debug"]
):
    if "values" in chunk:
        print(f"状态更新: {chunk['values']}")
    if "updates" in chunk:
        print(f"节点更新: {chunk['updates']}")
    if "debug" in chunk:
        print(f"调试信息: {chunk['debug']}")
```

### 自定义流式数据

```python
def custom_streaming_node(state: ChatState) -> ChatState:
    """支持自定义流式输出的节点"""
    from langgraph.prebuilt import stream

    # 流式输出处理进度
    stream("开始处理用户输入...")

    processed_input = process_input(state["user_input"])
    stream(f"输入处理完成: {processed_input[:50]}...")

    result = generate_response(processed_input)
    stream("响应生成完成!")

    return {**state, "response": result}

# 接收自定义流式数据
for chunk in app.stream({"user_input": "Hello"}, stream_mode="custom"):
    if isinstance(chunk, str):
        print(f"进度更新: {chunk}")
    else:
        print(f"状态更新: {chunk}")
```

## 🎮 执行配置

### 基本配置选项

```python
config = {
    # 线程配置
    "configurable": {
        "thread_id": "conversation_123",  # 对话线程ID
        "user_id": "user_456",           # 用户ID
    },

    # 执行配置
    "recursion_limit": 100,              # 递归限制
    "max_execution_time": 300,           # 最大执行时间（秒）

    # 中断配置
    "interrupt_before": ["human_review"],
    "interrupt_after": ["sensitive_operation"],

    # 调试配置
    "debug": True,
    "verbose": True
}

result = app.invoke(initial_state, config=config)
```

### 动态配置

```python
def get_dynamic_config(user_id: str, session_id: str) -> dict:
    """根据用户和会话动态生成配置"""
    user_profile = get_user_profile(user_id)

    config = {
        "configurable": {
            "thread_id": f"{user_id}_{session_id}",
            "model_name": user_profile.get("preferred_model", "gpt-3.5-turbo"),
            "temperature": user_profile.get("creativity_level", 0.7),
            "max_tokens": user_profile.get("response_length", 150)
        }
    }

    # 根据用户等级设置不同的限制
    if user_profile.get("is_premium"):
        config["recursion_limit"] = 200
        config["max_execution_time"] = 600
    else:
        config["recursion_limit"] = 50
        config["max_execution_time"] = 120

    return config

# 使用动态配置
config = get_dynamic_config("user_123", "session_456")
result = app.invoke({"user_input": "Hello"}, config=config)
```

## 🔄 检查点和状态管理

### 基本检查点操作

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "conversation_1"}}

# 第一次执行
result1 = app.invoke({"user_input": "Hello"}, config=config)

# 第二次执行（在同一线程中，会继续之前的状态）
result2 = app.invoke({"user_input": "How are you?"}, config=config)

# 查看检查点历史
checkpoints = list(checkpointer.list(config))
for checkpoint in checkpoints:
    print(f"检查点 {checkpoint.checkpoint_id}: {checkpoint.state}")
```

### 高级检查点操作

```python
# 从特定检查点恢复
specific_config = {
    "configurable": {
        "thread_id": "conversation_1",
        "checkpoint_id": "specific_checkpoint_id"
    }
}

result = app.invoke({"user_input": "Continue"}, config=specific_config)

# 获取当前状态
current_state = app.get_state(config)
print(f"当前状态: {current_state.values}")
print(f"下一个节点: {current_state.next}")

# 更新状态
new_state = {**current_state.values, "custom_field": "new_value"}
app.update_state(config, new_state)
```

## 🛠️ 调试和监控

### 内置调试工具

```python
# 启用详细调试
app = graph.compile(checkpointer=checkpointer, debug=True)

# 调试执行过程
import logging
logging.basicConfig(level=logging.DEBUG)

for chunk in app.stream({"user_input": "Hello"}, stream_mode="debug"):
    print(f"调试信息: {chunk}")
```

### 自定义监控

```python
class ExecutionMonitor:
    """自定义执行监控器"""

    def __init__(self):
        self.execution_log = []
        self.start_time = None

    def log_node_start(self, node_name: str, state: dict):
        """记录节点开始执行"""
        self.execution_log.append({
            "event": "node_start",
            "node": node_name,
            "timestamp": time.time(),
            "state_keys": list(state.keys())
        })

    def log_node_end(self, node_name: str, result: dict):
        """记录节点执行完成"""
        self.execution_log.append({
            "event": "node_end",
            "node": node_name,
            "timestamp": time.time(),
            "result_keys": list(result.keys())
        })

    def get_report(self):
        """生成执行报告"""
        total_time = time.time() - self.start_time
        node_count = len([log for log in self.execution_log if log["event"] == "node_start"])

        return {
            "total_execution_time": total_time,
            "nodes_executed": node_count,
            "execution_log": self.execution_log
        }

# 使用监控器
monitor = ExecutionMonitor()

def monitored_node(state: ChatState) -> ChatState:
    """带监控的节点"""
    monitor.log_node_start("monitored_node", state)

    # 实际处理逻辑
    result = {**state, "processed": True}

    monitor.log_node_end("monitored_node", result)
    return result
```

### 性能分析

```python
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(state):
        start_time = time.time()
        result = func(state)
        end_time = time.time()

        # 记录执行时间
        execution_time = end_time - start_time
        print(f"节点 {func.__name__} 执行时间: {execution_time:.3f}秒")

        # 添加性能信息到状态
        performance_info = result.get("_performance", {})
        performance_info[func.__name__] = execution_time

        return {**result, "_performance": performance_info}

    return wrapper

@performance_monitor
def slow_processing_node(state: ChatState) -> ChatState:
    """耗时的处理节点"""
    time.sleep(2)  # 模拟耗时操作
    return {**state, "processed": True}
```

## 🚨 错误处理和恢复

### 异常处理

```python
def robust_execution(app, initial_state, config):
    """健壮的执行函数"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            return app.invoke(initial_state, config)

        except Exception as e:
            retry_count += 1
            print(f"执行失败 (尝试 {retry_count}/{max_retries}): {e}")

            if retry_count >= max_retries:
                print("达到最大重试次数，执行失败")
                raise

            # 指数退避
            wait_time = 2 ** retry_count
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

# 使用健壮执行
try:
    result = robust_execution(app, {"user_input": "Hello"}, config)
except Exception as e:
    print(f"最终执行失败: {e}")
```

### 状态回滚

```python
def safe_execution_with_rollback(app, initial_state, config):
    """支持回滚的安全执行"""
    # 保存初始检查点
    initial_checkpoint = app.get_state(config)

    try:
        result = app.invoke(initial_state, config)
        return result

    except Exception as e:
        print(f"执行出错，回滚到初始状态: {e}")

        # 回滚到初始状态
        if initial_checkpoint:
            app.update_state(config, initial_checkpoint.values)

        raise
```

## 🎪 实战案例：完整的执行流程

```python
import asyncio
import time
from typing import AsyncGenerator

class AdvancedChatApp:
    """高级聊天应用示例"""

    def __init__(self, graph: StateGraph):
        self.checkpointer = MemorySaver()
        self.app = graph.compile(
            checkpointer=self.checkpointer,
            debug=True
        )
        self.execution_stats = {}

    async def chat(self, user_id: str, message: str) -> AsyncGenerator[dict, None]:
        """异步聊天接口"""
        config = {
            "configurable": {
                "thread_id": f"user_{user_id}",
                "user_id": user_id
            }
        }

        start_time = time.time()
        token_count = 0

        try:
            # 流式执行
            async for chunk in self.app.astream(
                {"user_input": message},
                config=config,
                stream_mode=["values", "updates"]
            ):
                # 计算token使用量
                if "values" in chunk and "tokens_used" in chunk["values"]:
                    token_count += chunk["values"]["tokens_used"]

                yield {
                    "type": "update",
                    "data": chunk,
                    "timestamp": time.time()
                }

            # 记录统计信息
            execution_time = time.time() - start_time
            self.execution_stats[user_id] = {
                "last_execution_time": execution_time,
                "total_tokens": token_count,
                "timestamp": time.time()
            }

            yield {
                "type": "complete",
                "execution_time": execution_time,
                "tokens_used": token_count
            }

        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_conversation_history(self, user_id: str) -> list:
        """获取对话历史"""
        config = {"configurable": {"thread_id": f"user_{user_id}"}}

        try:
            checkpoints = list(self.checkpointer.list(config))
            history = []

            for checkpoint in checkpoints:
                if "conversation_history" in checkpoint.state:
                    history.extend(checkpoint.state["conversation_history"])

            return history
        except Exception:
            return []

    def reset_conversation(self, user_id: str):
        """重置对话"""
        config = {"configurable": {"thread_id": f"user_{user_id}"}}

        # 清除检查点历史
        current_state = self.app.get_state(config)
        if current_state:
            # 重置为初始状态
            self.app.update_state(config, {
                "conversation_history": [],
                "user_context": {}
            })

# 使用示例
async def main():
    # 假设已经有定义好的图
    chat_app = AdvancedChatApp(chat_graph)

    # 异步聊天
    async for update in chat_app.chat("user_123", "Hello, how are you?"):
        if update["type"] == "update":
            print(f"更新: {update['data']}")
        elif update["type"] == "complete":
            print(f"完成 - 耗时: {update['execution_time']:.2f}秒")
        elif update["type"] == "error":
            print(f"错误: {update['error']}")

    # 查看历史
    history = chat_app.get_conversation_history("user_123")
    print(f"对话历史: {history}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 💡 最佳实践

### ✅ 编译最佳实践

1. **合理的检查点策略**：根据应用需求选择合适的检查点器
2. **适当的中断点**：在关键节点设置中断，支持人机协作
3. **错误处理**：为关键节点添加错误处理逻辑
4. **性能监控**：监控执行时间和资源使用
5. **状态验证**：在关键步骤验证状态的完整性

### ✅ 执行最佳实践

1. **选择合适的执行模式**：根据场景选择同步、异步或流式执行
2. **配置管理**：使用动态配置适应不同用户和场景
3. **资源管理**：合理设置执行限制和超时
4. **监控和日志**：记录执行过程，便于调试和优化
5. **优雅降级**：在异常情况下提供合理的降级方案

### ❌ 常见陷阱

1. **无限递归**：缺少退出条件的循环
2. **内存泄漏**：长时间运行时检查点数据积累
3. **阻塞操作**：在异步环境中使用同步阻塞操作
4. **配置错误**：不正确的线程ID或检查点配置
5. **异常吞没**：忽略或不当处理异常

## 🚀 下一步

掌握了图的编译和执行后，你可以：
- 查看 `示例代码/simple_graph.py` 了解完整实现
- 学习 `04-进阶特性` 中的高级功能
- 实践构建自己的 LangGraph 应用

---

*理解编译和执行机制，让你的 LangGraph 应用稳定高效地运行！* ⚡