# 08-Webhooks集成

## 概述

Webhooks 是事件驱动架构的核心组件，允许 LangGraph 应用与外部系统实时通信。本文档涵盖 Webhooks 的设计、实现、安全和监控，实现系统间的松耦合集成。

## Webhooks 架构

### 事件驱动架构图
```
┌─────────────┐    事件     ┌─────────────┐    HTTP POST    ┌─────────────┐
│ LangGraph   │────────────▶│  事件队列   │───────────────▶│  外部系统   │
│    应用     │             │  (Redis)    │                │  (Webhook)  │
└─────────────┘             └─────────────┘                └─────────────┘
       │                           │                              │
       │                           ▼                              │
       │                  ┌─────────────┐                         │
       │                  │ Webhook     │                         │
       │                  │ 处理器      │◀────────────────────────┘
       │                  └─────────────┘      确认/重试
       │                           │
       └───────────────────────────┼─────────── 状态更新
                                   ▼
                          ┌─────────────┐
                          │  监控系统   │
                          │ (Prometheus)│
                          └─────────────┘
```

### 核心概念
- **事件触发**: Agent 执行状态变化时触发事件
- **异步处理**: 使用队列确保可靠传递
- **重试机制**: 处理网络错误和临时故障
- **签名验证**: 确保 Webhook 安全性
- **监控追踪**: 记录传递状态和性能指标

## Webhooks 实现

### 基础 Webhook 系统
```python
# webhook_system.py
import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import httpx
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class WebhookStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class WebhookEvent:
    """Webhook事件数据类"""
    id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    source: str
    version: str = "1.0"

class WebhookEndpoint(Base):
    """Webhook端点模型"""
    __tablename__ = 'webhook_endpoints'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    url = Column(String(500), nullable=False)
    secret = Column(String(64))  # 用于签名验证
    is_active = Column(Boolean, default=True)
    event_types = Column(Text)  # JSON格式存储事件类型列表
    headers = Column(Text)  # JSON格式存储自定义头部
    timeout = Column(Integer, default=30)
    max_retries = Column(Integer, default=3)
    retry_delay = Column(Integer, default=60)  # 秒
    created_at = Column(DateTime, default=datetime.utcnow)

class WebhookDelivery(Base):
    """Webhook传递记录"""
    __tablename__ = 'webhook_deliveries'

    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, nullable=False)
    event_id = Column(String(36), nullable=False)
    event_type = Column(String(50), nullable=False)
    status = Column(String(20), default=WebhookStatus.PENDING.value)
    attempt_count = Column(Integer, default=0)
    last_attempt_at = Column(DateTime)
    response_status = Column(Integer)
    response_body = Column(Text)
    response_time = Column(Float)  # 响应时间（秒）
    error_message = Column(Text)
    next_retry_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class WebhookManager:
    """Webhook管理器"""

    def __init__(self, redis_client, db_session):
        self.redis = redis_client
        self.db = db_session
        self.event_queue = "webhook_events"
        self.retry_queue = "webhook_retries"

    async def register_endpoint(
        self,
        name: str,
        url: str,
        event_types: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ) -> WebhookEndpoint:
        """注册Webhook端点"""
        endpoint = WebhookEndpoint(
            name=name,
            url=url,
            secret=secret or self._generate_secret(),
            event_types=json.dumps(event_types),
            headers=json.dumps(headers or {}),
            timeout=timeout,
            max_retries=max_retries
        )

        self.db.add(endpoint)
        self.db.commit()
        self.db.refresh(endpoint)

        return endpoint

    def _generate_secret(self) -> str:
        """生成Webhook密钥"""
        import secrets
        return secrets.token_hex(32)

    async def trigger_event(self, event: WebhookEvent):
        """触发Webhook事件"""
        # 将事件加入队列
        event_data = asdict(event)
        await self.redis.lpush(self.event_queue, json.dumps(event_data))

        logger.info(f"事件已加入队列: {event.event_type} - {event.id}")

    async def process_events(self):
        """处理Webhook事件"""
        while True:
            try:
                # 从队列获取事件
                event_data = await self.redis.brpop(self.event_queue, timeout=1)
                if not event_data:
                    continue

                event_json = event_data[1]
                event_dict = json.loads(event_json)
                event = WebhookEvent(**event_dict)

                # 获取匹配的端点
                endpoints = self._get_matching_endpoints(event.event_type)

                # 为每个端点创建传递任务
                for endpoint in endpoints:
                    await self._schedule_delivery(endpoint, event)

            except Exception as e:
                logger.error(f"处理Webhook事件时发生错误: {e}")
                await asyncio.sleep(1)

    def _get_matching_endpoints(self, event_type: str) -> List[WebhookEndpoint]:
        """获取匹配的端点"""
        endpoints = (
            self.db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.is_active == True)
            .all()
        )

        matching_endpoints = []
        for endpoint in endpoints:
            event_types = json.loads(endpoint.event_types)
            if event_type in event_types or "*" in event_types:
                matching_endpoints.append(endpoint)

        return matching_endpoints

    async def _schedule_delivery(self, endpoint: WebhookEndpoint, event: WebhookEvent):
        """安排事件传递"""
        delivery = WebhookDelivery(
            endpoint_id=endpoint.id,
            event_id=event.id,
            event_type=event.event_type,
            status=WebhookStatus.PENDING.value
        )

        self.db.add(delivery)
        self.db.commit()

        # 立即尝试传递
        await self._attempt_delivery(endpoint, event, delivery)

    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        delivery: WebhookDelivery
    ):
        """尝试传递Webhook"""
        delivery.attempt_count += 1
        delivery.last_attempt_at = datetime.utcnow()

        try:
            # 准备请求数据
            payload = {
                "id": event.id,
                "event_type": event.event_type,
                "data": event.data,
                "timestamp": event.timestamp,
                "source": event.source,
                "version": event.version
            }

            # 生成签名
            signature = self._generate_signature(
                json.dumps(payload, separators=(',', ':')),
                endpoint.secret
            )

            # 准备头部
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": event.event_type,
                "X-Webhook-Delivery": str(delivery.id),
                "User-Agent": "LangGraph-Webhook/1.0"
            }

            # 添加自定义头部
            if endpoint.headers:
                custom_headers = json.loads(endpoint.headers)
                headers.update(custom_headers)

            # 发送请求
            start_time = time.time()
            async with httpx.AsyncClient(timeout=endpoint.timeout) as client:
                response = await client.post(
                    endpoint.url,
                    json=payload,
                    headers=headers
                )

            response_time = time.time() - start_time

            # 记录响应
            delivery.response_status = response.status_code
            delivery.response_body = response.text[:1000]  # 限制长度
            delivery.response_time = response_time

            # 检查响应状态
            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED.value
                logger.info(
                    f"Webhook传递成功: {endpoint.name} - {event.event_type}"
                )
            else:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=None,
                    response=response
                )

        except Exception as e:
            delivery.error_message = str(e)
            logger.error(
                f"Webhook传递失败: {endpoint.name} - {event.event_type} - {e}"
            )

            # 安排重试
            if delivery.attempt_count < endpoint.max_retries:
                await self._schedule_retry(endpoint, event, delivery)
            else:
                delivery.status = WebhookStatus.FAILED.value
                logger.error(
                    f"Webhook传递最终失败: {endpoint.name} - {event.event_type}"
                )

        finally:
            self.db.commit()

    def _generate_signature(self, payload: str, secret: str) -> str:
        """生成HMAC签名"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    async def _schedule_retry(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        delivery: WebhookDelivery
    ):
        """安排重试"""
        delivery.status = WebhookStatus.RETRYING.value

        # 计算下次重试时间（指数退避）
        delay = endpoint.retry_delay * (2 ** (delivery.attempt_count - 1))
        next_retry = datetime.utcnow() + timedelta(seconds=delay)
        delivery.next_retry_at = next_retry

        # 加入重试队列
        retry_data = {
            "endpoint_id": endpoint.id,
            "event": asdict(event),
            "delivery_id": delivery.id,
            "retry_at": next_retry.timestamp()
        }

        await self.redis.zadd(
            self.retry_queue,
            {json.dumps(retry_data): next_retry.timestamp()}
        )

        logger.info(
            f"安排重试: {endpoint.name} - {event.event_type} - "
            f"第{delivery.attempt_count}次尝试，延迟{delay}秒"
        )

    async def process_retries(self):
        """处理重试队列"""
        while True:
            try:
                current_time = time.time()

                # 获取应该重试的事件
                items = await self.redis.zrangebyscore(
                    self.retry_queue,
                    0,
                    current_time,
                    withscores=True
                )

                for item_data, score in items:
                    retry_data = json.loads(item_data)

                    # 从重试队列中移除
                    await self.redis.zrem(self.retry_queue, item_data)

                    # 重新尝试传递
                    endpoint = (
                        self.db.query(WebhookEndpoint)
                        .filter(WebhookEndpoint.id == retry_data["endpoint_id"])
                        .first()
                    )

                    if endpoint and endpoint.is_active:
                        event = WebhookEvent(**retry_data["event"])
                        delivery = (
                            self.db.query(WebhookDelivery)
                            .filter(WebhookDelivery.id == retry_data["delivery_id"])
                            .first()
                        )

                        if delivery:
                            await self._attempt_delivery(endpoint, event, delivery)

                await asyncio.sleep(10)  # 每10秒检查一次

            except Exception as e:
                logger.error(f"处理重试队列时发生错误: {e}")
                await asyncio.sleep(10)

# 全局Webhook管理器实例
webhook_manager = WebhookManager(redis_client, db_session)
```

### LangGraph 集成
```python
# langgraph_webhook_integration.py
from langgraph.graph import StateGraph
from langgraph.checkpoint import MemorySaver
import uuid

class WebhookEnabledAgent:
    """支持Webhook的Agent"""

    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager
        self.graph = self._create_graph()

    def _create_graph(self):
        """创建支持Webhook的图"""
        graph = StateGraph(AgentState)

        # 添加节点（带Webhook触发）
        graph.add_node("start", self._webhook_node("agent_started", self.start_node))
        graph.add_node("process", self._webhook_node("agent_processing", self.process_node))
        graph.add_node("end", self._webhook_node("agent_completed", self.end_node))

        # 设置流程
        graph.set_entry_point("start")
        graph.add_edge("start", "process")
        graph.add_edge("process", "end")
        graph.set_finish_point("end")

        return graph.compile(checkpointer=MemorySaver())

    def _webhook_node(self, event_type: str, node_func: Callable):
        """装饰节点以触发Webhook"""
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # 执行原始节点
            start_time = time.time()
            result = await node_func(state) if asyncio.iscoroutinefunction(node_func) else node_func(state)
            execution_time = time.time() - start_time

            # 触发Webhook事件
            event = WebhookEvent(
                id=str(uuid.uuid4()),
                event_type=event_type,
                data={
                    "agent_id": state.get("agent_id"),
                    "user_id": state.get("user_id"),
                    "input": state.get("input"),
                    "output": result,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                },
                timestamp=time.time(),
                source="langgraph_agent",
                version="1.0"
            )

            await self.webhook_manager.trigger_event(event)

            return result

        return wrapper

    def start_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """开始节点"""
        return {"status": "started", "message": "Agent execution started"}

    def process_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理节点"""
        user_input = state.get("input", "")
        result = f"处理结果: {user_input}"
        return {"result": result, "status": "processed"}

    def end_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """结束节点"""
        return {"status": "completed", "final_result": state.get("result")}

    async def invoke_with_webhooks(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """调用Agent并触发Webhook"""
        # 添加执行ID
        input_data["agent_id"] = str(uuid.uuid4())

        # 触发开始事件
        start_event = WebhookEvent(
            id=str(uuid.uuid4()),
            event_type="agent_invocation_started",
            data={
                "agent_id": input_data["agent_id"],
                "input": input_data,
                "timestamp": time.time()
            },
            timestamp=time.time(),
            source="langgraph_agent"
        )
        await self.webhook_manager.trigger_event(start_event)

        try:
            # 执行Agent
            result = self.graph.invoke(input_data)

            # 触发成功事件
            success_event = WebhookEvent(
                id=str(uuid.uuid4()),
                event_type="agent_invocation_success",
                data={
                    "agent_id": input_data["agent_id"],
                    "result": result,
                    "timestamp": time.time()
                },
                timestamp=time.time(),
                source="langgraph_agent"
            )
            await self.webhook_manager.trigger_event(success_event)

            return result

        except Exception as e:
            # 触发错误事件
            error_event = WebhookEvent(
                id=str(uuid.uuid4()),
                event_type="agent_invocation_error",
                data={
                    "agent_id": input_data["agent_id"],
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time()
                },
                timestamp=time.time(),
                source="langgraph_agent"
            )
            await self.webhook_manager.trigger_event(error_event)

            raise

# 使用示例
agent = WebhookEnabledAgent(webhook_manager)

# 注册Webhook端点
await webhook_manager.register_endpoint(
    name="external_system",
    url="https://external-system.com/webhooks/langgraph",
    event_types=["agent_started", "agent_completed", "agent_invocation_error"],
    headers={"Authorization": "Bearer your-api-key"}
)

# 启动事件处理器
asyncio.create_task(webhook_manager.process_events())
asyncio.create_task(webhook_manager.process_retries())
```

## Webhook 安全

### 签名验证
```python
# webhook_security.py
import hmac
import hashlib
import time
from typing import Optional

class WebhookSecurity:
    """Webhook安全管理"""

    @staticmethod
    def verify_signature(
        payload: str,
        signature: str,
        secret: str,
        tolerance: int = 300  # 5分钟容忍时间
    ) -> bool:
        """验证Webhook签名"""
        try:
            # 解析签名
            if not signature.startswith("sha256="):
                return False

            expected_signature = signature[7:]  # 移除 "sha256=" 前缀

            # 计算期望的签名
            computed_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # 使用安全比较
            return hmac.compare_digest(expected_signature, computed_signature)

        except Exception:
            return False

    @staticmethod
    def verify_timestamp(
        timestamp: float,
        tolerance: int = 300
    ) -> bool:
        """验证时间戳（防重放攻击）"""
        current_time = time.time()
        return abs(current_time - timestamp) <= tolerance

    @staticmethod
    def rate_limit_check(
        redis_client,
        endpoint_id: str,
        max_requests: int = 100,
        window: int = 3600
    ) -> bool:
        """速率限制检查"""
        key = f"webhook_rate_limit:{endpoint_id}"
        current_requests = redis_client.get(key)

        if current_requests is None:
            redis_client.setex(key, window, 1)
            return True
        else:
            current_requests = int(current_requests)
            if current_requests >= max_requests:
                return False
            else:
                redis_client.incr(key)
                return True

# Webhook接收端验证中间件
class WebhookVerificationMiddleware:
    """Webhook验证中间件"""

    def __init__(self, secret: str):
        self.secret = secret

    async def __call__(self, request: Request, call_next):
        """验证传入的Webhook"""
        if request.url.path.startswith("/webhooks/"):
            # 获取签名
            signature = request.headers.get("X-Webhook-Signature")
            if not signature:
                return PlainTextResponse(
                    "Missing signature",
                    status_code=401
                )

            # 读取请求体
            body = await request.body()
            payload = body.decode('utf-8')

            # 验证签名
            if not WebhookSecurity.verify_signature(payload, signature, self.secret):
                return PlainTextResponse(
                    "Invalid signature",
                    status_code=401
                )

            # 验证时间戳（如果提供）
            timestamp_header = request.headers.get("X-Webhook-Timestamp")
            if timestamp_header:
                try:
                    timestamp = float(timestamp_header)
                    if not WebhookSecurity.verify_timestamp(timestamp):
                        return PlainTextResponse(
                            "Request too old",
                            status_code=401
                        )
                except ValueError:
                    return PlainTextResponse(
                        "Invalid timestamp",
                        status_code=400
                    )

        response = await call_next(request)
        return response
```

## Webhook 接收端

### 接收端实现
```python
# webhook_receiver.py
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
import json

app = FastAPI()

class WebhookReceiver:
    """Webhook接收器"""

    def __init__(self):
        self.handlers = {}

    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.handlers[event_type] = handler

    async def process_webhook(self, request: Request, background_tasks: BackgroundTasks):
        """处理Webhook请求"""
        try:
            # 解析请求
            payload = await request.json()
            event_type = payload.get("event_type")
            event_id = payload.get("id")

            logger.info(f"收到Webhook事件: {event_type} - {event_id}")

            # 查找处理器
            handler = self.handlers.get(event_type)
            if not handler:
                logger.warning(f"未找到事件处理器: {event_type}")
                return {"status": "ignored", "message": f"No handler for {event_type}"}

            # 异步处理事件
            background_tasks.add_task(handler, payload)

            return {"status": "accepted", "event_id": event_id}

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        except Exception as e:
            logger.error(f"处理Webhook时发生错误: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

# 创建接收器实例
webhook_receiver = WebhookReceiver()

# 注册事件处理器
async def handle_agent_started(payload: Dict[str, Any]):
    """处理Agent开始事件"""
    data = payload["data"]
    agent_id = data["agent_id"]
    user_id = data.get("user_id")

    logger.info(f"Agent {agent_id} 开始执行，用户: {user_id}")

    # 更新外部系统状态
    await update_external_system_status(agent_id, "running")

async def handle_agent_completed(payload: Dict[str, Any]):
    """处理Agent完成事件"""
    data = payload["data"]
    agent_id = data["agent_id"]
    result = data["output"]

    logger.info(f"Agent {agent_id} 执行完成")

    # 发送结果到外部系统
    await send_result_to_external_system(agent_id, result)

async def handle_agent_error(payload: Dict[str, Any]):
    """处理Agent错误事件"""
    data = payload["data"]
    agent_id = data["agent_id"]
    error = data["error"]

    logger.error(f"Agent {agent_id} 执行失败: {error}")

    # 发送错误通知
    await send_error_notification(agent_id, error)

# 注册处理器
webhook_receiver.register_handler("agent_started", handle_agent_started)
webhook_receiver.register_handler("agent_completed", handle_agent_completed)
webhook_receiver.register_handler("agent_invocation_error", handle_agent_error)

# Webhook端点
@app.post("/webhooks/langgraph")
async def receive_langgraph_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """接收LangGraph Webhook"""
    return await webhook_receiver.process_webhook(request, background_tasks)

# 健康检查端点
@app.get("/webhooks/health")
async def webhook_health():
    """Webhook健康检查"""
    return {"status": "healthy", "timestamp": time.time()}

# 辅助函数
async def update_external_system_status(agent_id: str, status: str):
    """更新外部系统状态"""
    # 实现与外部系统的集成逻辑
    pass

async def send_result_to_external_system(agent_id: str, result: Any):
    """发送结果到外部系统"""
    # 实现结果传递逻辑
    pass

async def send_error_notification(agent_id: str, error: str):
    """发送错误通知"""
    # 实现错误通知逻辑
    pass
```

## Webhook 监控

### 监控指标
```python
# webhook_monitoring.py
from prometheus_client import Counter, Histogram, Gauge

# Webhook指标
WEBHOOK_EVENTS_TOTAL = Counter(
    'webhook_events_total',
    'Total webhook events',
    ['event_type', 'status']
)

WEBHOOK_DELIVERY_DURATION = Histogram(
    'webhook_delivery_duration_seconds',
    'Webhook delivery duration',
    ['endpoint_name', 'event_type']
)

WEBHOOK_DELIVERY_ATTEMPTS = Counter(
    'webhook_delivery_attempts_total',
    'Total webhook delivery attempts',
    ['endpoint_name', 'attempt_number']
)

WEBHOOK_QUEUE_SIZE = Gauge(
    'webhook_queue_size',
    'Number of events in webhook queue'
)

WEBHOOK_ENDPOINTS_ACTIVE = Gauge(
    'webhook_endpoints_active',
    'Number of active webhook endpoints'
)

class WebhookMonitor:
    """Webhook监控"""

    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager

    async def record_event(self, event_type: str, status: str):
        """记录事件指标"""
        WEBHOOK_EVENTS_TOTAL.labels(
            event_type=event_type,
            status=status
        ).inc()

    async def record_delivery(
        self,
        endpoint_name: str,
        event_type: str,
        duration: float,
        attempt_number: int
    ):
        """记录传递指标"""
        WEBHOOK_DELIVERY_DURATION.labels(
            endpoint_name=endpoint_name,
            event_type=event_type
        ).observe(duration)

        WEBHOOK_DELIVERY_ATTEMPTS.labels(
            endpoint_name=endpoint_name,
            attempt_number=str(attempt_number)
        ).inc()

    async def update_queue_metrics(self):
        """更新队列指标"""
        # 获取队列大小
        queue_size = await self.webhook_manager.redis.llen(
            self.webhook_manager.event_queue
        )
        WEBHOOK_QUEUE_SIZE.set(queue_size)

        # 获取活跃端点数量
        active_endpoints = (
            self.webhook_manager.db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.is_active == True)
            .count()
        )
        WEBHOOK_ENDPOINTS_ACTIVE.set(active_endpoints)

    async def monitor_loop(self):
        """监控循环"""
        while True:
            try:
                await self.update_queue_metrics()
                await asyncio.sleep(30)  # 每30秒更新一次
            except Exception as e:
                logger.error(f"监控更新失败: {e}")
                await asyncio.sleep(30)

# 集成监控到Webhook管理器
class MonitoredWebhookManager(WebhookManager):
    """带监控的Webhook管理器"""

    def __init__(self, redis_client, db_session):
        super().__init__(redis_client, db_session)
        self.monitor = WebhookMonitor(self)

    async def trigger_event(self, event: WebhookEvent):
        """触发事件（带监控）"""
        await super().trigger_event(event)
        await self.monitor.record_event(event.event_type, "triggered")

    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        delivery: WebhookDelivery
    ):
        """尝试传递（带监控）"""
        start_time = time.time()

        await super()._attempt_delivery(endpoint, event, delivery)

        # 记录监控指标
        duration = time.time() - start_time
        await self.monitor.record_delivery(
            endpoint.name,
            event.event_type,
            duration,
            delivery.attempt_count
        )

        # 记录事件状态
        status = "delivered" if delivery.status == WebhookStatus.DELIVERED.value else "failed"
        await self.monitor.record_event(event.event_type, status)
```

### 监控仪表板
```python
# webhook_dashboard.py
from fastapi import FastAPI, Depends
from typing import List, Dict, Any

app = FastAPI()

class WebhookDashboard:
    """Webhook仪表板"""

    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager

    async def get_delivery_stats(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """获取传递统计"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # 总传递数
        total_deliveries = (
            self.webhook_manager.db.query(WebhookDelivery)
            .filter(WebhookDelivery.created_at >= cutoff_time)
            .count()
        )

        # 成功传递数
        successful_deliveries = (
            self.webhook_manager.db.query(WebhookDelivery)
            .filter(WebhookDelivery.created_at >= cutoff_time)
            .filter(WebhookDelivery.status == WebhookStatus.DELIVERED.value)
            .count()
        )

        # 失败传递数
        failed_deliveries = (
            self.webhook_manager.db.query(WebhookDelivery)
            .filter(WebhookDelivery.created_at >= cutoff_time)
            .filter(WebhookDelivery.status == WebhookStatus.FAILED.value)
            .count()
        )

        # 平均响应时间
        avg_response_time = (
            self.webhook_manager.db.query(
                func.avg(WebhookDelivery.response_time)
            )
            .filter(WebhookDelivery.created_at >= cutoff_time)
            .filter(WebhookDelivery.response_time.isnot(None))
            .scalar()
        ) or 0

        success_rate = (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0

        return {
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": failed_deliveries,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "period_hours": hours
        }

    async def get_endpoint_stats(self) -> List[Dict[str, Any]]:
        """获取端点统计"""
        endpoints = (
            self.webhook_manager.db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.is_active == True)
            .all()
        )

        stats = []
        for endpoint in endpoints:
            # 获取最近24小时的传递统计
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            total = (
                self.webhook_manager.db.query(WebhookDelivery)
                .filter(WebhookDelivery.endpoint_id == endpoint.id)
                .filter(WebhookDelivery.created_at >= cutoff_time)
                .count()
            )

            successful = (
                self.webhook_manager.db.query(WebhookDelivery)
                .filter(WebhookDelivery.endpoint_id == endpoint.id)
                .filter(WebhookDelivery.created_at >= cutoff_time)
                .filter(WebhookDelivery.status == WebhookStatus.DELIVERED.value)
                .count()
            )

            stats.append({
                "id": endpoint.id,
                "name": endpoint.name,
                "url": endpoint.url,
                "event_types": json.loads(endpoint.event_types),
                "total_deliveries_24h": total,
                "successful_deliveries_24h": successful,
                "success_rate_24h": round((successful / total * 100) if total > 0 else 0, 2)
            })

        return stats

    async def get_recent_deliveries(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取最近的传递记录"""
        deliveries = (
            self.webhook_manager.db.query(WebhookDelivery)
            .order_by(WebhookDelivery.created_at.desc())
            .limit(limit)
            .all()
        )

        result = []
        for delivery in deliveries:
            endpoint = (
                self.webhook_manager.db.query(WebhookEndpoint)
                .filter(WebhookEndpoint.id == delivery.endpoint_id)
                .first()
            )

            result.append({
                "id": delivery.id,
                "endpoint_name": endpoint.name if endpoint else "Unknown",
                "event_type": delivery.event_type,
                "status": delivery.status,
                "attempt_count": delivery.attempt_count,
                "response_status": delivery.response_status,
                "response_time": delivery.response_time,
                "created_at": delivery.created_at.isoformat(),
                "error_message": delivery.error_message
            })

        return result

# API端点
dashboard = WebhookDashboard(webhook_manager)

@app.get("/api/webhooks/stats")
async def get_webhook_stats(hours: int = 24):
    """获取Webhook统计"""
    return await dashboard.get_delivery_stats(hours)

@app.get("/api/webhooks/endpoints")
async def get_webhook_endpoints():
    """获取Webhook端点统计"""
    return await dashboard.get_endpoint_stats()

@app.get("/api/webhooks/deliveries")
async def get_recent_deliveries(limit: int = 50):
    """获取最近的传递记录"""
    return await dashboard.get_recent_deliveries(limit)
```

## 故障排除

### 常见问题诊断
```python
# webhook_troubleshooting.py

class WebhookTroubleshooter:
    """Webhook故障排除工具"""

    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager

    async def diagnose_endpoint(self, endpoint_id: int) -> Dict[str, Any]:
        """诊断端点问题"""
        endpoint = (
            self.webhook_manager.db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.id == endpoint_id)
            .first()
        )

        if not endpoint:
            return {"error": "端点不存在"}

        # 检查端点配置
        issues = []
        recommendations = []

        # 检查URL可达性
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.head(endpoint.url)
                if response.status_code >= 400:
                    issues.append(f"端点返回错误状态码: {response.status_code}")
        except Exception as e:
            issues.append(f"无法连接到端点: {e}")
            recommendations.append("检查URL是否正确，服务是否运行")

        # 检查最近传递情况
        recent_deliveries = (
            self.webhook_manager.db.query(WebhookDelivery)
            .filter(WebhookDelivery.endpoint_id == endpoint_id)
            .order_by(WebhookDelivery.created_at.desc())
            .limit(10)
            .all()
        )

        failed_count = sum(1 for d in recent_deliveries if d.status == WebhookStatus.FAILED.value)
        if failed_count > 5:
            issues.append(f"最近10次传递中有{failed_count}次失败")
            recommendations.append("检查端点实现和错误日志")

        # 检查响应时间
        avg_response_time = sum(
            d.response_time for d in recent_deliveries
            if d.response_time is not None
        ) / len(recent_deliveries) if recent_deliveries else 0

        if avg_response_time > 5:
            issues.append(f"平均响应时间过长: {avg_response_time:.2f}秒")
            recommendations.append("优化端点处理逻辑")

        return {
            "endpoint_id": endpoint_id,
            "endpoint_name": endpoint.name,
            "is_active": endpoint.is_active,
            "issues": issues,
            "recommendations": recommendations,
            "recent_delivery_count": len(recent_deliveries),
            "failed_delivery_count": failed_count,
            "avg_response_time": round(avg_response_time, 2)
        }

    async def test_endpoint(
        self,
        endpoint_id: int,
        test_event: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """测试端点"""
        endpoint = (
            self.webhook_manager.db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.id == endpoint_id)
            .first()
        )

        if not endpoint:
            return {"error": "端点不存在"}

        # 创建测试事件
        if not test_event:
            test_event = {
                "id": str(uuid.uuid4()),
                "event_type": "test_event",
                "data": {"message": "这是一个测试事件"},
                "timestamp": time.time(),
                "source": "webhook_test",
                "version": "1.0"
            }

        # 发送测试请求
        try:
            payload = json.dumps(test_event, separators=(',', ':'))
            signature = self.webhook_manager._generate_signature(payload, endpoint.secret)

            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": test_event["event_type"],
                "X-Webhook-Test": "true",
                "User-Agent": "LangGraph-Webhook-Test/1.0"
            }

            start_time = time.time()
            async with httpx.AsyncClient(timeout=endpoint.timeout) as client:
                response = await client.post(
                    endpoint.url,
                    json=test_event,
                    headers=headers
                )
            response_time = time.time() - start_time

            return {
                "success": True,
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "response_body": response.text[:500],
                "test_event": test_event
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "test_event": test_event
            }

# 故障排除CLI
@app.get("/api/webhooks/{endpoint_id}/diagnose")
async def diagnose_webhook_endpoint(endpoint_id: int):
    """诊断Webhook端点"""
    troubleshooter = WebhookTroubleshooter(webhook_manager)
    return await troubleshooter.diagnose_endpoint(endpoint_id)

@app.post("/api/webhooks/{endpoint_id}/test")
async def test_webhook_endpoint(
    endpoint_id: int,
    test_event: Optional[Dict[str, Any]] = None
):
    """测试Webhook端点"""
    troubleshooter = WebhookTroubleshooter(webhook_manager)
    return await troubleshooter.test_endpoint(endpoint_id, test_event)
```

## 部署配置

### 生产环境配置
```yaml
# webhook-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webhook-processor
  namespace: langgraph
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webhook-processor
  template:
    metadata:
      labels:
        app: webhook-processor
    spec:
      containers:
      - name: webhook-processor
        image: langgraph/webhook-processor:latest
        env:
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: WEBHOOK_WORKERS
          value: "5"
        - name: MAX_RETRIES
          value: "3"
        - name: RETRY_DELAY
          value: "60"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 下一步

- 📖 查看 [README.md](./README.md) - 部署选项总览
- 🔄 回到 [01-本地开发](./01-本地开发.md) - 开始实践

## 相关链接

- [LangGraph Platform Webhooks](https://langchain-ai.github.io/langgraph/cloud/how-tos/webhooks/)
- [Webhook 安全最佳实践](https://webhooks.fyi/security/overview)
- [事件驱动架构指南](https://martinfowler.com/articles/201701-event-driven.html)
- [HMAC 签名验证](https://tools.ietf.org/html/rfc2104)