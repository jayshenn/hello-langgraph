# webhook_handler.py
import asyncio
import json
import hashlib
import hmac
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

app = FastAPI()

class WebhookHandler:
    """Webhook处理器"""

    def __init__(self, secret: str):
        self.secret = secret
        self.handlers = {}

    def register_handler(self, event_type: str, handler):
        """注册事件处理器"""
        self.handlers[event_type] = handler

    def verify_signature(self, payload: str, signature: str) -> bool:
        """验证Webhook签名"""
        expected_signature = hmac.new(
            self.secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected_signature}", signature)

    async def process_webhook(self, request: Request, background_tasks: BackgroundTasks):
        """处理Webhook请求"""
        # 验证签名
        signature = request.headers.get("X-Webhook-Signature")
        if not signature:
            raise HTTPException(status_code=401, detail="缺少签名")

        body = await request.body()
        if not self.verify_signature(body.decode('utf-8'), signature):
            raise HTTPException(status_code=401, detail="签名验证失败")

        # 解析事件
        try:
            event_data = json.loads(body)
            event_type = event_data.get("event_type")

            # 查找处理器
            handler = self.handlers.get(event_type)
            if handler:
                background_tasks.add_task(handler, event_data)
                return {"status": "accepted", "event_type": event_type}
            else:
                return {"status": "ignored", "event_type": event_type}

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="无效的JSON")

# 创建处理器实例
webhook_handler = WebhookHandler("your-webhook-secret")

# 注册事件处理器
async def handle_agent_completed(event_data: Dict[str, Any]):
    """处理Agent完成事件"""
    print(f"Agent执行完成: {event_data}")
    # 在这里添加你的业务逻辑

async def handle_agent_error(event_data: Dict[str, Any]):
    """处理Agent错误事件"""
    print(f"Agent执行错误: {event_data}")
    # 发送告警通知

webhook_handler.register_handler("agent_completed", handle_agent_completed)
webhook_handler.register_handler("agent_invocation_error", handle_agent_error)

@app.post("/webhooks/langgraph")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """接收LangGraph Webhook"""
    return await webhook_handler.process_webhook(request, background_tasks)