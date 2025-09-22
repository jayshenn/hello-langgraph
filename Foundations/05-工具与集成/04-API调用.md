# API 调用

## 概述

API（Application Programming Interface）调用是 LangGraph Agent 与外部服务交互的重要方式。通过 API 集成，Agent 可以获取实时数据、执行远程操作、访问第三方服务等。本文档将详细介绍如何在 LangGraph 中安全、高效地集成各种 API。

## API 集成的核心概念

### 为什么需要 API 集成？

1. **实时数据访问**: 获取最新的天气、股价、新闻等信息
2. **功能扩展**: 利用专业服务如翻译、图像识别、语音合成
3. **系统集成**: 连接企业内部系统、数据库、微服务
4. **自动化操作**: 执行发邮件、创建文档、更新记录等任务

### API 调用的挑战

- **认证和授权**: 安全地管理 API 密钥和访问权限
- **错误处理**: 处理网络错误、超时、限流等问题
- **数据转换**: 处理不同的数据格式和结构
- **性能优化**: 管理并发请求、缓存、重试机制

## 基础 HTTP API 工具

### REST API 工具封装

```python
import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional, Union
from langchain_core.tools import BaseTool
from pydantic import Field
import json
import time
from urllib.parse import urljoin, urlencode

class HTTPAPITool(BaseTool):
    """通用 HTTP API 调用工具"""

    name: str = "http_api_call"
    description: str = "调用 HTTP API 获取数据或执行操作"

    base_url: str = Field(description="API 基础URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="默认请求头")
    timeout: int = Field(default=30, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")

    def __init__(self, **data):
        super().__init__(**data)
        # 设置默认请求头
        if "User-Agent" not in self.headers:
            self.headers["User-Agent"] = "LangGraph-Agent/1.0"

    def _run(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """执行 HTTP 请求"""

        url = urljoin(self.base_url, endpoint)
        headers = {**self.headers}
        if additional_headers:
            headers.update(additional_headers)

        # 准备请求参数
        request_kwargs = {
            "url": url,
            "method": method.upper(),
            "headers": headers,
            "timeout": self.timeout
        }

        if params:
            request_kwargs["params"] = params

        if data:
            if headers.get("Content-Type", "").startswith("application/json"):
                request_kwargs["json"] = data
            else:
                request_kwargs["data"] = data

        # 执行请求（带重试机制）
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(**request_kwargs)
                return self._handle_response(response)

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(wait_time)
                else:
                    break

        # 所有重试都失败
        return {
            "error": f"请求失败: {str(last_exception)}",
            "url": url,
            "method": method,
            "attempts": self.max_retries + 1
        }

    async def _arun(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """异步执行 HTTP 请求"""

        url = urljoin(self.base_url, endpoint)
        headers = {**self.headers}
        if additional_headers:
            headers.update(additional_headers)

        # 准备请求参数
        request_kwargs = {
            "method": method.upper(),
            "headers": headers,
            "timeout": aiohttp.ClientTimeout(total=self.timeout)
        }

        if params:
            request_kwargs["params"] = params

        if data:
            if headers.get("Content-Type", "").startswith("application/json"):
                request_kwargs["json"] = data
            else:
                request_kwargs["data"] = data

        # 执行异步请求（带重试机制）
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(url=url, **request_kwargs) as response:
                        return await self._handle_async_response(response)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    print(f"异步请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    break

        return {
            "error": f"异步请求失败: {str(last_exception)}",
            "url": url,
            "method": method,
            "attempts": self.max_retries + 1
        }

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """处理同步响应"""
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "url": response.url
        }

        # 检查状态码
        if response.status_code >= 400:
            result["error"] = f"HTTP {response.status_code}: {response.reason}"
            try:
                result["error_details"] = response.json()
            except:
                result["error_details"] = response.text

            return result

        # 解析响应内容
        content_type = response.headers.get("Content-Type", "")
        try:
            if "application/json" in content_type:
                result["data"] = response.json()
            elif "text/" in content_type:
                result["data"] = response.text
            else:
                result["data"] = response.content.decode('utf-8', errors='ignore')

        except Exception as e:
            result["error"] = f"解析响应失败: {str(e)}"
            result["raw_content"] = response.text

        return result

    async def _handle_async_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """处理异步响应"""
        result = {
            "status_code": response.status,
            "headers": dict(response.headers),
            "url": str(response.url)
        }

        # 检查状态码
        if response.status >= 400:
            result["error"] = f"HTTP {response.status}: {response.reason}"
            try:
                result["error_details"] = await response.json()
            except:
                result["error_details"] = await response.text()

            return result

        # 解析响应内容
        content_type = response.headers.get("Content-Type", "")
        try:
            if "application/json" in content_type:
                result["data"] = await response.json()
            elif "text/" in content_type:
                result["data"] = await response.text()
            else:
                result["data"] = await response.text()

        except Exception as e:
            result["error"] = f"解析异步响应失败: {str(e)}"
            result["raw_content"] = await response.text()

        return result

# 创建具体的 API 工具示例
def create_weather_api_tool():
    """创建天气 API 工具"""

    weather_tool = HTTPAPITool(
        name="weather_api",
        description="获取天气信息",
        base_url="http://api.openweathermap.org/data/2.5/",
        headers={
            "Content-Type": "application/json"
        }
    )

    def get_weather(city: str, api_key: str) -> Dict[str, Any]:
        """获取指定城市的天气信息"""
        return weather_tool.run(
            endpoint="weather",
            method="GET",
            params={
                "q": city,
                "appid": api_key,
                "units": "metric",
                "lang": "zh_cn"
            }
        )

    return get_weather

# 测试天气 API
def test_weather_api():
    """测试天气 API（需要真实的 API 密钥）"""
    get_weather = create_weather_api_tool()

    # 注意: 需要替换为真实的 OpenWeatherMap API 密钥
    api_key = "your_openweathermap_api_key"

    if api_key != "your_openweathermap_api_key":
        result = get_weather("Beijing", api_key)
        print("天气信息:", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("请设置有效的 OpenWeatherMap API 密钥")

# test_weather_api()
```

### 特定服务 API 工具

```python
from langchain_core.tools import tool
from typing import List
import base64

class NewsAPITool(HTTPAPITool):
    """新闻 API 工具"""

    def __init__(self, api_key: str):
        super().__init__(
            name="news_api",
            description="获取最新新闻",
            base_url="https://newsapi.org/v2/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

    def get_top_headlines(self, country: str = "cn", category: str = None, page_size: int = 10) -> Dict[str, Any]:
        """获取头条新闻"""
        params = {
            "country": country,
            "pageSize": page_size
        }
        if category:
            params["category"] = category

        return self._run("top-headlines", "GET", params=params)

    def search_everything(self, query: str, sort_by: str = "relevancy", page_size: int = 10) -> Dict[str, Any]:
        """搜索新闻"""
        params = {
            "q": query,
            "sortBy": sort_by,
            "pageSize": page_size,
            "language": "zh"
        }

        return self._run("everything", "GET", params=params)

class TranslationAPITool(HTTPAPITool):
    """翻译 API 工具"""

    def __init__(self, api_key: str, service: str = "google"):
        self.service = service

        if service == "google":
            base_url = "https://translation.googleapis.com/language/translate/v2/"
            headers = {"Authorization": f"Bearer {api_key}"}
        elif service == "baidu":
            base_url = "https://fanyi-api.baidu.com/api/trans/vip/"
            headers = {}
        else:
            raise ValueError(f"不支持的翻译服务: {service}")

        super().__init__(
            name="translation_api",
            description="文本翻译服务",
            base_url=base_url,
            headers=headers
        )

    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
        """翻译文本"""
        if self.service == "google":
            return self._google_translate(text, target_language, source_language)
        elif self.service == "baidu":
            return self._baidu_translate(text, target_language, source_language)

    def _google_translate(self, text: str, target: str, source: str) -> Dict[str, Any]:
        """Google 翻译"""
        data = {
            "q": text,
            "target": target,
            "format": "text"
        }
        if source != "auto":
            data["source"] = source

        return self._run("", "POST", data=data)

    def _baidu_translate(self, text: str, target: str, source: str) -> Dict[str, Any]:
        """百度翻译（需要实现签名算法）"""
        # 注意: 百度翻译需要复杂的签名算法，这里只是示例结构
        import hashlib
        import random
        import time

        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5(f"{text}{salt}your_secret_key".encode()).hexdigest()

        data = {
            "q": text,
            "from": source,
            "to": target,
            "appid": "your_app_id",
            "salt": salt,
            "sign": sign
        }

        return self._run("translate", "POST", data=data)

# 创建综合 API 工具集合
@tool
def get_news_headlines(category: str = None, count: int = 5) -> str:
    """获取新闻头条

    Args:
        category: 新闻类别 (business, entertainment, general, health, science, sports, technology)
        count: 新闻数量

    Returns:
        新闻摘要
    """
    # 模拟新闻 API 调用
    mock_news = {
        "technology": [
            "人工智能技术在医疗领域取得重大突破",
            "新型量子计算机研发成功",
            "5G技术推动智慧城市建设"
        ],
        "business": [
            "全球经济复苏势头良好",
            "新能源汽车市场持续增长",
            "数字货币监管政策出台"
        ],
        "general": [
            "气候变化应对措施加强",
            "教育改革深入推进",
            "城市规划注重可持续发展"
        ]
    }

    news_list = mock_news.get(category, mock_news["general"])
    selected_news = news_list[:count]

    return f"最新{category or '综合'}新闻:\n" + "\n".join([f"• {news}" for news in selected_news])

@tool
def translate_text(text: str, target_language: str = "en") -> str:
    """翻译文本

    Args:
        text: 要翻译的文本
        target_language: 目标语言 (en, zh, ja, ko, fr, es, de)

    Returns:
        翻译结果
    """
    # 模拟翻译 API 调用
    translations = {
        "en": {
            "你好": "Hello",
            "谢谢": "Thank you",
            "再见": "Goodbye",
            "人工智能": "Artificial Intelligence"
        },
        "ja": {
            "你好": "こんにちは",
            "谢谢": "ありがとう",
            "再见": "さようなら",
            "人工智能": "人工知能"
        }
    }

    # 简单的词典查找（实际应该调用真实的翻译 API）
    if target_language in translations:
        for chinese, translation in translations[target_language].items():
            if chinese in text:
                text = text.replace(chinese, translation)

    return f"翻译结果 ({target_language}): {text}"

@tool
def get_stock_price(symbol: str) -> str:
    """获取股票价格

    Args:
        symbol: 股票代码

    Returns:
        股票价格信息
    """
    # 模拟股票 API 调用
    mock_prices = {
        "AAPL": {"price": 150.25, "change": "+2.15", "percent": "+1.45%"},
        "GOOGL": {"price": 2750.80, "change": "-15.30", "percent": "-0.55%"},
        "TSLA": {"price": 245.67, "change": "+8.92", "percent": "+3.77%"},
        "MSFT": {"price": 310.45, "change": "+5.23", "percent": "+1.71%"}
    }

    if symbol.upper() in mock_prices:
        data = mock_prices[symbol.upper()]
        return f"{symbol.upper()}: ${data['price']} ({data['change']}, {data['percent']})"
    else:
        return f"未找到股票代码 {symbol} 的信息"
```

## 高级 API 集成技术

### 认证和安全管理

```python
import os
from typing import Optional
from cryptography.fernet import Fernet
import jwt
from datetime import datetime, timedelta

class APIKeyManager:
    """API 密钥管理器"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.keys = {}

    def store_key(self, service_name: str, api_key: str) -> None:
        """安全存储 API 密钥"""
        encrypted_key = self.cipher.encrypt(api_key.encode())
        self.keys[service_name] = encrypted_key

    def get_key(self, service_name: str) -> Optional[str]:
        """获取 API 密钥"""
        encrypted_key = self.keys.get(service_name)
        if encrypted_key:
            return self.cipher.decrypt(encrypted_key).decode()
        return None

    def load_from_env(self, service_name: str, env_var: str) -> None:
        """从环境变量加载密钥"""
        api_key = os.getenv(env_var)
        if api_key:
            self.store_key(service_name, api_key)

    def remove_key(self, service_name: str) -> None:
        """删除 API 密钥"""
        if service_name in self.keys:
            del self.keys[service_name]

class OAuthManager:
    """OAuth 认证管理器"""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    def get_access_token(self) -> Optional[str]:
        """获取访问令牌"""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # 令牌过期，尝试刷新
        if self.refresh_token:
            return self.refresh_access_token()

        return None

    def refresh_access_token(self) -> Optional[str]:
        """刷新访问令牌"""
        # 这里应该实现实际的 OAuth 刷新逻辑
        # 每个服务的实现会有所不同
        pass

    def set_tokens(self, access_token: str, refresh_token: str = None, expires_in: int = 3600):
        """设置令牌"""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

class SecureAPITool(HTTPAPITool):
    """安全的 API 工具"""

    def __init__(self, key_manager: APIKeyManager, service_name: str, **kwargs):
        self.key_manager = key_manager
        self.service_name = service_name
        super().__init__(**kwargs)

    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        api_key = self.key_manager.get_key(self.service_name)
        if api_key:
            # 根据服务类型返回不同的认证头
            if self.service_name == "openai":
                return {"Authorization": f"Bearer {api_key}"}
            elif self.service_name == "anthropic":
                return {"x-api-key": api_key}
            elif self.service_name == "google":
                return {"Authorization": f"Bearer {api_key}"}
            else:
                return {"Authorization": f"Bearer {api_key}"}
        return {}

    def _run(self, endpoint: str, method: str = "GET", **kwargs):
        """安全执行请求"""
        # 添加认证头
        auth_headers = self._get_auth_headers()
        additional_headers = kwargs.pop("additional_headers", {})
        additional_headers.update(auth_headers)
        kwargs["additional_headers"] = additional_headers

        return super()._run(endpoint, method, **kwargs)

# 使用示例
def create_secure_api_setup():
    """创建安全的 API 设置"""

    # 创建密钥管理器
    key_manager = APIKeyManager()

    # 从环境变量加载密钥
    key_manager.load_from_env("openai", "OPENAI_API_KEY")
    key_manager.load_from_env("anthropic", "ANTHROPIC_API_KEY")
    key_manager.load_from_env("google", "GOOGLE_API_KEY")

    # 创建安全的 API 工具
    openai_tool = SecureAPITool(
        key_manager=key_manager,
        service_name="openai",
        name="openai_api",
        description="OpenAI API 调用",
        base_url="https://api.openai.com/v1/"
    )

    return key_manager, openai_tool
```

### 速率限制和并发控制

```python
import asyncio
from asyncio import Semaphore
from collections import defaultdict, deque
from time import time
from typing import Dict, Any

class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)

    def is_allowed(self, key: str = "default") -> bool:
        """检查是否允许请求"""
        now = time()
        request_times = self.requests[key]

        # 清理过期的请求记录
        while request_times and request_times[0] <= now - self.time_window:
            request_times.popleft()

        # 检查是否超过限制
        if len(request_times) >= self.max_requests:
            return False

        # 记录当前请求
        request_times.append(now)
        return True

    def wait_time(self, key: str = "default") -> float:
        """计算需要等待的时间"""
        if self.is_allowed(key):
            return 0

        request_times = self.requests[key]
        if request_times:
            oldest_request = request_times[0]
            return self.time_window - (time() - oldest_request)
        return 0

class ConcurrencyController:
    """并发控制器"""

    def __init__(self, max_concurrent: int):
        self.semaphore = Semaphore(max_concurrent)
        self.active_requests = 0

    async def __aenter__(self):
        await self.semaphore.acquire()
        self.active_requests += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()
        self.active_requests -= 1

class SmartAPITool(SecureAPITool):
    """智能 API 工具（支持速率限制和并发控制）"""

    def __init__(
        self,
        rate_limiter: RateLimiter = None,
        max_concurrent: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rate_limiter = rate_limiter or RateLimiter(60, 60)  # 默认每分钟60请求
        self.concurrency_controller = ConcurrencyController(max_concurrent)

    def _run(self, endpoint: str, method: str = "GET", **kwargs):
        """带速率限制的同步请求"""
        # 检查速率限制
        if not self.rate_limiter.is_allowed(self.service_name):
            wait_time = self.rate_limiter.wait_time(self.service_name)
            print(f"速率限制，等待 {wait_time:.2f} 秒")
            time.sleep(wait_time)

        return super()._run(endpoint, method, **kwargs)

    async def _arun(self, endpoint: str, method: str = "GET", **kwargs):
        """带速率限制和并发控制的异步请求"""
        # 检查速率限制
        if not self.rate_limiter.is_allowed(self.service_name):
            wait_time = self.rate_limiter.wait_time(self.service_name)
            print(f"速率限制，等待 {wait_time:.2f} 秒")
            await asyncio.sleep(wait_time)

        # 并发控制
        async with self.concurrency_controller:
            return await super()._arun(endpoint, method, **kwargs)

# 批量 API 调用管理
class BatchAPIManager:
    """批量 API 调用管理器"""

    def __init__(self, api_tool: SmartAPITool, batch_size: int = 10):
        self.api_tool = api_tool
        self.batch_size = batch_size

    async def batch_process(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理请求"""
        results = []

        # 分批处理
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

            # 批次间等待，避免过载
            if i + self.batch_size < len(requests):
                await asyncio.sleep(1)

        return results

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理单个批次"""
        tasks = []

        for request in batch:
            task = self.api_tool._arun(**request)
            tasks.append(task)

        # 并发执行批次内的请求
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "request": batch[i]
                })
            else:
                processed_results.append(result)

        return processed_results

# 使用示例
async def demonstrate_advanced_api_usage():
    """演示高级 API 使用"""

    # 创建速率限制器（每分钟30个请求）
    rate_limiter = RateLimiter(max_requests=30, time_window=60)

    # 创建密钥管理器
    key_manager = APIKeyManager()
    key_manager.store_key("weather", "demo_api_key")

    # 创建智能 API 工具
    api_tool = SmartAPITool(
        key_manager=key_manager,
        service_name="weather",
        rate_limiter=rate_limiter,
        max_concurrent=3,
        name="weather_api",
        description="天气 API",
        base_url="http://api.openweathermap.org/data/2.5/"
    )

    # 创建批量管理器
    batch_manager = BatchAPIManager(api_tool, batch_size=5)

    # 准备批量请求
    cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"]
    requests = [
        {
            "endpoint": "weather",
            "method": "GET",
            "params": {"q": city, "units": "metric"}
        }
        for city in cities
    ]

    # 执行批量请求
    print("开始批量天气查询...")
    results = await batch_manager.batch_process(requests)

    print(f"完成 {len(results)} 个城市的天气查询")
    for i, result in enumerate(results):
        if "error" in result:
            print(f"{cities[i]}: 查询失败 - {result['error']}")
        else:
            print(f"{cities[i]}: 查询成功 - 状态码 {result.get('status_code', 'unknown')}")

# 运行演示
# asyncio.run(demonstrate_advanced_api_usage())
```

### 缓存和性能优化

```python
import pickle
import hashlib
from typing import Any, Optional
from datetime import datetime, timedelta
import json

class APICache:
    """API 响应缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}

    def _generate_key(self, endpoint: str, method: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        cache_data = {
            "endpoint": endpoint,
            "method": method,
            "params": params or {},
            "data": data or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存内容"""
        if key in self.cache:
            cached_item = self.cache[key]

            # 检查是否过期
            if datetime.now() < cached_item["expires_at"]:
                self.access_times[key] = datetime.now()
                return cached_item["data"]
            else:
                # 过期，删除
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        return None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """设置缓存内容"""
        # 检查缓存大小限制
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        self.cache[key] = {
            "data": data,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        self.access_times[key] = datetime.now()

    def _evict_lru(self) -> None:
        """移除最近最少使用的缓存项"""
        if not self.access_times:
            return

        # 找到最久未访问的键
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # 删除
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, '_hit_count', 0) / getattr(self, '_total_requests', 1)
        }

class CachedAPITool(SmartAPITool):
    """带缓存的 API 工具"""

    def __init__(self, cache: APICache = None, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache or APICache()
        self._hit_count = 0
        self._miss_count = 0

    def _run(self, endpoint: str, method: str = "GET", cache_ttl: Optional[int] = None, **kwargs):
        """带缓存的同步请求"""
        # 生成缓存键
        cache_key = self.cache._generate_key(
            endpoint,
            method,
            kwargs.get("params"),
            kwargs.get("data")
        )

        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self._hit_count += 1
            print(f"缓存命中: {endpoint}")
            return cached_result

        # 缓存未命中，执行请求
        self._miss_count += 1
        print(f"缓存未命中: {endpoint}")
        result = super()._run(endpoint, method, **kwargs)

        # 如果请求成功，缓存结果
        if result and "error" not in result:
            self.cache.set(cache_key, result, cache_ttl)

        return result

    async def _arun(self, endpoint: str, method: str = "GET", cache_ttl: Optional[int] = None, **kwargs):
        """带缓存的异步请求"""
        # 生成缓存键
        cache_key = self.cache._generate_key(
            endpoint,
            method,
            kwargs.get("params"),
            kwargs.get("data")
        )

        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self._hit_count += 1
            print(f"异步缓存命中: {endpoint}")
            return cached_result

        # 缓存未命中，执行异步请求
        self._miss_count += 1
        print(f"异步缓存未命中: {endpoint}")
        result = await super()._arun(endpoint, method, **kwargs)

        # 如果请求成功，缓存结果
        if result and "error" not in result:
            self.cache.set(cache_key, result, cache_ttl)

        return result

    def cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache.cache)
        }

# 性能监控装饰器
def api_performance_monitor(func):
    """API 性能监控装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time()

        try:
            result = func(*args, **kwargs)
            duration = time() - start_time

            print(f"API 调用完成: {func.__name__} - 耗时: {duration:.3f}s")

            # 记录性能指标
            if hasattr(func, '_performance_stats'):
                func._performance_stats.append(duration)
            else:
                func._performance_stats = [duration]

            return result

        except Exception as e:
            duration = time() - start_time
            print(f"API 调用失败: {func.__name__} - 耗时: {duration:.3f}s - 错误: {e}")
            raise

    return wrapper

class PerformanceAPITool(CachedAPITool):
    """性能监控的 API 工具"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_stats = []

    @api_performance_monitor
    def _run(self, *args, **kwargs):
        return super()._run(*args, **kwargs)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.performance_stats:
            return {"message": "暂无性能数据"}

        durations = [stat for stat in self.performance_stats if isinstance(stat, (int, float))]

        if not durations:
            return {"message": "暂无有效性能数据"}

        return {
            "total_calls": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "cache_stats": self.cache_stats()
        }
```

## 实际应用示例

### 创建综合 API 工具集

```python
from langchain_core.tools import tool

class ComprehensiveAPIToolkit:
    """综合 API 工具包"""

    def __init__(self):
        self.key_manager = APIKeyManager()
        self.cache = APICache(max_size=500, default_ttl=1800)  # 30分钟缓存
        self.tools = self._create_tools()

    def _create_tools(self):
        """创建所有 API 工具"""
        tools = {}

        # 天气工具
        tools['weather'] = self._create_weather_tool()

        # 新闻工具
        tools['news'] = self._create_news_tool()

        # 翻译工具
        tools['translation'] = self._create_translation_tool()

        # 股票工具
        tools['stock'] = self._create_stock_tool()

        return tools

    def _create_weather_tool(self):
        @tool
        def get_weather_info(city: str, units: str = "metric") -> str:
            """获取城市天气信息

            Args:
                city: 城市名称
                units: 温度单位 (metric, imperial)

            Returns:
                天气信息描述
            """
            # 模拟天气 API 调用
            weather_data = {
                "Beijing": {"temp": 15, "desc": "多云", "humidity": 65},
                "Shanghai": {"temp": 20, "desc": "晴天", "humidity": 55},
                "Guangzhou": {"temp": 25, "desc": "小雨", "humidity": 80},
                "Shenzhen": {"temp": 26, "desc": "晴天", "humidity": 70}
            }

            city_data = weather_data.get(city, {"temp": 18, "desc": "未知", "humidity": 60})

            temp_unit = "°C" if units == "metric" else "°F"

            return f"{city}天气: {city_data['desc']}, 温度 {city_data['temp']}{temp_unit}, 湿度 {city_data['humidity']}%"

        return get_weather_info

    def _create_news_tool(self):
        @tool
        def search_news(query: str, category: str = "general") -> str:
            """搜索新闻

            Args:
                query: 搜索关键词
                category: 新闻类别

            Returns:
                新闻搜索结果
            """
            # 模拟新闻搜索
            news_results = [
                f"关于'{query}'的最新报道：技术发展迅速",
                f"专家分析'{query}'的未来趋势",
                f"'{query}'相关政策解读"
            ]

            return f"新闻搜索结果:\n" + "\n".join([f"• {news}" for news in news_results])

        return search_news

    def _create_translation_tool(self):
        @tool
        def translate_content(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
            """翻译文本内容

            Args:
                text: 要翻译的文本
                target_lang: 目标语言
                source_lang: 源语言

            Returns:
                翻译结果
            """
            # 模拟翻译服务
            simple_translations = {
                "zh-en": {
                    "你好": "Hello",
                    "谢谢": "Thank you",
                    "人工智能": "Artificial Intelligence",
                    "机器学习": "Machine Learning"
                }
            }

            translation_key = f"{source_lang}-{target_lang}" if source_lang != "auto" else f"zh-{target_lang}"
            translations = simple_translations.get(translation_key, {})

            translated_text = text
            for chinese, english in translations.items():
                if chinese in text:
                    translated_text = translated_text.replace(chinese, english)

            return f"翻译结果 ({target_lang}): {translated_text}"

        return translate_content

    def _create_stock_tool(self):
        @tool
        def get_stock_info(symbol: str, info_type: str = "price") -> str:
            """获取股票信息

            Args:
                symbol: 股票代码
                info_type: 信息类型 (price, company, analysis)

            Returns:
                股票信息
            """
            # 模拟股票数据
            stock_data = {
                "AAPL": {
                    "price": {"current": 150.25, "change": "+2.15%"},
                    "company": "Apple Inc.",
                    "analysis": "技术指标显示买入信号"
                },
                "GOOGL": {
                    "price": {"current": 2750.80, "change": "-0.55%"},
                    "company": "Alphabet Inc.",
                    "analysis": "基本面强劲，长期看涨"
                }
            }

            if symbol.upper() not in stock_data:
                return f"未找到股票代码 {symbol} 的信息"

            data = stock_data[symbol.upper()]

            if info_type == "price":
                price_info = data["price"]
                return f"{symbol}: ${price_info['current']} ({price_info['change']})"
            elif info_type == "company":
                return f"{symbol}: {data['company']}"
            elif info_type == "analysis":
                return f"{symbol} 分析: {data['analysis']}"
            else:
                return f"{symbol} 综合信息: {data}"

        return get_stock_info

    def get_all_tools(self):
        """获取所有工具"""
        return list(self.tools.values())

    def get_tool(self, tool_name: str):
        """获取特定工具"""
        return self.tools.get(tool_name)

# 创建和使用 API 工具包
def create_api_toolkit_demo():
    """创建 API 工具包演示"""

    # 创建工具包
    toolkit = ComprehensiveAPIToolkit()

    # 获取所有工具
    tools = toolkit.get_all_tools()

    print(f"可用工具数量: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # 测试各个工具
    print("\n=== 工具测试 ===")

    # 测试天气工具
    weather_tool = toolkit.get_tool('weather')
    weather_result = weather_tool.invoke({"city": "Beijing"})
    print(f"天气查询: {weather_result}")

    # 测试翻译工具
    translation_tool = toolkit.get_tool('translation')
    translation_result = translation_tool.invoke({
        "text": "人工智能",
        "target_lang": "en"
    })
    print(f"翻译结果: {translation_result}")

    # 测试股票工具
    stock_tool = toolkit.get_tool('stock')
    stock_result = stock_tool.invoke({
        "symbol": "AAPL",
        "info_type": "price"
    })
    print(f"股票信息: {stock_result}")

    return toolkit

# 运行演示
toolkit = create_api_toolkit_demo()
```

## 最佳实践

### 1. API 设计原则

- **统一接口**: 保持一致的调用方式和返回格式
- **错误处理**: 优雅地处理各种异常情况
- **文档完善**: 详细的参数说明和使用示例
- **性能优化**: 合理使用缓存、批处理、并发控制

### 2. 安全考虑

```python
# 安全最佳实践示例
class SecureAPIBestPractices:
    """API 安全最佳实践"""

    @staticmethod
    def validate_input(data: Any) -> bool:
        """验证输入数据"""
        # 检查数据类型
        if not isinstance(data, (str, dict, list)):
            return False

        # 检查字符串长度
        if isinstance(data, str) and len(data) > 10000:
            return False

        # 检查恶意内容
        dangerous_patterns = ['<script>', 'javascript:', 'eval(', 'exec(']
        if isinstance(data, str):
            for pattern in dangerous_patterns:
                if pattern.lower() in data.lower():
                    return False

        return True

    @staticmethod
    def sanitize_output(data: Any) -> Any:
        """清理输出数据"""
        if isinstance(data, dict):
            # 移除敏感字段
            sensitive_fields = ['password', 'token', 'key', 'secret']
            return {k: v for k, v in data.items() if k.lower() not in sensitive_fields}

        return data

    @staticmethod
    def log_api_call(endpoint: str, success: bool, duration: float):
        """记录 API 调用日志"""
        timestamp = datetime.now().isoformat()
        status = "SUCCESS" if success else "FAILED"
        print(f"[{timestamp}] API Call - {endpoint} - {status} - {duration:.3f}s")
```

### 3. 监控和调试

```python
# API 监控和调试工具
class APIMonitor:
    """API 监控器"""

    def __init__(self):
        self.call_history = []
        self.error_count = 0
        self.total_calls = 0

    def record_call(self, endpoint: str, success: bool, duration: float, error: str = None):
        """记录 API 调用"""
        self.total_calls += 1
        if not success:
            self.error_count += 1

        self.call_history.append({
            "timestamp": datetime.now(),
            "endpoint": endpoint,
            "success": success,
            "duration": duration,
            "error": error
        })

        # 保持历史记录大小
        if len(self.call_history) > 1000:
            self.call_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.total_calls == 0:
            return {"message": "暂无调用记录"}

        success_rate = (self.total_calls - self.error_count) / self.total_calls
        avg_duration = sum(call["duration"] for call in self.call_history) / len(self.call_history)

        return {
            "total_calls": self.total_calls,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "recent_errors": [
                call for call in self.call_history[-10:]
                if not call["success"]
            ]
        }
```

## 总结

API 集成是扩展 LangGraph Agent 能力的关键技术，要点包括：

1. **统一封装**: 创建标准化的 API 工具接口
2. **安全管理**: 妥善保护 API 密钥和敏感信息
3. **性能优化**: 使用缓存、批处理、并发控制
4. **错误处理**: 实现重试机制和优雅降级
5. **监控调试**: 记录调用日志和性能指标

下一章节将介绍 MCP（Model Context Protocol）协议的集成和使用。