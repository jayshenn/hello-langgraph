"""
Python 异步编程练习

包含从基础到进阶的异步编程示例，模拟 LangGraph 中的实际应用场景。
每个练习都有 TODO 标记，需要你完成异步实现。
"""

import asyncio
import aiohttp
import time
import random
from typing import TypedDict, List, Dict, Any, Optional
from asyncio import Queue

# =============================================================================
# 练习 1: 基础异步函数
# =============================================================================

async def simulate_api_call(api_name: str, delay: float = 1.0) -> str:
    """
    TODO: 完善这个函数，模拟异步 API 调用
    参数:
        api_name: API 名称
        delay: 模拟延迟时间
    返回:
        API 响应字符串
    """
    # TODO: 添加开始日志
    # TODO: 使用 asyncio.sleep 模拟网络延迟
    # TODO: 添加完成日志
    # TODO: 返回模拟的 API 响应
    pass


async def test_basic_async():
    """测试基础异步函数"""
    print("🧪 测试基础异步函数")

    # TODO: 调用 simulate_api_call
    # TODO: 打印结果
    pass


# =============================================================================
# 练习 2: 并发执行多个任务
# =============================================================================

async def fetch_user_profile(user_id: str) -> Dict[str, Any]:
    """
    TODO: 获取用户档案信息
    模拟从多个服务并发获取用户数据
    """
    # TODO: 定义多个异步任务:
    #   - get_basic_info(user_id) - 耗时 1 秒
    #   - get_preferences(user_id) - 耗时 0.5 秒
    #   - get_activity_history(user_id) - 耗时 2 秒
    # TODO: 使用 asyncio.gather 并发执行
    # TODO: 组合结果并返回
    pass


async def get_basic_info(user_id: str) -> Dict[str, Any]:
    """获取基础信息"""
    await asyncio.sleep(1)
    return {
        "user_id": user_id,
        "name": f"用户_{user_id}",
        "email": f"user_{user_id}@example.com"
    }


async def get_preferences(user_id: str) -> Dict[str, Any]:
    """获取用户偏好"""
    await asyncio.sleep(0.5)
    return {
        "theme": "dark",
        "language": "zh-CN",
        "notifications": True
    }


async def get_activity_history(user_id: str) -> List[str]:
    """获取活动历史"""
    await asyncio.sleep(2)
    return [
        "2024-01-01: 登录",
        "2024-01-02: 查看文档",
        "2024-01-03: 更新设置"
    ]


async def test_concurrent_fetch():
    """测试并发获取数据"""
    print("🧪 测试并发获取用户数据")

    start_time = time.time()
    # TODO: 调用 fetch_user_profile
    end_time = time.time()

    # TODO: 打印结果和耗时
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")


# =============================================================================
# 练习 3: LangGraph 风格的异步节点
# =============================================================================

class ChatState(TypedDict):
    user_input: str
    llm_response: str
    search_results: List[str]
    summary: str
    processing_time: float


async def llm_processing_node(state: ChatState) -> ChatState:
    """
    TODO: LLM 处理节点
    模拟调用大语言模型生成响应
    """
    # TODO: 模拟 LLM 调用（耗时 1.5 秒）
    # TODO: 生成响应内容
    # TODO: 更新状态并返回
    pass


async def web_search_node(state: ChatState) -> ChatState:
    """
    TODO: 网络搜索节点
    模拟并发搜索多个搜索引擎
    """
    # TODO: 定义多个搜索任务（每个耗时 1 秒）
    # TODO: 并发执行搜索
    # TODO: 整合搜索结果
    # TODO: 更新状态并返回
    pass


async def summarization_node(state: ChatState) -> ChatState:
    """
    TODO: 总结节点
    基于 LLM 响应和搜索结果生成总结
    """
    # TODO: 模拟总结处理（耗时 0.8 秒）
    # TODO: 生成总结内容
    # TODO: 更新状态并返回
    pass


async def run_chat_pipeline(user_input: str) -> ChatState:
    """
    TODO: 运行完整的聊天流水线
    要求：LLM 和搜索节点并发执行，然后执行总结
    """
    # TODO: 创建初始状态
    # TODO: 并发执行 LLM 和搜索节点
    # TODO: 合并两个节点的结果
    # TODO: 执行总结节点
    # TODO: 返回最终状态
    pass


async def test_chat_pipeline():
    """测试聊天流水线"""
    print("🧪 测试聊天流水线")

    start_time = time.time()
    # TODO: 运行流水线
    end_time = time.time()

    # TODO: 打印最终状态和总耗时


# =============================================================================
# 练习 4: 异步队列处理
# =============================================================================

async def task_producer(queue: Queue, task_count: int = 5):
    """
    TODO: 任务生产者
    向队列中添加处理任务
    """
    # TODO: 循环生成任务
    # TODO: 将任务放入队列
    # TODO: 添加生产日志
    pass


async def task_consumer(queue: Queue, consumer_id: str):
    """
    TODO: 任务消费者
    从队列中获取并处理任务
    """
    while True:
        try:
            # TODO: 从队列获取任务（设置超时）
            # TODO: 模拟处理任务
            # TODO: 标记任务完成
            # TODO: 添加处理日志
            pass
        except asyncio.TimeoutError:
            # TODO: 处理超时退出
            break
        except Exception as e:
            # TODO: 处理异常
            print(f"❌ 消费者 {consumer_id} 处理异常: {e}")


async def test_producer_consumer():
    """测试生产者-消费者模式"""
    print("🧪 测试生产者-消费者模式")

    # TODO: 创建队列
    # TODO: 启动生产者
    # TODO: 启动多个消费者
    # TODO: 等待生产者完成
    # TODO: 等待队列中的任务完成
    # TODO: 取消消费者
    pass


# =============================================================================
# 练习 5: 异步上下文管理器
# =============================================================================

class AsyncResourceManager:
    """
    TODO: 异步资源管理器
    管理异步资源的获取和释放
    """

    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.resource = None

    async def __aenter__(self):
        """
        TODO: 异步进入上下文
        模拟资源获取
        """
        # TODO: 模拟资源获取过程
        # TODO: 记录获取日志
        # TODO: 返回资源
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        TODO: 异步退出上下文
        模拟资源释放
        """
        # TODO: 模拟资源释放过程
        # TODO: 记录释放日志
        pass

    async def do_work(self, work_item: str) -> str:
        """
        TODO: 使用资源执行工作
        """
        # TODO: 模拟工作处理
        # TODO: 返回工作结果
        pass


async def test_async_context_manager():
    """测试异步上下文管理器"""
    print("🧪 测试异步上下文管理器")

    # TODO: 使用异步上下文管理器
    # TODO: 执行多个工作项
    # TODO: 打印结果


# =============================================================================
# 练习 6: 异步错误处理和重试
# =============================================================================

class AsyncRetryManager:
    """异步重试管理器"""

    @staticmethod
    async def retry_async(
        coro_func,
        *args,
        max_retries: int = 3,
        delay: float = 1.0,
        **kwargs
    ):
        """
        TODO: 异步重试装饰器
        对失败的异步操作进行重试
        """
        # TODO: 实现重试逻辑
        # TODO: 记录重试日志
        # TODO: 处理最终失败情况
        pass


async def unreliable_async_service(data: str, failure_rate: float = 0.7) -> str:
    """
    不可靠的异步服务
    模拟有一定失败率的网络服务
    """
    await asyncio.sleep(0.5)  # 模拟网络延迟

    if random.random() < failure_rate:
        raise ConnectionError(f"服务调用失败: {data}")

    return f"处理成功: {data}"


async def test_async_retry():
    """测试异步重试"""
    print("🧪 测试异步重试机制")

    # TODO: 使用 AsyncRetryManager 重试不可靠服务
    # TODO: 测试多个数据项
    # TODO: 记录成功和失败情况


# =============================================================================
# 练习 7: 异步流处理
# =============================================================================

async def async_data_stream(data_source: List[str]):
    """
    TODO: 异步数据流生成器
    逐个产生处理后的数据项
    """
    # TODO: 遍历数据源
    # TODO: 异步处理每个数据项
    # TODO: 使用 yield 产生结果
    pass


async def batch_process_stream(stream, batch_size: int = 3):
    """
    TODO: 批量处理流数据
    将流数据按批次处理
    """
    # TODO: 收集批次数据
    # TODO: 批量处理
    # TODO: 产生批次结果
    pass


async def test_async_stream():
    """测试异步流处理"""
    print("🧪 测试异步流处理")

    # TODO: 创建数据源
    # TODO: 创建异步流
    # TODO: 批量处理流数据
    # TODO: 处理每个批次结果


# =============================================================================
# 运行所有测试
# =============================================================================

async def run_all_tests():
    """运行所有异步练习测试"""
    print("🚀 开始异步编程练习测试\n")

    # TODO: 按顺序运行所有测试函数
    # 建议添加分隔线和计时
    tests = [
        ("基础异步函数", test_basic_async),
        ("并发数据获取", test_concurrent_fetch),
        ("聊天流水线", test_chat_pipeline),
        ("生产者消费者", test_producer_consumer),
        ("异步上下文管理器", test_async_context_manager),
        ("异步重试", test_async_retry),
        ("异步流处理", test_async_stream),
    ]

    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")

        start_time = time.time()
        try:
            await test_func()
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        end_time = time.time()

        print(f"⏱️ 测试耗时: {end_time - start_time:.2f} 秒\n")

    print("🎉 所有测试完成!")


if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(run_all_tests())


# =============================================================================
# 答案提示 (完成练习后可以查看)
# =============================================================================

"""
答案提示:

1. simulate_api_call:
   - 使用 print 记录开始和结束
   - 使用 await asyncio.sleep(delay) 模拟延迟
   - 返回格式化的响应字符串

2. fetch_user_profile:
   - 使用 asyncio.gather 并发执行三个获取函数
   - 组合结果到一个字典中

3. LangGraph 节点:
   - 每个节点函数都要返回更新后的状态
   - 使用 {**state, "new_field": value} 语法更新状态
   - run_chat_pipeline 中使用 asyncio.gather 并发执行

4. 队列处理:
   - 生产者使用 await queue.put(item)
   - 消费者使用 await asyncio.wait_for(queue.get(), timeout=3)
   - 使用 queue.task_done() 标记任务完成

5. 异步上下文管理器:
   - __aenter__ 获取资源并返回 self
   - __aexit__ 释放资源，处理异常

6. 重试机制:
   - 使用 for 循环实现重试
   - 捕获异常并在最后一次重试时重新抛出
   - 使用 await asyncio.sleep(delay) 在重试间隔

7. 异步流:
   - 使用 async def 定义异步生成器
   - 使用 yield 产生数据
   - 使用 async for 消费异步流

记住: 异步编程的关键是理解何时使用 await，何时使用并发！
"""