"""
聊天机器人系统配置

这个文件包含了聊天机器人的所有配置选项，包括模型参数、功能开关、业务规则等。
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """大语言模型配置"""
    # 提供商选择
    provider: str = "openai"  # openai, anthropic, azure, local

    # 模型配置
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # API 配置
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

    # 特殊配置
    stream: bool = False

    def __post_init__(self):
        """初始化后处理"""
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

        # 根据提供商调整默认模型
        if self.provider == "anthropic" and self.model_name == "gpt-3.5-turbo":
            self.model_name = "claude-3-haiku-20240307"


@dataclass
class MemoryConfig:
    """记忆管理配置"""
    # 短期记忆
    short_term_size: int = 10
    short_term_ttl: int = 3600  # 秒

    # 长期记忆
    enable_long_term: bool = True
    importance_threshold: float = 0.7
    max_long_term_memories: int = 100

    # 记忆存储
    storage_type: str = "memory"  # memory, file, database
    storage_path: str = "./data/memories"

    # 记忆检索
    max_relevant_memories: int = 5
    similarity_threshold: float = 0.6


@dataclass
class ToolsConfig:
    """工具配置"""
    # 工具开关
    enable_tools: bool = True
    enabled_tools: List[str] = field(default_factory=lambda: [
        "get_current_time",
        "calculate",
        "get_weather",
        "search_knowledge_base"
    ])

    # 工具限制
    max_tool_calls_per_turn: int = 3
    tool_timeout: int = 10

    # 外部服务配置
    weather_api_key: Optional[str] = None
    search_api_key: Optional[str] = None


@dataclass
class ConversationConfig:
    """对话配置"""
    # 对话限制
    max_turns: int = 50
    max_input_length: int = 2000
    max_response_length: int = 1000

    # 会话管理
    session_timeout: int = 1800  # 30分钟
    enable_context_compression: bool = True
    context_window_size: int = 4000

    # 多轮对话
    enable_intent_tracking: bool = True
    enable_emotion_analysis: bool = True
    enable_satisfaction_tracking: bool = True


@dataclass
class QualityConfig:
    """质量控制配置"""
    # 置信度阈值
    min_confidence_threshold: float = 0.5
    escalation_threshold: float = 0.3

    # 情感监控
    negative_emotion_threshold: float = 0.3

    # 错误处理
    max_consecutive_errors: int = 3
    error_recovery_enabled: bool = True

    # 质量评估
    enable_quality_scoring: bool = True
    quality_threshold: float = 0.6


@dataclass
class EscalationConfig:
    """升级规则配置"""
    # 自动升级条件
    enable_auto_escalation: bool = True

    # 关键词触发
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        "投诉", "举报", "法律", "律师", "起诉", "退款", "赔偿",
        "差评", "不满", "愤怒", "欺骗", "虚假", "骗子"
    ])

    # 情感触发
    emotion_escalation_enabled: bool = True
    emotion_threshold: float = 0.2

    # 错误触发
    error_escalation_enabled: bool = True
    max_errors_before_escalation: int = 2

    # 复杂度触发
    complexity_escalation_enabled: bool = True
    max_turns_before_escalation: int = 15

    # 人工客服配置
    human_agent_available_hours: List[int] = field(default_factory=lambda: list(range(9, 18)))
    fallback_message: str = "当前非工作时间，您的问题已记录，我们会在工作时间内回复您。"


@dataclass
class SecurityConfig:
    """安全配置"""
    # 输入过滤
    enable_input_filtering: bool = True
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"eval\(",
        r"exec\("
    ])

    # 输出过滤
    enable_output_filtering: bool = True

    # 速率限制
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000

    # 数据保护
    log_user_input: bool = False
    anonymize_logs: bool = True
    data_retention_days: int = 30


@dataclass
class LoggingConfig:
    """日志配置"""
    # 日志级别
    log_level: str = "INFO"

    # 日志文件
    enable_file_logging: bool = True
    log_file_path: str = "./logs/chatbot.log"
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # 日志内容
    log_user_interactions: bool = True
    log_tool_calls: bool = True
    log_llm_requests: bool = True
    log_errors: bool = True

    # 结构化日志
    structured_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 性能监控
    enable_performance_monitoring: bool = True
    response_time_threshold: float = 5.0

    # 错误监控
    enable_error_tracking: bool = True
    error_notification_threshold: int = 10

    # 用户满意度监控
    enable_satisfaction_monitoring: bool = True
    satisfaction_survey_frequency: int = 10  # 每10次对话


@dataclass
class ChatbotSettings:
    """聊天机器人完整配置"""

    # 各模块配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # 全局设置
    debug_mode: bool = False
    environment: str = "development"  # development, staging, production
    version: str = "1.0.0"

    @classmethod
    def load_from_env(cls) -> 'ChatbotSettings':
        """从环境变量加载配置"""
        settings = cls()

        # LLM配置
        if os.getenv("LLM_PROVIDER"):
            settings.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            settings.llm.model_name = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            settings.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))

        # 环境配置
        if os.getenv("ENVIRONMENT"):
            settings.environment = os.getenv("ENVIRONMENT")
        if os.getenv("DEBUG_MODE"):
            settings.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"

        return settings

    @classmethod
    def load_from_file(cls, config_file: str) -> 'ChatbotSettings':
        """从配置文件加载"""
        import json

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 这里可以实现更复杂的配置解析逻辑
        settings = cls()
        # ... 配置解析逻辑 ...

        return settings

    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        import json
        from dataclasses import asdict

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = asdict(self)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        # 验证LLM配置
        if not self.llm.api_key and self.llm.provider in ["openai", "anthropic"]:
            errors.append(f"LLM提供商 {self.llm.provider} 需要API密钥")

        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append("temperature必须在0-2之间")

        # 验证记忆配置
        if self.memory.short_term_size <= 0:
            errors.append("短期记忆大小必须大于0")

        # 验证质量配置
        if not (0 <= self.quality.min_confidence_threshold <= 1):
            errors.append("置信度阈值必须在0-1之间")

        return errors

    def get_runtime_config(self) -> Dict[str, Any]:
        """获取运行时配置字典"""
        return {
            "llm_provider": self.llm.provider,
            "model_name": self.llm.model_name,
            "enable_tools": self.tools.enable_tools,
            "enable_memory": self.memory.enable_long_term,
            "debug_mode": self.debug_mode,
            "environment": self.environment
        }


# 预设配置模板
class ConfigTemplates:
    """配置模板"""

    @staticmethod
    def development_config() -> ChatbotSettings:
        """开发环境配置"""
        settings = ChatbotSettings()
        settings.environment = "development"
        settings.debug_mode = True
        settings.llm.temperature = 0.9
        settings.logging.log_level = "DEBUG"
        settings.security.enable_rate_limiting = False
        return settings

    @staticmethod
    def production_config() -> ChatbotSettings:
        """生产环境配置"""
        settings = ChatbotSettings()
        settings.environment = "production"
        settings.debug_mode = False
        settings.llm.temperature = 0.7
        settings.logging.log_level = "INFO"
        settings.security.enable_rate_limiting = True
        settings.quality.enable_quality_scoring = True
        return settings

    @staticmethod
    def minimal_config() -> ChatbotSettings:
        """最小配置"""
        settings = ChatbotSettings()
        settings.tools.enable_tools = False
        settings.memory.enable_long_term = False
        settings.conversation.enable_emotion_analysis = False
        settings.escalation.enable_auto_escalation = False
        return settings


if __name__ == "__main__":
    # 测试配置
    print("=== 聊天机器人配置测试 ===")

    # 创建默认配置
    settings = ChatbotSettings()
    print(f"默认配置创建成功")
    print(f"LLM提供商: {settings.llm.provider}")
    print(f"环境: {settings.environment}")

    # 验证配置
    errors = settings.validate()
    if errors:
        print(f"配置错误: {errors}")
    else:
        print("配置验证通过")

    # 从环境变量加载
    env_settings = ChatbotSettings.load_from_env()
    print(f"环境变量配置加载完成")

    # 获取运行时配置
    runtime_config = settings.get_runtime_config()
    print(f"运行时配置: {runtime_config}")

    print("\n✅ 配置系统测试完成！")