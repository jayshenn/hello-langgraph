# 05-工具与集成 示例代码

本目录包含了 LangGraph 工具与集成的完整示例代码，展示了从基础工具定义到复杂系统集成的各种实现方法。

## 📁 文件结构

```
示例代码/
├── basic_tools.py           # 基础工具定义示例
├── llm_integration.py       # LLM 集成示例
├── rag_system.py           # RAG 系统实现
├── api_tools.py            # API 工具封装
├── mcp_integration.py      # MCP 集成示例
└── README.md               # 本文件
```

## 🛠️ basic_tools.py

演示了基础工具的定义和使用方法：

### 主要内容
- **函数工具**: 使用 `@tool` 装饰器创建简单工具
- **类工具**: 继承 `BaseTool` 创建复杂工具
- **参数验证**: 使用 Pydantic 模型进行输入验证
- **错误处理**: 实现重试机制和异常处理
- **状态管理**: 创建能够访问和修改状态的工具

### 示例工具
1. `simple_calculator` - 数学计算器
2. `text_analyzer` - 文本分析工具
3. `random_data_generator` - 随机数据生成器
4. `WeatherSimulatorTool` - 天气模拟器（BaseTool 示例）
5. `ReliableAPITool` - 带重试机制的 API 工具

### 运行方式
```bash
python basic_tools.py
```

## 🤖 llm_integration.py

展示了 LLM 模型的集成和配置：

### 主要功能
- **多模型支持**: OpenAI、Anthropic、Azure、Google 等
- **工具绑定**: Tool Calling 机制实现
- **流式输出**: 实时响应流处理
- **模型降级**: Fallback 机制实现
- **Token 管理**: 成本控制和使用追踪

### 核心组件
1. `ModelFactory` - 模型创建工厂
2. `ModelManager` - 模型管理器
3. `ConversationManager` - 对话管理
4. `TokenManager` - Token 使用管理

### 运行方式
```bash
python llm_integration.py
```

## 🔍 rag_system.py

完整的 RAG（检索增强生成）系统实现：

### 系统层次
1. **基础 RAG**: 文档检索 + LLM 生成
2. **高级 RAG**: 多策略检索、重排序、缓存
3. **智能 RAG**: 查询分析、迭代检索、对话感知

### 核心组件
- `DocumentProcessor` - 文档处理和分块
- `BasicRAGSystem` - 基础 RAG 实现
- `AdvancedRAGSystem` - 高级特性 RAG
- `AgenticRAGSystem` - 智能 RAG 系统
- `RAGEvaluator` - 系统性能评估

### 运行方式
```bash
python rag_system.py
```

## 🌐 api_tools.py

API 调用工具的封装和集成：

### 主要特性
- **通用 HTTP 工具**: 支持各种 REST API
- **认证管理**: 安全的密钥管理
- **错误处理**: 重试和降级机制
- **性能优化**: 缓存、并发控制、速率限制
- **批量处理**: 高效的批量 API 调用

### 核心工具
1. `HTTPAPITool` - 通用 HTTP API 工具
2. `SecureAPITool` - 带认证的安全 API 工具
3. `SmartAPITool` - 智能 API 工具（速率限制+并发控制）
4. `CachedAPITool` - 带缓存的 API 工具

### 运行方式
```bash
python api_tools.py
```

## 🔌 mcp_integration.py

MCP（Model Context Protocol）协议集成：

### 主要功能
- **MCP 服务器**: 自定义 MCP 服务器实现
- **客户端集成**: LangGraph 与 MCP 的集成
- **资源管理**: 文件系统、数据库等资源访问
- **工具调用**: 标准化的工具调用接口
- **安全控制**: 权限管理和访问控制

### 核心组件
1. `BaseMCPServer` - MCP 服务器基类
2. `FileSystemMCPServer` - 文件系统服务器
3. `DatabaseMCPServer` - 数据库服务器
4. `MCPAgent` - 集成 MCP 的 LangGraph Agent

### 运行方式
```bash
python mcp_integration.py
```

## 🚀 快速开始

### 环境准备

1. **基础环境**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装基础依赖
pip install pydantic typing-extensions
```

2. **完整环境（可选）**
```bash
# 安装 LangChain 相关包
pip install langchain langchain-core langchain-community
pip install langchain-openai langchain-anthropic
pip install chromadb tiktoken

# 安装其他依赖
pip install aiohttp requests cryptography
```

### 运行示例

1. **运行单个示例**
```bash
# 基础工具示例
python basic_tools.py

# LLM 集成示例
python llm_integration.py

# RAG 系统示例
python rag_system.py
```

2. **运行所有示例**
```bash
# 批量运行
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

## 📖 学习路径

### 初学者路径
1. 先运行 `basic_tools.py` 了解工具基础
2. 学习 `llm_integration.py` 掌握模型集成
3. 探索 `rag_system.py` 理解检索增强

### 进阶路径
1. 深入研究 `api_tools.py` 的高级特性
2. 学习 `mcp_integration.py` 的协议集成
3. 结合实际项目进行实践

### 实战建议
1. **修改配置**: 尝试不同的参数配置
2. **扩展功能**: 在示例基础上添加新功能
3. **集成测试**: 将多个组件组合使用
4. **性能优化**: 测试和优化系统性能

## 🔧 自定义和扩展

### 工具扩展
```python
# 创建自定义工具
@tool
def my_custom_tool(input_data: str) -> str:
    """我的自定义工具"""
    # 实现你的逻辑
    return f"处理结果: {input_data}"

# 添加到工具集合
toolkit = ToolCollectionManager()
toolkit.tools.append(my_custom_tool)
```

### 模型扩展
```python
# 添加新的模型提供商
class MyModelProvider:
    @staticmethod
    def create_model(model_name: str, **kwargs):
        # 实现模型创建逻辑
        return MyCustomModel(model_name, **kwargs)

# 注册到模型工厂
ModelFactory.register_provider("my_provider", MyModelProvider)
```

### RAG 扩展
```python
# 创建自定义检索策略
class MyRAGSystem(AdvancedRAGSystem):
    def my_custom_retrieval(self, query: str) -> List[Document]:
        # 实现自定义检索逻辑
        return self.vectorstore.similarity_search(query)
```

## 🐛 常见问题

### Q: 运行时提示模块未找到？
A: 确保安装了相应的依赖包，或者使用示例中的模拟实现。

### Q: API 调用失败？
A: 检查网络连接和 API 密钥配置，确保密钥有效且有足够的配额。

### Q: 向量化速度很慢？
A: 可以使用更快的嵌入模型，或者调整文档分块大小。

### Q: 内存占用过高？
A: 减少缓存大小，或者使用流式处理来降低内存使用。

## 📚 相关文档

- [工具定义与使用](../01-Tool定义与使用.md)
- [LLM集成](../02-LLM集成.md)
- [RAG实现](../03-RAG实现.md)
- [API调用](../04-API调用.md)
- [MCP协议](../05-MCP协议.md)

## 🤝 贡献指南

1. **Bug 报告**: 如果发现问题，请在项目中提出 Issue
2. **功能建议**: 欢迎提出改进建议和新功能需求
3. **代码贡献**: 遵循代码风格，添加必要的测试和文档

## 📄 许可证

本示例代码遵循项目整体许可证，仅用于学习和研究目的。

---

希望这些示例能帮助你快速掌握 LangGraph 的工具与集成技术！如有问题，请参考相关文档或提出 Issue。