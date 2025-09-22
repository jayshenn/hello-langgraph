"""
RAG应用系统 - LangGraph完整实现

这个系统展示了如何构建检索增强生成（RAG）应用：
- 文档向量化和存储
- 智能检索和排序
- 上下文增强生成
- 引用追踪和质量控制

适合学习现代AI问答系统的核心技术。
"""

import os
import asyncio
import datetime
import hashlib
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# LangChain相关
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    source: str
    similarity_score: float
    metadata: Dict[str, Any]
    chunk_id: str


@dataclass
class RAGAnswer:
    """RAG回答"""
    content: str
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    query_analysis: Dict[str, Any]
    generation_metadata: Dict[str, Any]


class RAGState(TypedDict):
    """RAG系统状态"""
    # 用户输入
    user_query: str
    processed_query: str

    # 查询分析
    query_type: str
    query_intent: str
    query_complexity: float
    keywords: List[str]

    # 检索结果
    retrieved_documents: List[RetrievalResult]
    reranked_documents: List[RetrievalResult]
    selected_context: str

    # 生成结果
    generated_answer: str
    answer_confidence: float
    citations: List[str]

    # 系统状态
    step_count: int
    processing_time: float
    error_message: str
    quality_scores: Dict[str, float]


# =============================================================================
# 文档处理和向量化
# =============================================================================

class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        """初始化文档处理器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """从目录加载文档"""
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            print(f"⚠️ 目录不存在: {directory_path}")
            return documents

        # 支持的文件格式
        supported_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css'}

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 创建文档对象
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": file_path.suffix,
                            "file_size": len(content),
                            "last_modified": datetime.datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat()
                        }
                    )
                    documents.append(doc)

                except Exception as e:
                    print(f"❌ 读取文件失败 {file_path}: {e}")

        print(f"✅ 成功加载 {len(documents)} 个文档")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """文档分块"""
        chunks = []

        for doc in documents:
            # 使用文本分割器进行分块
            doc_chunks = self.text_splitter.split_documents([doc])

            # 为每个块添加唯一ID
            for i, chunk in enumerate(doc_chunks):
                chunk_id = hashlib.md5(
                    f"{doc.metadata['source']}_{i}_{chunk.page_content[:100]}".encode()
                ).hexdigest()

                chunk.metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "parent_doc": doc.metadata["source"]
                })

                chunks.append(chunk)

        print(f"✅ 文档分块完成，共 {len(chunks)} 个块")
        return chunks

    def create_sample_documents(self) -> List[Document]:
        """创建示例文档"""
        sample_docs = [
            Document(
                page_content="""
                人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统。
                AI的主要目标是开发能够执行通常需要人类智能的任务的计算机程序，如学习、推理、感知、语言理解等。

                机器学习是人工智能的一个重要子领域，它使计算机能够在没有明确编程的情况下学习和改进。
                机器学习算法通过分析大量数据来识别模式，并使用这些模式来预测或分类新数据。

                深度学习是机器学习的一个子集，它使用人工神经网络来模拟人脑的学习过程。
                深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。
                """,
                metadata={
                    "source": "ai_basics.md",
                    "title": "人工智能基础",
                    "category": "technology",
                    "author": "AI专家"
                }
            ),
            Document(
                page_content="""
                自然语言处理（Natural Language Processing，NLP）是人工智能的一个重要分支，
                专注于使计算机能够理解、解释和生成人类语言。

                NLP的主要应用包括：
                1. 机器翻译：将一种语言的文本翻译成另一种语言
                2. 情感分析：分析文本中表达的情感倾向
                3. 文本摘要：自动生成文本的简洁摘要
                4. 问答系统：回答用户用自然语言提出的问题
                5. 语音识别：将语音转换为文本

                现代NLP系统大多基于深度学习技术，特别是Transformer架构，
                如GPT、BERT等预训练语言模型在各种NLP任务中都表现出色。
                """,
                metadata={
                    "source": "nlp_guide.md",
                    "title": "自然语言处理指南",
                    "category": "technology",
                    "author": "NLP研究员"
                }
            ),
            Document(
                page_content="""
                数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识，
                从数据中提取有价值的洞察和知识。

                数据科学的典型工作流程包括：
                1. 数据收集：从各种来源收集相关数据
                2. 数据清洗：处理缺失值、异常值和数据质量问题
                3. 探索性数据分析：通过可视化和统计分析理解数据
                4. 特征工程：创建和选择对模型有用的特征
                5. 模型构建：选择和训练适当的机器学习模型
                6. 模型评估：评估模型性能并进行调优
                7. 部署和监控：将模型部署到生产环境并持续监控

                Python和R是数据科学中最常用的编程语言，
                配合pandas、scikit-learn、matplotlib等库进行数据分析和建模。
                """,
                metadata={
                    "source": "data_science.md",
                    "title": "数据科学入门",
                    "category": "technology",
                    "author": "数据科学家"
                }
            )
        ]

        return sample_docs


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, persist_directory: str = "./data/vector_db"):
        """初始化向量存储"""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = None
        self._ensure_directory()

    def _ensure_directory(self):
        """确保向量数据库目录存在"""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def initialize_vector_store(self, documents: List[Document]):
        """初始化向量存储"""
        try:
            print("🔄 正在创建向量存储...")

            # 创建Chroma向量数据库
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="rag_collection"
            )

            # 持久化
            self.vector_store.persist()
            print(f"✅ 向量存储创建完成，包含 {len(documents)} 个文档")

        except Exception as e:
            print(f"❌ 向量存储创建失败: {e}")
            raise

    def load_existing_vector_store(self):
        """加载已存在的向量存储"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
            print("✅ 成功加载现有向量存储")
            return True
        except Exception as e:
            print(f"⚠️ 无法加载现有向量存储: {e}")
            return False

    def search_similar_documents(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """搜索相似文档"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        try:
            # 执行相似度搜索
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # 转换为RetrievalResult格式
            retrieval_results = []
            for doc, score in results:
                result = RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    similarity_score=float(score),
                    metadata=doc.metadata,
                    chunk_id=doc.metadata.get("chunk_id", "")
                )
                retrieval_results.append(result)

            return retrieval_results

        except Exception as e:
            print(f"❌ 相似度搜索失败: {e}")
            return []

    def add_documents(self, documents: List[Document]):
        """添加新文档到向量存储"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        self.vector_store.add_documents(documents)
        self.vector_store.persist()
        print(f"✅ 成功添加 {len(documents)} 个文档")

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.vector_store:
            return {"error": "向量存储未初始化"}

        try:
            collection = self.vector_store._collection
            return {
                "document_count": collection.count(),
                "collection_name": collection.name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# RAG系统节点
# =============================================================================

def query_analysis_node(state: RAGState) -> RAGState:
    """查询分析节点"""
    user_query = state["user_query"]

    # 简单的查询分析（实际应用中可以使用更复杂的NLP技术）
    query_length = len(user_query)
    word_count = len(user_query.split())

    # 查询类型分析
    query_type = "general"
    if "什么是" in user_query or "定义" in user_query:
        query_type = "definition"
    elif "如何" in user_query or "怎么" in user_query:
        query_type = "how_to"
    elif "为什么" in user_query:
        query_type = "explanation"
    elif "比较" in user_query or "区别" in user_query:
        query_type = "comparison"

    # 查询复杂度评估
    complexity = min(1.0, (word_count * 0.1 + query_length * 0.001))

    # 关键词提取（简化版）
    stop_words = {"的", "是", "在", "有", "和", "与", "或", "但", "如果", "那么"}
    keywords = [word for word in user_query.split() if word not in stop_words and len(word) > 1]

    return {
        **state,
        "processed_query": user_query.strip(),
        "query_type": query_type,
        "query_intent": "information_seeking",
        "query_complexity": complexity,
        "keywords": keywords[:10],  # 限制关键词数量
        "step_count": state["step_count"] + 1
    }


def document_retrieval_node(state: RAGState) -> RAGState:
    """文档检索节点"""
    query = state["processed_query"]

    # 获取全局向量存储管理器
    vector_manager = getattr(document_retrieval_node, 'vector_manager', None)
    if not vector_manager:
        return {
            **state,
            "error_message": "向量存储未初始化",
            "step_count": state["step_count"] + 1
        }

    try:
        # 执行检索
        k = min(10, max(3, int(state["query_complexity"] * 15)))  # 动态调整检索数量
        retrieved_docs = vector_manager.search_similar_documents(query, k=k)

        return {
            **state,
            "retrieved_documents": retrieved_docs,
            "step_count": state["step_count"] + 1
        }

    except Exception as e:
        return {
            **state,
            "error_message": f"文档检索失败: {str(e)}",
            "step_count": state["step_count"] + 1
        }


def document_reranking_node(state: RAGState) -> RAGState:
    """文档重排序节点"""
    retrieved_docs = state["retrieved_documents"]
    query = state["processed_query"]

    if not retrieved_docs:
        return {
            **state,
            "reranked_documents": [],
            "step_count": state["step_count"] + 1
        }

    # 简单的重排序策略：结合相似度分数和元数据
    def calculate_relevance_score(doc: RetrievalResult) -> float:
        base_score = 1.0 - doc.similarity_score  # 转换为相似度分数

        # 根据查询类型调整分数
        query_type = state["query_type"]
        if query_type == "definition" and "定义" in doc.content:
            base_score *= 1.2
        elif query_type == "how_to" and any(word in doc.content for word in ["方法", "步骤", "如何"]):
            base_score *= 1.15

        # 根据文档新鲜度调整（如果有时间戳）
        if "last_modified" in doc.metadata:
            # 简化的新鲜度加权
            base_score *= 1.05

        # 根据文档长度调整
        content_length = len(doc.content)
        if 200 <= content_length <= 1500:  # 适中长度的文档
            base_score *= 1.1

        return base_score

    # 重新排序
    reranked_docs = sorted(retrieved_docs, key=calculate_relevance_score, reverse=True)

    # 保留前5个最相关的文档
    top_docs = reranked_docs[:5]

    return {
        **state,
        "reranked_documents": top_docs,
        "step_count": state["step_count"] + 1
    }


def context_construction_node(state: RAGState) -> RAGState:
    """上下文构建节点"""
    reranked_docs = state["reranked_documents"]

    if not reranked_docs:
        return {
            **state,
            "selected_context": "没有找到相关文档。",
            "step_count": state["step_count"] + 1
        }

    # 构建上下文
    context_parts = []
    total_length = 0
    max_context_length = 3000  # 控制上下文长度

    for i, doc in enumerate(reranked_docs, 1):
        doc_text = doc.content.strip()
        if total_length + len(doc_text) > max_context_length:
            break

        context_part = f"文档 {i} (来源: {doc.source}):\n{doc_text}\n"
        context_parts.append(context_part)
        total_length += len(context_part)

    selected_context = "\n".join(context_parts)

    return {
        **state,
        "selected_context": selected_context,
        "step_count": state["step_count"] + 1
    }


def answer_generation_node(state: RAGState) -> RAGState:
    """答案生成节点"""
    query = state["processed_query"]
    context = state["selected_context"]
    reranked_docs = state["reranked_documents"]

    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,  # 较低的温度确保答案更准确
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 构建提示词
    system_prompt = f"""你是一个专业的知识问答助手。请基于提供的文档内容回答用户问题。

要求：
1. 答案必须基于提供的文档内容，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 引用具体的文档来源
4. 保持答案准确、简洁、有用
5. 使用中文回答

查询类型：{state['query_type']}
查询意图：{state['query_intent']}

参考文档：
{context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"问题：{query}")
    ]

    try:
        # 生成答案
        response = llm.invoke(messages)
        generated_answer = response.content

        # 生成引用
        citations = []
        for doc in reranked_docs:
            citation = f"{doc.source} (相似度: {1-doc.similarity_score:.3f})"
            citations.append(citation)

        # 计算置信度（简化版）
        confidence = 0.8  # 基础置信度
        if len(reranked_docs) >= 3:
            confidence += 0.1
        if state["query_complexity"] <= 0.5:
            confidence += 0.1
        confidence = min(1.0, confidence)

        return {
            **state,
            "generated_answer": generated_answer,
            "answer_confidence": confidence,
            "citations": citations,
            "step_count": state["step_count"] + 1
        }

    except Exception as e:
        return {
            **state,
            "error_message": f"答案生成失败: {str(e)}",
            "generated_answer": "抱歉，无法生成答案，请稍后重试。",
            "answer_confidence": 0.0,
            "citations": [],
            "step_count": state["step_count"] + 1
        }


def quality_assessment_node(state: RAGState) -> RAGState:
    """质量评估节点"""
    answer = state["generated_answer"]
    retrieved_docs = state["retrieved_documents"]
    query = state["processed_query"]

    # 计算各种质量指标
    quality_scores = {}

    # 1. 答案长度适中性
    answer_length = len(answer)
    if 50 <= answer_length <= 500:
        quality_scores["length_appropriateness"] = 1.0
    elif answer_length < 50:
        quality_scores["length_appropriateness"] = 0.6
    else:
        quality_scores["length_appropriateness"] = 0.8

    # 2. 检索覆盖度
    if retrieved_docs:
        avg_similarity = sum(1 - doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
        quality_scores["retrieval_quality"] = avg_similarity
    else:
        quality_scores["retrieval_quality"] = 0.0

    # 3. 答案完整性（简化评估）
    if "抱歉" in answer or "无法" in answer or "不知道" in answer:
        quality_scores["completeness"] = 0.3
    elif len(answer.split("。")) >= 2:  # 至少两个句子
        quality_scores["completeness"] = 0.9
    else:
        quality_scores["completeness"] = 0.7

    # 4. 引用质量
    citations = state.get("citations", [])
    if citations and len(citations) > 0:
        quality_scores["citation_quality"] = min(1.0, len(citations) * 0.3)
    else:
        quality_scores["citation_quality"] = 0.0

    # 综合质量分数
    overall_quality = sum(quality_scores.values()) / len(quality_scores)

    return {
        **state,
        "quality_scores": quality_scores,
        "step_count": state["step_count"] + 1
    }


# =============================================================================
# RAG系统图构建
# =============================================================================

def create_rag_graph() -> StateGraph:
    """创建RAG系统图"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("analyze_query", query_analysis_node)
    graph.add_node("retrieve_documents", document_retrieval_node)
    graph.add_node("rerank_documents", document_reranking_node)
    graph.add_node("construct_context", context_construction_node)
    graph.add_node("generate_answer", answer_generation_node)
    graph.add_node("assess_quality", quality_assessment_node)

    # 设置入口点
    graph.set_entry_point("analyze_query")

    # 添加边
    graph.add_edge("analyze_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "rerank_documents")
    graph.add_edge("rerank_documents", "construct_context")
    graph.add_edge("construct_context", "generate_answer")
    graph.add_edge("generate_answer", "assess_quality")
    graph.add_edge("assess_quality", END)

    return graph


# =============================================================================
# RAG系统主类
# =============================================================================

class RAGSystem:
    """RAG系统主类"""

    def __init__(self, vector_db_path: str = "./data/vector_db"):
        """初始化RAG系统"""
        self.vector_manager = VectorStoreManager(vector_db_path)
        self.document_processor = DocumentProcessor()
        self.graph = create_rag_graph()
        self.memory_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory_saver)

        # 设置全局向量管理器
        document_retrieval_node.vector_manager = self.vector_manager

        print("🤖 RAG系统已启动")

    def initialize_knowledge_base(self, documents_path: str = None):
        """初始化知识库"""
        print("📚 正在初始化知识库...")

        # 加载文档
        if documents_path and Path(documents_path).exists():
            documents = self.document_processor.load_documents_from_directory(documents_path)
        else:
            print("📝 使用示例文档初始化知识库")
            documents = self.document_processor.create_sample_documents()

        if not documents:
            print("❌ 没有找到文档")
            return False

        # 文档分块
        chunks = self.document_processor.chunk_documents(documents)

        # 创建向量存储
        self.vector_manager.initialize_vector_store(chunks)

        print("✅ 知识库初始化完成")
        return True

    def load_existing_knowledge_base(self) -> bool:
        """加载现有知识库"""
        return self.vector_manager.load_existing_vector_store()

    def create_initial_state(self, user_query: str) -> RAGState:
        """创建初始状态"""
        return {
            "user_query": user_query,
            "processed_query": "",
            "query_type": "",
            "query_intent": "",
            "query_complexity": 0.0,
            "keywords": [],
            "retrieved_documents": [],
            "reranked_documents": [],
            "selected_context": "",
            "generated_answer": "",
            "answer_confidence": 0.0,
            "citations": [],
            "step_count": 0,
            "processing_time": 0.0,
            "error_message": "",
            "quality_scores": {}
        }

    async def ask_async(self, question: str) -> RAGAnswer:
        """异步问答"""
        if not question.strip():
            return RAGAnswer(
                content="请输入有效的问题。",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                query_analysis={},
                generation_metadata={}
            )

        # 创建初始状态
        initial_state = self.create_initial_state(question)

        # 配置
        config = {"configurable": {"thread_id": f"rag_session_{hash(question)}"}}

        try:
            start_time = datetime.datetime.now()

            # 执行RAG流程
            result = await self.app.ainvoke(initial_state, config)

            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            # 构建回答对象
            answer = RAGAnswer(
                content=result["generated_answer"],
                sources=result["reranked_documents"],
                confidence=result["answer_confidence"],
                processing_time=processing_time,
                query_analysis={
                    "query_type": result["query_type"],
                    "query_intent": result["query_intent"],
                    "complexity": result["query_complexity"],
                    "keywords": result["keywords"]
                },
                generation_metadata={
                    "quality_scores": result["quality_scores"],
                    "citations": result["citations"],
                    "step_count": result["step_count"]
                }
            )

            return answer

        except Exception as e:
            return RAGAnswer(
                content=f"处理问题时发生错误: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                query_analysis={},
                generation_metadata={"error": str(e)}
            )

    def ask(self, question: str) -> RAGAnswer:
        """同步问答"""
        return asyncio.run(self.ask_async(question))

    def add_documents_to_knowledge_base(self, documents_path: str):
        """向知识库添加新文档"""
        documents = self.document_processor.load_documents_from_directory(documents_path)
        chunks = self.document_processor.chunk_documents(documents)
        self.vector_manager.add_documents(chunks)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_manager.get_collection_stats()


# =============================================================================
# 演示和测试
# =============================================================================

async def demo_basic_qa():
    """演示基础问答功能"""
    print("=== 基础问答演示 ===\n")

    # 初始化RAG系统
    rag = RAGSystem()

    # 尝试加载现有知识库，否则初始化新的
    if not rag.load_existing_knowledge_base():
        rag.initialize_knowledge_base()

    # 测试问题
    questions = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理的主要应用有哪些？",
        "数据科学的工作流程是什么？",
        "Python在数据科学中的作用是什么？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"问题 {i}: {question}")
        answer = await rag.ask_async(question)

        print(f"回答: {answer.content}")
        print(f"置信度: {answer.confidence:.2f}")
        print(f"处理时间: {answer.processing_time:.2f}秒")

        if answer.sources:
            print("参考来源:")
            for j, source in enumerate(answer.sources[:3], 1):
                print(f"  {j}. {source.source} (相似度: {1-source.similarity_score:.3f})")

        print("-" * 60)


def demo_knowledge_base_management():
    """演示知识库管理功能"""
    print("=== 知识库管理演示 ===\n")

    rag = RAGSystem()

    # 初始化知识库
    print("1. 初始化知识库:")
    rag.initialize_knowledge_base()

    # 获取统计信息
    print("\n2. 知识库统计信息:")
    stats = rag.get_knowledge_base_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 测试搜索
    print("\n3. 测试检索功能:")
    test_query = "人工智能"
    results = rag.vector_manager.search_similar_documents(test_query, k=3)

    print(f"查询: {test_query}")
    for i, result in enumerate(results, 1):
        print(f"   结果 {i}: {result.source} (相似度: {1-result.similarity_score:.3f})")
        print(f"   内容: {result.content[:100]}...")


def interactive_demo():
    """交互式演示"""
    print("🤖 RAG智能问答系统")
    print("=" * 50)
    print("输入 'quit' 退出，'stats' 查看统计信息\n")

    # 初始化系统
    rag = RAGSystem()

    # 加载或创建知识库
    if not rag.load_existing_knowledge_base():
        print("📚 首次运行，正在初始化知识库...")
        rag.initialize_knowledge_base()

    while True:
        try:
            user_input = input("🔍 请输入问题: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break

            if user_input.lower() == 'stats':
                stats = rag.get_knowledge_base_stats()
                print("\n📊 知识库统计:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                print()
                continue

            if not user_input:
                print("请输入有效问题。")
                continue

            print("🤖 正在思考...")

            # 获取答案
            answer = rag.ask(user_input)

            # 显示结果
            print(f"\n📝 回答: {answer.content}")
            print(f"🎯 置信度: {answer.confidence:.2f}")
            print(f"⏱️ 处理时间: {answer.processing_time:.2f}秒")

            if answer.sources:
                print("\n📚 参考来源:")
                for i, source in enumerate(answer.sources[:3], 1):
                    print(f"   {i}. {source.source} (相似度: {1-source.similarity_score:.3f})")

            # 显示质量分数
            if answer.generation_metadata.get("quality_scores"):
                print(f"\n📈 质量评分:")
                for metric, score in answer.generation_metadata["quality_scores"].items():
                    print(f"   {metric}: {score:.2f}")

            print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 对话结束！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


# =============================================================================
# 主程序
# =============================================================================

async def main():
    """主程序"""
    print("🚀 RAG应用系统 - LangGraph完整示例")
    print("=" * 50)

    while True:
        print("\n选择演示模式:")
        print("1. 基础问答演示")
        print("2. 知识库管理演示")
        print("3. 交互式问答")
        print("4. 退出")

        choice = input("\n请选择 (1-4): ").strip()

        if choice == "1":
            await demo_basic_qa()

        elif choice == "2":
            demo_knowledge_base_management()

        elif choice == "3":
            interactive_demo()

        elif choice == "4":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选择，请重试。")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# 学习指南和扩展建议
# =============================================================================

"""
🎯 RAG系统学习要点:

1. **文档处理流程**:
   - 多格式文档加载和解析
   - 智能文档分块策略
   - 元数据提取和管理

2. **向量化技术**:
   - 文本嵌入模型的选择和使用
   - 向量数据库的管理和优化
   - 相似度计算和检索策略

3. **检索优化**:
   - 查询理解和重写
   - 多阶段检索和重排序
   - 检索结果的质量评估

4. **生成增强**:
   - 上下文构建和管理
   - 提示词工程技巧
   - 生成质量控制

5. **系统集成**:
   - 端到端的RAG流程设计
   - 错误处理和容错机制
   - 性能监控和优化

🔧 扩展建议:

1. **高级检索技术**:
   - 混合检索（Dense + Sparse）
   - 多跳检索和推理
   - 动态检索策略

2. **多模态RAG**:
   - 图片和表格的处理
   - 多模态嵌入技术
   - 跨模态检索

3. **实时更新**:
   - 增量文档更新
   - 在线学习和适应
   - 缓存策略优化

4. **评估和监控**:
   - 自动化评估指标
   - A/B测试框架
   - 用户反馈集成

这个RAG系统为构建智能问答应用提供了完整的技术基础！
"""