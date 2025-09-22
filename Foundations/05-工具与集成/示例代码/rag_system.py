#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 系统实现示例

本文件演示了完整的 RAG（检索增强生成）系统实现：
1. 文档处理和向量化
2. 检索器配置和优化
3. 基础 RAG 实现
4. 高级 RAG 技术
5. Agentic RAG 系统

运行方式:
    python rag_system.py
"""

import os
import json
import time
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import deque, defaultdict

# 模拟 LangChain 组件（实际使用时请安装相应包）
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告: LangChain 组件未安装，使用模拟实现")


# ================================
# 1. 模拟组件（当 LangChain 不可用时）
# ================================

class MockDocument:
    """模拟文档类"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockEmbeddings:
    """模拟嵌入模型"""
    def __init__(self):
        self.call_count = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        self.call_count += len(texts)
        # 生成模拟向量（基于文本哈希）
        embeddings = []
        for text in texts:
            # 简单的文本向量化（实际应用中不要这样做）
            hash_obj = hashlib.md5(text.encode())
            vector = [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
            embeddings.append(vector)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self.embed_documents([text])[0]


class MockVectorStore:
    """模拟向量存储"""
    def __init__(self, documents: List[MockDocument] = None, embeddings=None):
        self.documents = documents or []
        self.embeddings = embeddings or MockEmbeddings()
        self.vectors = []
        self.doc_embeddings = {}

        if documents:
            self._build_index()

    def _build_index(self):
        """构建索引"""
        texts = [doc.page_content for doc in self.documents]
        self.vectors = self.embeddings.embed_documents(texts)

        for i, doc in enumerate(self.documents):
            self.doc_embeddings[i] = self.vectors[i]

    def similarity_search(self, query: str, k: int = 4) -> List[MockDocument]:
        """相似性搜索"""
        if not self.documents:
            return []

        query_vector = self.embeddings.embed_query(query)

        # 计算相似度
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((similarity, i))

        # 排序并返回前k个
        similarities.sort(reverse=True)
        results = []
        for _, doc_idx in similarities[:k]:
            results.append(self.documents[doc_idx])

        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

    def add_documents(self, documents: List[MockDocument]):
        """添加文档"""
        self.documents.extend(documents)
        self._build_index()

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        """创建检索器"""
        search_kwargs = search_kwargs or {}
        return MockRetriever(self, **search_kwargs)


class MockRetriever:
    """模拟检索器"""
    def __init__(self, vectorstore: MockVectorStore, k: int = 4):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query: str) -> List[MockDocument]:
        """检索文档"""
        return self.vectorstore.similarity_search(query, k=self.k)


class MockLLM:
    """模拟大语言模型"""
    def __init__(self, model: str = "mock-gpt"):
        self.model = model

    def invoke(self, messages: List) -> Any:
        """调用模型"""
        # 模拟响应
        if isinstance(messages, list) and messages:
            content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        else:
            content = str(messages)

        # 生成基于内容的响应
        if "什么是" in content or "介绍" in content:
            return type('Response', (), {'content': f"基于提供的上下文，{content.replace('什么是', '').replace('介绍', '')}是一个重要的概念..."})()
        elif "如何" in content:
            return type('Response', (), {'content': f"要{content.replace('如何', '')}，您可以按照以下步骤进行..."})()
        else:
            return type('Response', (), {'content': f"根据相关信息，{content}的答案是..."})()


# ================================
# 2. 文档处理系统
# ================================

class DocumentProcessor:
    """文档处理器"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_docs = []

    def create_sample_documents(self) -> List[MockDocument]:
        """创建示例文档"""
        documents = [
            MockDocument(
                page_content="""
                LangGraph 是一个用于构建有状态、多参与者应用程序的强大库。它基于 LangChain 表达式语言，
                提供了定义包含循环和条件的复杂图结构的能力。LangGraph 特别适合构建需要循环执行、
                状态管理和复杂决策流程的 AI 应用程序。其核心特性包括状态管理、图形工作流、
                条件执行和持久化能力。
                """,
                metadata={"source": "langgraph_intro", "category": "introduction", "length": "medium"}
            ),
            MockDocument(
                page_content="""
                在 LangGraph 中，状态(State)是通过 TypedDict 定义的核心概念。状态代表在图的节点之间
                传递的所有信息。每个节点函数接收当前状态作为输入，可以读取、修改状态，并返回更新后的状态。
                状态管理确保了数据在整个工作流中的一致性和可访问性。状态可以包含消息历史、用户信息、
                处理结果等各种数据类型。
                """,
                metadata={"source": "langgraph_state", "category": "core_concept", "length": "medium"}
            ),
            MockDocument(
                page_content="""
                工具(Tools)是 LangGraph 中 Agent 与外部世界交互的重要机制。工具可以是函数、API调用、
                数据库查询或任何其他操作。通过 @tool 装饰器或继承 BaseTool 类，开发者可以创建自定义工具。
                Agent 根据用户输入和当前上下文决定是否需要调用工具以及调用哪个工具。工具的正确设计和
                使用是构建强大 AI 应用的关键。
                """,
                metadata={"source": "langgraph_tools", "category": "core_concept", "length": "medium"}
            ),
            MockDocument(
                page_content="""
                条件路由(Conditional Routing)是 LangGraph 的高级特性，允许根据当前状态或处理结果
                动态决定下一步的执行路径。通过 add_conditional_edges 方法，可以定义复杂的条件逻辑。
                条件函数接收当前状态并返回字符串键，该键决定了下一个要执行的节点。这种机制使得
                图可以根据运行时的情况进行动态调整，实现智能的分支执行。
                """,
                metadata={"source": "langgraph_routing", "category": "advanced", "length": "medium"}
            ),
            MockDocument(
                page_content="""
                Human-in-the-Loop (人在回路中)是 LangGraph 提供的重要功能，允许在自动化流程中
                适时引入人工干预。通过 interrupt 函数，可以在关键决策点暂停执行，等待人工输入
                或确认。这种机制在需要人工审核、决策或提供额外信息的场景中非常有用。
                HIL 功能增强了系统的可控性和可靠性。
                """,
                metadata={"source": "langgraph_hil", "category": "advanced", "length": "medium"}
            ),
            MockDocument(
                page_content="""
                RAG(检索增强生成)是一种将信息检索与文本生成相结合的技术。它允许 AI 模型在生成回答时
                从外部知识库中检索相关信息，从而提供更准确、更及时的答案。RAG 系统通常包括文档预处理、
                向量化、检索和生成四个主要组件。这种方法解决了大语言模型的知识截止时间限制和
                可能出现的幻觉问题。
                """,
                metadata={"source": "rag_concept", "category": "technology", "length": "medium"}
            )
        ]

        return documents

    def split_documents(self, documents: List[MockDocument]) -> List[MockDocument]:
        """分割文档为较小的块"""
        chunks = []

        for doc in documents:
            content = doc.page_content.strip()

            # 简单的文本分割（实际中应使用更复杂的分割策略）
            if len(content) <= self.chunk_size:
                chunks.append(MockDocument(
                    page_content=content,
                    metadata={**doc.metadata, "chunk_id": 0, "chunk_size": len(content)}
                ))
            else:
                # 分割长文档
                start = 0
                chunk_id = 0
                while start < len(content):
                    end = min(start + self.chunk_size, len(content))

                    # 尝试在单词边界分割
                    if end < len(content):
                        while end > start and content[end] not in [' ', '\n', '。', '，']:
                            end -= 1
                        if end == start:  # 如果找不到合适的分割点
                            end = min(start + self.chunk_size, len(content))

                    chunk_content = content[start:end].strip()
                    if chunk_content:
                        chunks.append(MockDocument(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": chunk_id,
                                "chunk_size": len(chunk_content),
                                "original_length": len(content)
                            }
                        ))
                        chunk_id += 1

                    start = max(end - self.chunk_overlap, start + 1)

        self.processed_docs = chunks
        return chunks

    def enhance_metadata(self, documents: List[MockDocument]) -> List[MockDocument]:
        """增强文档元数据"""
        enhanced_docs = []

        for doc in documents:
            content = doc.page_content
            enhanced_metadata = {
                **doc.metadata,
                "word_count": len(content.split()),
                "char_count": len(content),
                "has_chinese": any('\u4e00' <= c <= '\u9fff' for c in content),
                "has_technical_terms": any(term in content.lower() for term in [
                    'langgraph', 'rag', 'ai', 'agent', 'llm', 'api'
                ]),
                "processing_timestamp": time.time()
            }

            enhanced_docs.append(MockDocument(
                page_content=content,
                metadata=enhanced_metadata
            ))

        return enhanced_docs


# ================================
# 3. 基础 RAG 系统
# ================================

class BasicRAGSystem:
    """基础 RAG 系统"""

    def __init__(self, documents: List[MockDocument] = None):
        self.documents = documents or []
        self.vectorstore = None
        self.retriever = None
        self.llm = MockLLM("rag-gpt")
        self.embeddings = MockEmbeddings()

        if documents:
            self._build_vectorstore()

    def _build_vectorstore(self):
        """构建向量存储"""
        print(f"构建向量存储，文档数量: {len(self.documents)}")

        self.vectorstore = MockVectorStore(self.documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        print("向量存储构建完成")

    def add_documents(self, documents: List[MockDocument]):
        """添加新文档"""
        self.documents.extend(documents)
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
        else:
            self._build_vectorstore()

    def retrieve(self, query: str, k: int = 3) -> List[MockDocument]:
        """检索相关文档"""
        if not self.retriever:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"检索失败: {e}")
            return []

    def generate_answer(self, query: str, context_docs: List[MockDocument]) -> str:
        """基于上下文生成答案"""
        if not context_docs:
            return "抱歉，我没有找到相关信息。"

        # 构建上下文
        context = "\n\n".join([
            f"文档{i+1}: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        # 构建提示词
        prompt = f"""基于以下上下文信息回答问题：

上下文：
{context}

问题：{query}

请基于上下文提供准确、详细的回答。如果上下文中没有相关信息，请明确说明。

回答："""

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content
        except Exception as e:
            return f"生成回答失败: {str(e)}"

    def query(self, question: str) -> Dict[str, Any]:
        """完整的 RAG 查询流程"""
        start_time = time.time()

        # 1. 检索相关文档
        relevant_docs = self.retrieve(question)

        # 2. 生成答案
        answer = self.generate_answer(question, relevant_docs)

        # 3. 准备响应
        response = {
            "question": question,
            "answer": answer,
            "retrieved_docs": len(relevant_docs),
            "sources": [
                {
                    "content": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ],
            "processing_time": time.time() - start_time
        }

        return response


# ================================
# 4. 高级 RAG 系统
# ================================

class AdvancedRAGSystem(BasicRAGSystem):
    """高级 RAG 系统"""

    def __init__(self, documents: List[MockDocument] = None):
        super().__init__(documents)
        self.query_cache = {}
        self.retrieval_strategies = {
            "similarity": self._similarity_search,
            "mmr": self._mmr_search,
            "threshold": self._threshold_search
        }

    def _similarity_search(self, query: str, k: int = 3) -> List[MockDocument]:
        """相似性搜索"""
        return self.vectorstore.similarity_search(query, k=k)

    def _mmr_search(self, query: str, k: int = 3) -> List[MockDocument]:
        """最大边际相关性搜索"""
        # 简化的 MMR 实现
        initial_docs = self.vectorstore.similarity_search(query, k=k*2)

        if not initial_docs:
            return []

        # 选择多样化的文档
        selected_docs = [initial_docs[0]]  # 最相关的文档
        remaining_docs = initial_docs[1:]

        while len(selected_docs) < k and remaining_docs:
            # 简单的多样性选择：选择与已选文档差异较大的
            best_doc = None
            max_diversity = -1

            for doc in remaining_docs:
                diversity = self._calculate_diversity(doc, selected_docs)
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_doc = doc

            if best_doc:
                selected_docs.append(best_doc)
                remaining_docs.remove(best_doc)

        return selected_docs

    def _threshold_search(self, query: str, threshold: float = 0.7) -> List[MockDocument]:
        """阈值搜索"""
        # 简化实现：返回相似度高于阈值的文档
        docs = self.vectorstore.similarity_search(query, k=10)
        # 在实际实现中，这里应该基于实际的相似度分数过滤
        return docs[:3]  # 简化处理

    def _calculate_diversity(self, doc: MockDocument, selected_docs: List[MockDocument]) -> float:
        """计算文档多样性"""
        # 简化的多样性计算
        doc_words = set(doc.page_content.lower().split())

        total_overlap = 0
        for selected_doc in selected_docs:
            selected_words = set(selected_doc.page_content.lower().split())
            overlap = len(doc_words.intersection(selected_words))
            total_overlap += overlap

        # 重叠越少，多样性越高
        return 1.0 / (1.0 + total_overlap / len(doc_words))

    def enhanced_retrieve(self, query: str, strategy: str = "similarity", **kwargs) -> List[MockDocument]:
        """增强检索"""
        if strategy not in self.retrieval_strategies:
            strategy = "similarity"

        retrieval_func = self.retrieval_strategies[strategy]
        return retrieval_func(query, **kwargs)

    def rerank_documents(self, query: str, documents: List[MockDocument]) -> List[MockDocument]:
        """重新排序文档"""
        if not documents:
            return documents

        # 简单的重排序：基于查询词匹配
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            match_score = len(query_words.intersection(doc_words))
            length_penalty = len(doc.page_content) / 1000  # 偏好较短的文档
            final_score = match_score - length_penalty
            scored_docs.append((final_score, doc))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]

    def query_with_cache(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """带缓存的查询"""
        cache_key = hashlib.md5(question.encode()).hexdigest()

        # 检查缓存
        if use_cache and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result

        # 执行查询
        result = self.query(question)

        # 缓存结果
        if use_cache:
            self.query_cache[cache_key] = result.copy()

        result["from_cache"] = False
        return result

    def multi_strategy_query(self, question: str) -> Dict[str, Any]:
        """多策略查询"""
        strategies = ["similarity", "mmr"]
        all_docs = []

        for strategy in strategies:
            docs = self.enhanced_retrieve(question, strategy=strategy, k=2)
            all_docs.extend(docs)

        # 去重
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        # 重排序
        reranked_docs = self.rerank_documents(question, unique_docs)

        # 生成答案
        answer = self.generate_answer(question, reranked_docs[:3])

        return {
            "question": question,
            "answer": answer,
            "strategy": "multi_strategy",
            "total_docs_retrieved": len(all_docs),
            "unique_docs": len(unique_docs),
            "sources": [
                {
                    "content": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                }
                for doc in reranked_docs[:3]
            ]
        }


# ================================
# 5. Agentic RAG 系统
# ================================

class AgenticRAGSystem:
    """智能 RAG 系统"""

    def __init__(self, rag_system: AdvancedRAGSystem):
        self.rag_system = rag_system
        self.llm = MockLLM("agentic-rag")
        self.conversation_history = []

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        analysis = {
            "query_type": "factual",  # factual, how_to, comparison, etc.
            "complexity": "medium",   # low, medium, high
            "needs_retrieval": True,
            "suggested_strategy": "similarity",
            "keywords": self._extract_keywords(query)
        }

        # 简单的查询类型分析
        if any(word in query.lower() for word in ["如何", "怎么", "how to"]):
            analysis["query_type"] = "how_to"
            analysis["suggested_strategy"] = "mmr"
        elif any(word in query.lower() for word in ["比较", "区别", "vs", "difference"]):
            analysis["query_type"] = "comparison"
            analysis["suggested_strategy"] = "multi_strategy"
        elif any(word in query.lower() for word in ["什么是", "介绍", "what is"]):
            analysis["query_type"] = "definition"

        return analysis

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        stop_words = {"的", "是", "在", "和", "或", "与", "了", "吗", "呢", "a", "an", "the", "is", "are", "in", "on", "at"}
        words = [word.lower() for word in query.split() if word.lower() not in stop_words]
        return words

    def iterative_retrieval(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """迭代检索"""
        all_results = []
        current_query = query
        confidence_scores = []

        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}: {current_query}")

            # 检索文档
            docs = self.rag_system.enhanced_retrieve(current_query, k=3)

            # 评估检索质量
            confidence = self._evaluate_retrieval_quality(current_query, docs)
            confidence_scores.append(confidence)

            result = {
                "iteration": iteration + 1,
                "query": current_query,
                "docs_found": len(docs),
                "confidence": confidence,
                "docs": docs
            }
            all_results.append(result)

            # 如果置信度足够高，停止迭代
            if confidence > 0.8:
                break

            # 改进查询
            if iteration < max_iterations - 1:
                current_query = self._refine_query(current_query, docs)

        # 选择最佳结果
        best_result = max(all_results, key=lambda x: x["confidence"])

        return {
            "final_query": best_result["query"],
            "best_docs": best_result["docs"],
            "iterations": len(all_results),
            "confidence_progression": confidence_scores,
            "all_results": all_results
        }

    def _evaluate_retrieval_quality(self, query: str, docs: List[MockDocument]) -> float:
        """评估检索质量"""
        if not docs:
            return 0.0

        query_words = set(query.lower().split())
        total_relevance = 0

        for doc in docs:
            doc_words = set(doc.page_content.lower().split())
            relevance = len(query_words.intersection(doc_words)) / len(query_words)
            total_relevance += relevance

        return min(total_relevance / len(docs), 1.0)

    def _refine_query(self, original_query: str, docs: List[MockDocument]) -> str:
        """改进查询"""
        if not docs:
            return original_query

        # 从文档中提取相关术语
        doc_text = " ".join([doc.page_content for doc in docs])
        doc_words = doc_text.lower().split()

        # 找到频繁出现的术语
        word_freq = defaultdict(int)
        for word in doc_words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] += 1

        # 添加高频术语到查询中
        frequent_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        additional_terms = [term for term, _ in frequent_terms if term not in original_query.lower()]

        if additional_terms:
            refined_query = f"{original_query} {' '.join(additional_terms[:1])}"
            return refined_query

        return original_query

    def conversational_rag(self, query: str) -> Dict[str, Any]:
        """对话式 RAG"""
        # 分析查询
        analysis = self.analyze_query(query)

        # 考虑对话历史
        context_aware_query = self._build_context_aware_query(query)

        # 执行检索
        if analysis["complexity"] == "high":
            retrieval_result = self.iterative_retrieval(context_aware_query)
            docs = retrieval_result["best_docs"]
        else:
            strategy = analysis["suggested_strategy"]
            docs = self.rag_system.enhanced_retrieve(context_aware_query, strategy=strategy)

        # 生成答案
        answer = self.rag_system.generate_answer(query, docs)

        # 添加到对话历史
        self.conversation_history.append({
            "query": query,
            "answer": answer,
            "timestamp": time.time()
        })

        return {
            "query": query,
            "context_aware_query": context_aware_query,
            "answer": answer,
            "analysis": analysis,
            "docs_used": len(docs),
            "conversation_turn": len(self.conversation_history)
        }

    def _build_context_aware_query(self, query: str) -> str:
        """构建上下文感知查询"""
        if not self.conversation_history:
            return query

        # 获取最近的对话历史
        recent_history = self.conversation_history[-2:]  # 最近2轮对话

        # 提取上下文关键词
        context_keywords = []
        for turn in recent_history:
            keywords = self._extract_keywords(turn["query"])
            context_keywords.extend(keywords)

        # 如果当前查询很短且可能需要上下文
        if len(query.split()) <= 3 and context_keywords:
            # 添加相关的上下文关键词
            relevant_context = context_keywords[:2]  # 限制上下文关键词数量
            enhanced_query = f"{query} {' '.join(relevant_context)}"
            return enhanced_query

        return query


# ================================
# 6. RAG 评估和优化
# ================================

class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self):
        self.test_queries = [
            "什么是 LangGraph？",
            "如何在 LangGraph 中管理状态？",
            "LangGraph 的工具机制是如何工作的？",
            "条件路由和普通路由的区别是什么？",
            "RAG 系统的主要组件有哪些？"
        ]

    def evaluate_system(self, rag_system: BasicRAGSystem) -> Dict[str, Any]:
        """评估 RAG 系统性能"""
        results = []

        for query in self.test_queries:
            start_time = time.time()

            # 执行查询
            result = rag_system.query(query)

            # 计算指标
            metrics = {
                "query": query,
                "response_time": time.time() - start_time,
                "docs_retrieved": result["retrieved_docs"],
                "answer_length": len(result["answer"]),
                "has_answer": "抱歉" not in result["answer"] and "没有" not in result["answer"]
            }

            results.append(metrics)

        # 计算总体指标
        total_time = sum(r["response_time"] for r in results)
        avg_docs = sum(r["docs_retrieved"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["has_answer"]) / len(results)

        return {
            "individual_results": results,
            "summary": {
                "total_queries": len(self.test_queries),
                "avg_response_time": total_time / len(results),
                "avg_docs_retrieved": avg_docs,
                "success_rate": success_rate,
                "total_evaluation_time": total_time
            }
        }

    def compare_systems(self, systems: Dict[str, BasicRAGSystem]) -> Dict[str, Any]:
        """比较多个 RAG 系统"""
        comparison_results = {}

        for name, system in systems.items():
            print(f"评估系统: {name}")
            comparison_results[name] = self.evaluate_system(system)

        # 生成比较报告
        report = {
            "systems": comparison_results,
            "winner": self._determine_winner(comparison_results)
        }

        return report

    def _determine_winner(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """确定最佳系统"""
        winners = {
            "fastest": "",
            "most_accurate": "",
            "most_comprehensive": ""
        }

        fastest_time = float('inf')
        highest_success = 0
        most_docs = 0

        for name, result in results.items():
            summary = result["summary"]

            if summary["avg_response_time"] < fastest_time:
                fastest_time = summary["avg_response_time"]
                winners["fastest"] = name

            if summary["success_rate"] > highest_success:
                highest_success = summary["success_rate"]
                winners["most_accurate"] = name

            if summary["avg_docs_retrieved"] > most_docs:
                most_docs = summary["avg_docs_retrieved"]
                winners["most_comprehensive"] = name

        return winners


# ================================
# 7. 示例和测试
# ================================

def create_sample_rag_systems():
    """创建示例 RAG 系统"""
    # 准备文档
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
    documents = processor.create_sample_documents()
    enhanced_docs = processor.enhance_metadata(documents)
    chunks = processor.split_documents(enhanced_docs)

    print(f"处理文档: {len(documents)} 个原始文档, {len(chunks)} 个文档块")

    # 创建不同的 RAG 系统
    systems = {
        "基础RAG": BasicRAGSystem(chunks),
        "高级RAG": AdvancedRAGSystem(chunks),
    }

    return systems


def test_basic_rag():
    """测试基础 RAG 功能"""
    print("\n🔍 基础 RAG 测试")
    print("=" * 50)

    systems = create_sample_rag_systems()
    basic_rag = systems["基础RAG"]

    test_queries = [
        "什么是 LangGraph？",
        "如何使用工具？",
        "状态管理的作用是什么？"
    ]

    for query in test_queries:
        print(f"\n🤔 问题: {query}")
        result = basic_rag.query(query)
        print(f"📝 回答: {result['answer'][:200]}...")
        print(f"📚 使用文档: {result['retrieved_docs']} 个")
        print(f"⏱️  处理时间: {result['processing_time']:.2f}s")


def test_advanced_rag():
    """测试高级 RAG 功能"""
    print("\n🚀 高级 RAG 测试")
    print("=" * 50)

    systems = create_sample_rag_systems()
    advanced_rag = systems["高级RAG"]

    # 测试多策略查询
    query = "LangGraph 的核心特性有哪些？"
    print(f"\n🎯 多策略查询: {query}")

    result = advanced_rag.multi_strategy_query(query)
    print(f"📝 回答: {result['answer'][:200]}...")
    print(f"📊 策略: {result['strategy']}")
    print(f"📚 检索文档: {result['total_docs_retrieved']} 个 (去重后: {result['unique_docs']} 个)")

    # 测试缓存功能
    print(f"\n💾 测试缓存功能")
    start_time = time.time()
    result1 = advanced_rag.query_with_cache(query)
    time1 = time.time() - start_time

    start_time = time.time()
    result2 = advanced_rag.query_with_cache(query)
    time2 = time.time() - start_time

    print(f"首次查询: {time1:.3f}s (缓存: {result1.get('from_cache', False)})")
    print(f"二次查询: {time2:.3f}s (缓存: {result2.get('from_cache', False)})")


def test_agentic_rag():
    """测试智能 RAG 系统"""
    print("\n🧠 智能 RAG 测试")
    print("=" * 50)

    systems = create_sample_rag_systems()
    agentic_rag = AgenticRAGSystem(systems["高级RAG"])

    # 测试查询分析
    query = "如何在 LangGraph 中实现条件路由？"
    print(f"\n🔍 查询分析: {query}")

    analysis = agentic_rag.analyze_query(query)
    print(f"📊 分析结果: {json.dumps(analysis, ensure_ascii=False, indent=2)}")

    # 测试对话式 RAG
    print(f"\n💬 对话式 RAG 测试")
    conversation_queries = [
        "什么是 LangGraph？",
        "它的主要特性是什么？",  # 应该利用上下文
        "如何使用工具功能？"
    ]

    for i, conv_query in enumerate(conversation_queries):
        print(f"\n第{i+1}轮对话: {conv_query}")
        result = agentic_rag.conversational_rag(conv_query)
        print(f"🎯 上下文感知查询: {result['context_aware_query']}")
        print(f"📝 回答: {result['answer'][:150]}...")

    # 测试迭代检索
    print(f"\n🔄 迭代检索测试")
    complex_query = "复杂的 AI 系统设计"
    iteration_result = agentic_rag.iterative_retrieval(complex_query, max_iterations=2)
    print(f"🎯 最终查询: {iteration_result['final_query']}")
    print(f"🔢 迭代次数: {iteration_result['iterations']}")
    print(f"📈 置信度变化: {iteration_result['confidence_progression']}")


def test_rag_evaluation():
    """测试 RAG 系统评估"""
    print("\n📊 RAG 系统评估")
    print("=" * 50)

    systems = create_sample_rag_systems()
    evaluator = RAGEvaluator()

    # 比较系统性能
    comparison = evaluator.compare_systems(systems)

    print("\n📈 评估结果:")
    for name, results in comparison["systems"].items():
        summary = results["summary"]
        print(f"\n{name}:")
        print(f"  平均响应时间: {summary['avg_response_time']:.3f}s")
        print(f"  平均检索文档数: {summary['avg_docs_retrieved']:.1f}")
        print(f"  成功率: {summary['success_rate']:.1%}")

    print(f"\n🏆 最佳系统:")
    winners = comparison["winner"]
    for category, winner in winners.items():
        print(f"  {category}: {winner}")


async def run_all_tests():
    """运行所有测试"""
    print("🤖 RAG 系统实现示例")
    print("=" * 60)

    test_basic_rag()
    test_advanced_rag()
    test_agentic_rag()
    test_rag_evaluation()

    print("\n" + "=" * 60)
    print("🎉 所有 RAG 测试完成!")


if __name__ == "__main__":
    # 检查依赖
    if not LANGCHAIN_AVAILABLE:
        print("📝 注意: 本示例使用模拟组件，实际使用请安装 LangChain 相关包:")
        print("pip install langchain langchain-community langchain-openai chromadb")
        print()

    # 运行所有测试
    asyncio.run(run_all_tests())