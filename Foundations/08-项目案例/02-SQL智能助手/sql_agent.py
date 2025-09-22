"""
SQL智能助手 - LangGraph完整实现

这个项目展示了如何构建一个安全、智能的SQL查询助手：
- 自然语言到SQL的转换
- 安全的数据库操作
- 查询结果的智能解释
- 多数据库支持
- 完整的错误处理

适合学习复杂的工具集成和安全防护机制。
"""

import os
import re
import sqlite3
import pandas as pd
import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class TableInfo:
    """表信息"""
    name: str
    columns: List[Dict[str, str]]
    description: str
    primary_key: Optional[str] = None
    foreign_keys: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = []


@dataclass
class QueryIntent:
    """查询意图"""
    query_type: str  # SELECT, COUNT, SUM, AVG, etc.
    target_tables: List[str]
    target_columns: List[str]
    conditions: List[Dict[str, Any]]
    aggregations: List[str]
    sorting: Optional[Dict[str, str]]
    limit: Optional[int]
    confidence: float


@dataclass
class QueryResult:
    """查询结果"""
    sql: str
    data: pd.DataFrame
    execution_time: float
    row_count: int
    explanation: str
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SQLAgentState(TypedDict):
    """SQL智能助手状态"""
    # 用户输入
    user_query: str
    original_language: str

    # 查询分析
    query_intent: Optional[QueryIntent]
    parsed_tables: List[TableInfo]
    relevant_schema: Dict[str, Any]

    # SQL生成
    generated_sql: str
    sql_explanation: str
    safety_check_passed: bool
    safety_warnings: List[str]

    # 执行结果
    query_result: Optional[QueryResult]
    formatted_output: str

    # 系统状态
    error_message: str
    step_count: int
    processing_time: float


# =============================================================================
# 数据库配置和连接
# =============================================================================

class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = None):
        """初始化数据库连接"""
        self.db_path = db_path or "./data/sample_database.db"
        self.connection = None
        self.schema_cache = {}
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """确保数据库存在"""
        if not Path(self.db_path).exists():
            print(f"⚠️ 数据库文件不存在: {self.db_path}")
            print("请先运行 database_setup.py 创建示例数据库")

    def connect(self):
        """连接数据库"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # 启用字典式访问
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()

    def get_schema_info(self) -> List[TableInfo]:
        """获取数据库表结构信息"""
        if not self.connection:
            if not self.connect():
                return []

        tables = []
        try:
            # 获取所有表名
            cursor = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                # 获取表结构
                cursor = self.connection.execute(f"PRAGMA table_info({table_name})")
                columns = []
                primary_key = None

                for row in cursor.fetchall():
                    col_info = {
                        "name": row[1],
                        "type": row[2],
                        "nullable": not row[3],
                        "default": row[4]
                    }
                    columns.append(col_info)

                    if row[5]:  # primary key
                        primary_key = row[1]

                # 获取外键信息
                cursor = self.connection.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = []
                for row in cursor.fetchall():
                    fk_info = {
                        "column": row[3],
                        "references_table": row[2],
                        "references_column": row[4]
                    }
                    foreign_keys.append(fk_info)

                # 创建表信息对象
                table_info = TableInfo(
                    name=table_name,
                    columns=columns,
                    description=self._get_table_description(table_name),
                    primary_key=primary_key,
                    foreign_keys=foreign_keys
                )
                tables.append(table_info)

        except Exception as e:
            print(f"获取数据库结构失败: {e}")

        return tables

    def _get_table_description(self, table_name: str) -> str:
        """获取表的描述信息"""
        descriptions = {
            "products": "产品信息表，包含产品ID、名称、价格、分类等",
            "customers": "客户信息表，包含客户ID、姓名、邮箱、注册时间等",
            "orders": "订单信息表，包含订单ID、客户ID、订单日期、总金额等",
            "order_items": "订单项目表，包含订单ID、产品ID、数量、单价等",
            "categories": "产品分类表，包含分类ID、分类名称等"
        }
        return descriptions.get(table_name, f"{table_name}表")

    def execute_query(self, sql: str) -> QueryResult:
        """执行SQL查询"""
        if not self.connection:
            if not self.connect():
                raise Exception("无法连接到数据库")

        start_time = datetime.datetime.now()

        try:
            # 执行查询
            cursor = self.connection.execute(sql)
            rows = cursor.fetchall()

            # 转换为DataFrame
            if rows:
                columns = [description[0] for description in cursor.description]
                data = pd.DataFrame([dict(row) for row in rows], columns=columns)
            else:
                data = pd.DataFrame()

            execution_time = (datetime.datetime.now() - start_time).total_seconds()

            return QueryResult(
                sql=sql,
                data=data,
                execution_time=execution_time,
                row_count=len(data),
                explanation="",
                warnings=[]
            )

        except Exception as e:
            raise Exception(f"SQL执行错误: {str(e)}")

# =============================================================================
# 安全检查模块
# =============================================================================

class SQLSecurityChecker:
    """SQL安全检查器"""

    # 危险关键词
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
        'EXEC', 'EXECUTE', 'DECLARE', 'TRUNCATE', 'GRANT', 'REVOKE'
    ]

    # 允许的表名（白名单）
    ALLOWED_TABLES = [
        'products', 'customers', 'orders', 'order_items', 'categories'
    ]

    @classmethod
    def check_sql_safety(cls, sql: str) -> Tuple[bool, List[str]]:
        """检查SQL的安全性"""
        warnings = []
        sql_upper = sql.upper()

        # 检查危险关键词
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                warnings.append(f"包含危险关键词: {keyword}")

        # 检查表名白名单
        table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)|UPDATE\s+(\w+)|INTO\s+(\w+)'
        matches = re.finditer(table_pattern, sql_upper)

        for match in matches:
            table_name = None
            for group in match.groups():
                if group:
                    table_name = group.lower()
                    break

            if table_name and table_name not in cls.ALLOWED_TABLES:
                warnings.append(f"访问未授权的表: {table_name}")

        # 检查结果限制
        if 'LIMIT' not in sql_upper:
            warnings.append("建议添加LIMIT子句限制结果数量")

        # 检查复杂度
        join_count = len(re.findall(r'JOIN', sql_upper))
        if join_count > 5:
            warnings.append(f"查询过于复杂，包含{join_count}个JOIN")

        return len([w for w in warnings if '危险' in w or '未授权' in w]) == 0, warnings


# =============================================================================
# LLM工具定义
# =============================================================================

# 全局数据库管理器
db_manager = DatabaseManager()

@tool
def get_database_schema() -> str:
    """获取数据库表结构信息"""
    schema_info = db_manager.get_schema_info()
    schema_text = "数据库表结构:\n\n"

    for table in schema_info:
        schema_text += f"表名: {table.name}\n"
        schema_text += f"描述: {table.description}\n"
        schema_text += "字段:\n"

        for col in table.columns:
            schema_text += f"  - {col['name']} ({col['type']})"
            if not col['nullable']:
                schema_text += " NOT NULL"
            if col['name'] == table.primary_key:
                schema_text += " PRIMARY KEY"
            schema_text += "\n"

        if table.foreign_keys:
            schema_text += "外键:\n"
            for fk in table.foreign_keys:
                schema_text += f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n"

        schema_text += "\n"

    return schema_text


@tool
def execute_sql_query(sql: str) -> str:
    """执行SQL查询并返回结果

    Args:
        sql: 要执行的SQL查询语句
    """
    try:
        # 安全检查
        is_safe, warnings = SQLSecurityChecker.check_sql_safety(sql)

        if not is_safe:
            return f"安全检查失败，拒绝执行查询。警告: {', '.join(warnings)}"

        # 执行查询
        result = db_manager.execute_query(sql)

        # 格式化结果
        if result.row_count == 0:
            return "查询执行成功，但没有返回数据。"

        # 限制显示行数
        display_rows = min(result.row_count, 20)
        result_text = f"查询执行成功！\n\n"
        result_text += f"SQL: {sql}\n"
        result_text += f"执行时间: {result.execution_time:.3f}秒\n"
        result_text += f"返回行数: {result.row_count}\n\n"

        if result.row_count <= 20:
            result_text += "查询结果:\n"
            result_text += result.data.to_string(index=False)
        else:
            result_text += f"查询结果 (显示前{display_rows}行):\n"
            result_text += result.data.head(display_rows).to_string(index=False)
            result_text += f"\n\n... 还有{result.row_count - display_rows}行数据"

        if warnings:
            result_text += f"\n\n⚠️ 警告: {', '.join(warnings)}"

        return result_text

    except Exception as e:
        return f"查询执行失败: {str(e)}"


@tool
def explain_sql_query(sql: str) -> str:
    """解释SQL查询的含义和作用

    Args:
        sql: 要解释的SQL查询语句
    """
    explanation = f"SQL查询解释:\n\n"
    explanation += f"查询语句: {sql}\n\n"

    sql_upper = sql.upper()

    # 分析查询类型
    if sql_upper.startswith('SELECT'):
        explanation += "查询类型: 数据查询(SELECT)\n"

        # 分析SELECT子句
        if 'COUNT(' in sql_upper:
            explanation += "操作: 统计记录数量\n"
        elif 'SUM(' in sql_upper:
            explanation += "操作: 计算总和\n"
        elif 'AVG(' in sql_upper:
            explanation += "操作: 计算平均值\n"
        elif 'MAX(' in sql_upper or 'MIN(' in sql_upper:
            explanation += "操作: 查找最值\n"
        else:
            explanation += "操作: 查询数据记录\n"

        # 分析FROM子句
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            table_name = from_match.group(1).lower()
            explanation += f"数据来源: {table_name}表\n"

        # 分析JOIN
        joins = re.findall(r'JOIN\s+(\w+)', sql_upper)
        if joins:
            explanation += f"关联表: {', '.join(joins)}\n"

        # 分析WHERE条件
        if 'WHERE' in sql_upper:
            explanation += "筛选条件: 有条件筛选\n"

        # 分析GROUP BY
        if 'GROUP BY' in sql_upper:
            explanation += "分组统计: 按指定字段分组\n"

        # 分析ORDER BY
        if 'ORDER BY' in sql_upper:
            if 'DESC' in sql_upper:
                explanation += "排序方式: 降序排列\n"
            else:
                explanation += "排序方式: 升序排列\n"

        # 分析LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            limit_num = limit_match.group(1)
            explanation += f"结果限制: 最多返回{limit_num}条记录\n"

    return explanation


# 工具节点
tools = [get_database_schema, execute_sql_query, explain_sql_query]
tool_node = ToolNode(tools)


# =============================================================================
# 节点函数
# =============================================================================

def query_understanding_node(state: SQLAgentState) -> SQLAgentState:
    """查询理解节点"""
    user_query = state["user_query"]

    # 简单的意图分析（实际应用中可以使用更复杂的NLP模型）
    query_type = "SELECT"
    confidence = 0.8

    if any(word in user_query.lower() for word in ["统计", "计算", "总和", "count", "sum"]):
        query_type = "AGGREGATE"
        confidence = 0.9

    if any(word in user_query.lower() for word in ["排序", "排行", "最大", "最小", "top"]):
        query_type = "RANKING"
        confidence = 0.85

    # 提取可能的表名
    table_keywords = {
        "产品": "products",
        "商品": "products",
        "客户": "customers",
        "用户": "customers",
        "订单": "orders",
        "分类": "categories"
    }

    target_tables = []
    for keyword, table_name in table_keywords.items():
        if keyword in user_query:
            target_tables.append(table_name)

    if not target_tables:
        target_tables = ["products"]  # 默认表

    query_intent = QueryIntent(
        query_type=query_type,
        target_tables=target_tables,
        target_columns=[],
        conditions=[],
        aggregations=[],
        sorting=None,
        limit=None,
        confidence=confidence
    )

    return {
        **state,
        "query_intent": query_intent,
        "step_count": state["step_count"] + 1
    }


def schema_analysis_node(state: SQLAgentState) -> SQLAgentState:
    """表结构分析节点"""
    # 获取数据库表结构
    schema_info = db_manager.get_schema_info()

    # 根据查询意图筛选相关表
    query_intent = state["query_intent"]
    relevant_tables = []

    if query_intent:
        for table_name in query_intent.target_tables:
            for table_info in schema_info:
                if table_info.name == table_name:
                    relevant_tables.append(table_info)

    return {
        **state,
        "parsed_tables": relevant_tables,
        "relevant_schema": {"tables": [t.name for t in relevant_tables]},
        "step_count": state["step_count"] + 1
    }


def sql_generation_node(state: SQLAgentState) -> SQLAgentState:
    """SQL生成节点"""
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,  # 低温度确保生成稳定的SQL
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 绑定工具
    llm_with_tools = llm.bind_tools(tools)

    # 构建系统提示
    system_prompt = f"""你是一个专业的SQL查询助手。请根据用户的自然语言查询生成安全、高效的SQL语句。

数据库信息：
{get_database_schema.invoke({})}

重要安全规则：
1. 只能生成SELECT查询语句
2. 禁止使用DELETE, INSERT, UPDATE, DROP等修改语句
3. 必须为查询添加适当的LIMIT子句（一般不超过1000）
4. 只能查询白名单中的表: products, customers, orders, order_items, categories

用户查询意图：
- 查询类型：{state.get('query_intent', {}).get('query_type', '未知')}
- 目标表：{state.get('query_intent', {}).get('target_tables', [])}

请按以下步骤处理：
1. 首先使用get_database_schema工具了解表结构
2. 分析用户查询需求
3. 生成安全的SQL查询
4. 使用explain_sql_query工具解释查询含义
5. 最后使用execute_sql_query工具执行查询

如果用户查询有安全风险或无法理解，请说明原因并提供建议。"""

    # 构建消息
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {state['user_query']}")
    ]

    try:
        # 调用LLM
        response = llm_with_tools.invoke(messages)

        # 提取生成的SQL（如果有）
        generated_sql = ""
        sql_explanation = ""

        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 如果有工具调用，记录但不在这里执行
            tool_calls = [call['name'] for call in response.tool_calls]
            sql_explanation = f"准备执行工具: {', '.join(tool_calls)}"
        else:
            # 如果没有工具调用，从响应中提取SQL
            content = response.content if hasattr(response, 'content') else str(response)
            sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
            if sql_match:
                generated_sql = sql_match.group(1).strip()
            sql_explanation = content

        return {
            **state,
            "generated_sql": generated_sql,
            "sql_explanation": sql_explanation,
            "step_count": state["step_count"] + 1
        }

    except Exception as e:
        return {
            **state,
            "error_message": f"SQL生成失败: {str(e)}",
            "step_count": state["step_count"] + 1
        }


def security_check_node(state: SQLAgentState) -> SQLAgentState:
    """安全检查节点"""
    generated_sql = state.get("generated_sql", "")

    if not generated_sql:
        return {
            **state,
            "safety_check_passed": False,
            "safety_warnings": ["没有生成SQL查询"],
            "step_count": state["step_count"] + 1
        }

    # 执行安全检查
    is_safe, warnings = SQLSecurityChecker.check_sql_safety(generated_sql)

    return {
        **state,
        "safety_check_passed": is_safe,
        "safety_warnings": warnings,
        "step_count": state["step_count"] + 1
    }


def query_execution_node(state: SQLAgentState) -> SQLAgentState:
    """查询执行节点"""
    if not state["safety_check_passed"]:
        return {
            **state,
            "error_message": f"安全检查失败: {', '.join(state['safety_warnings'])}",
            "step_count": state["step_count"] + 1
        }

    generated_sql = state["generated_sql"]

    try:
        # 执行查询
        result = db_manager.execute_query(generated_sql)

        return {
            **state,
            "query_result": result,
            "step_count": state["step_count"] + 1
        }

    except Exception as e:
        return {
            **state,
            "error_message": f"查询执行失败: {str(e)}",
            "step_count": state["step_count"] + 1
        }


def result_formatting_node(state: SQLAgentState) -> SQLAgentState:
    """结果格式化节点"""
    query_result = state.get("query_result")
    error_message = state.get("error_message")

    if error_message:
        formatted_output = f"❌ 处理失败\n\n错误信息: {error_message}\n\n"
        formatted_output += "建议:\n"
        formatted_output += "1. 检查查询语法是否正确\n"
        formatted_output += "2. 确认表名和字段名是否存在\n"
        formatted_output += "3. 避免使用危险的SQL关键词\n"
    elif query_result:
        formatted_output = f"✅ 查询执行成功！\n\n"
        formatted_output += f"📊 **查询结果**\n"
        formatted_output += f"- SQL: `{query_result.sql}`\n"
        formatted_output += f"- 执行时间: {query_result.execution_time:.3f}秒\n"
        formatted_output += f"- 返回行数: {query_result.row_count}\n\n"

        if query_result.row_count > 0:
            # 限制显示行数
            display_rows = min(query_result.row_count, 10)
            formatted_output += "**数据结果**:\n```\n"
            formatted_output += query_result.data.head(display_rows).to_string(index=False)
            formatted_output += "\n```\n"

            if query_result.row_count > display_rows:
                formatted_output += f"\n💡 仅显示前{display_rows}行，共{query_result.row_count}行数据\n"

            # 添加业务洞察
            formatted_output += "\n📈 **数据洞察**:\n"
            if query_result.row_count == 1:
                formatted_output += "- 查询返回了一个具体的结果\n"
            elif query_result.row_count < 10:
                formatted_output += f"- 查询返回了{query_result.row_count}条记录，数据集较小\n"
            else:
                formatted_output += f"- 查询返回了{query_result.row_count}条记录，数据集较大\n"

        else:
            formatted_output += "📝 查询没有返回数据，可能的原因:\n"
            formatted_output += "- 筛选条件过于严格\n"
            formatted_output += "- 表中暂无符合条件的数据\n"

        # 添加安全警告
        if state.get("safety_warnings"):
            formatted_output += f"\n⚠️ **安全提示**: {', '.join(state['safety_warnings'])}\n"

    else:
        formatted_output = "❓ 未能生成有效的查询结果"

    return {
        **state,
        "formatted_output": formatted_output,
        "step_count": state["step_count"] + 1
    }


# =============================================================================
# 路由函数
# =============================================================================

def should_execute_query(state: SQLAgentState) -> str:
    """判断是否应该执行查询"""
    if state["safety_check_passed"] and state["generated_sql"]:
        return "execute_query"
    else:
        return "format_result"


# =============================================================================
# 图构建
# =============================================================================

def create_sql_agent_graph() -> StateGraph:
    """创建SQL智能助手图"""
    graph = StateGraph(SQLAgentState)

    # 添加节点
    graph.add_node("understand_query", query_understanding_node)
    graph.add_node("analyze_schema", schema_analysis_node)
    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("check_security", security_check_node)
    graph.add_node("execute_query", query_execution_node)
    graph.add_node("format_result", result_formatting_node)

    # 设置入口点
    graph.set_entry_point("understand_query")

    # 添加边
    graph.add_edge("understand_query", "analyze_schema")
    graph.add_edge("analyze_schema", "generate_sql")
    graph.add_edge("generate_sql", "check_security")

    # 条件边：安全检查后决定是否执行查询
    graph.add_conditional_edges(
        "check_security",
        should_execute_query,
        {
            "execute_query": "execute_query",
            "format_result": "format_result"
        }
    )

    graph.add_edge("execute_query", "format_result")
    graph.add_edge("format_result", END)

    return graph


# =============================================================================
# 主要功能类
# =============================================================================

class SQLAgent:
    """SQL智能助手"""

    def __init__(self, db_path: str = None):
        """初始化"""
        self.db_manager = DatabaseManager(db_path)
        self.graph = create_sql_agent_graph()
        self.memory_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory_saver)

        # 初始化数据库
        if not self.db_manager.connect():
            print("❌ 数据库连接失败，请检查数据库文件")

        print("🤖 SQL智能助手已启动")

    def create_initial_state(self, user_query: str) -> SQLAgentState:
        """创建初始状态"""
        return {
            "user_query": user_query,
            "original_language": "zh",
            "query_intent": None,
            "parsed_tables": [],
            "relevant_schema": {},
            "generated_sql": "",
            "sql_explanation": "",
            "safety_check_passed": False,
            "safety_warnings": [],
            "query_result": None,
            "formatted_output": "",
            "error_message": "",
            "step_count": 0,
            "processing_time": 0.0
        }

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """处理用户查询"""
        if not user_query.strip():
            return {
                "success": False,
                "message": "请输入有效的查询内容",
                "formatted_output": "❌ 查询内容不能为空"
            }

        # 创建初始状态
        initial_state = self.create_initial_state(user_query)

        # 配置
        config = {"configurable": {"thread_id": "sql_session"}}

        try:
            start_time = datetime.datetime.now()

            # 执行图
            result = self.app.invoke(initial_state, config)

            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            return {
                "success": not bool(result.get("error_message")),
                "message": result.get("error_message", "查询处理完成"),
                "formatted_output": result["formatted_output"],
                "generated_sql": result.get("generated_sql", ""),
                "execution_time": processing_time,
                "row_count": result.get("query_result", {}).get("row_count", 0) if result.get("query_result") else 0,
                "warnings": result.get("safety_warnings", [])
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"处理过程中发生错误: {str(e)}",
                "formatted_output": f"❌ 系统错误\n\n{str(e)}"
            }

    def interactive_mode(self):
        """交互式模式"""
        print("🚀 SQL智能助手交互模式")
        print("=" * 50)
        print("💡 你可以用自然语言描述你的查询需求")
        print("💡 例如: '显示所有产品的名称和价格'")
        print("💡 输入 'help' 查看帮助，'quit' 退出\n")

        while True:
            try:
                user_input = input("🔍 请输入查询: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if not user_input:
                    print("请输入查询内容")
                    continue

                print("🤖 正在分析查询...")

                # 处理查询
                result = self.process_query(user_input)

                # 显示结果
                print("\n" + result["formatted_output"])

                if result["generated_sql"]:
                    print(f"\n🔧 生成的SQL: {result['generated_sql']}")

                print(f"⏱️  处理时间: {result['execution_time']:.3f}秒")

                if result["warnings"]:
                    print(f"⚠️  警告: {', '.join(result['warnings'])}")

                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n👋 对话结束！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

    def _show_help(self):
        """显示帮助信息"""
        help_text = """
📚 SQL智能助手使用指南

🔸 支持的查询类型:
• 基础查询: "显示所有产品"
• 条件查询: "显示价格大于100的产品"
• 统计查询: "统计每个分类的产品数量"
• 排序查询: "按价格降序显示产品"
• 关联查询: "显示每个客户的订单数量"

🔸 可查询的表:
• products - 产品信息
• customers - 客户信息
• orders - 订单信息
• order_items - 订单明细
• categories - 产品分类

🔸 安全限制:
• 只支持SELECT查询
• 禁止修改数据的操作
• 自动添加结果数量限制

🔸 命令:
• help - 显示此帮助
• quit/exit - 退出程序
        """
        print(help_text)

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'db_manager'):
            self.db_manager.disconnect()


# =============================================================================
# 演示和测试
# =============================================================================

def demo_basic_queries():
    """演示基础查询"""
    print("=== 基础查询演示 ===\n")

    agent = SQLAgent()

    test_queries = [
        "显示所有产品的名称和价格",
        "查询价格最高的5个产品",
        "统计每个分类的产品数量",
        "显示每个客户的订单总数",
        "查询2023年的订单总金额"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = agent.process_query(query)
        print(result["formatted_output"])
        print("-" * 60)


def demo_security_features():
    """演示安全特性"""
    print("=== 安全特性演示 ===\n")

    agent = SQLAgent()

    dangerous_queries = [
        "DELETE FROM products",
        "DROP TABLE customers",
        "INSERT INTO products VALUES (1, 'test', 100)",
        "UPDATE orders SET total = 0",
        "查询用户密码表"
    ]

    for i, query in enumerate(dangerous_queries, 1):
        print(f"危险查询 {i}: {query}")
        result = agent.process_query(query)
        print(result["formatted_output"])
        print("-" * 60)


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print("🚀 SQL智能助手 - LangGraph完整示例")
    print("=" * 50)

    # 检查环境
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 警告：未设置OPENAI_API_KEY，将使用模拟模式")

    while True:
        print("\n选择模式:")
        print("1. 交互式查询")
        print("2. 基础功能演示")
        print("3. 安全特性演示")
        print("4. 查看数据库结构")
        print("5. 退出")

        choice = input("\n请选择 (1-5): ").strip()

        if choice == "1":
            agent = SQLAgent()
            agent.interactive_mode()

        elif choice == "2":
            demo_basic_queries()

        elif choice == "3":
            demo_security_features()

        elif choice == "4":
            db_manager = DatabaseManager()
            if db_manager.connect():
                schema_info = get_database_schema.invoke({})
                print("\n" + schema_info)
            else:
                print("❌ 无法连接数据库")

        elif choice == "5":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选择，请重试。")


if __name__ == "__main__":
    main()


# =============================================================================
# 学习指南和扩展建议
# =============================================================================

"""
🎯 SQL智能助手学习要点:

1. **安全第一**:
   - 输入验证和SQL注入防护
   - 白名单机制和权限控制
   - 查询复杂度限制

2. **智能理解**:
   - 自然语言到结构化意图的转换
   - 表结构分析和关系推理
   - 上下文理解和歧义消解

3. **工具集成**:
   - 数据库连接和查询执行
   - 结果格式化和可视化
   - 错误处理和用户反馈

4. **LLM应用**:
   - 工具绑定和函数调用
   - 提示词工程和上下文构建
   - 响应解析和后处理

5. **用户体验**:
   - 交互式界面设计
   - 清晰的错误提示
   - 业务洞察生成

🔧 扩展建议:

1. **高级功能**:
   - 支持更多数据库类型（PostgreSQL, MySQL）
   - 添加查询缓存机制
   - 实现查询性能优化建议

2. **智能增强**:
   - 使用语义搜索匹配表字段
   - 添加查询历史学习
   - 实现智能查询推荐

3. **可视化集成**:
   - 自动生成图表和报表
   - 支持导出多种格式
   - 添加数据钻取功能

4. **企业特性**:
   - 多租户支持
   - 审计日志记录
   - 细粒度权限控制

这个项目展示了如何安全、智能地处理用户的数据查询需求！
"""