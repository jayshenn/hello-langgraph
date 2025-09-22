"""
Python 类型注解练习

完成这些练习来巩固你对 Python 类型系统的理解。
每个练习都有 TODO 标记，需要你添加正确的类型注解或实现功能。
"""

from typing import TypedDict, List, Dict, Optional, Union, Literal
from typing_extensions import NotRequired

# =============================================================================
# 练习 1: 基础类型注解
# =============================================================================

def calculate_bmi(weight, height):
    """
    TODO: 为这个函数添加类型注解
    计算 BMI = weight(kg) / height(m)^2
    """
    return weight / (height ** 2)


def format_person_info(name, age, is_student):
    """
    TODO: 添加类型注解
    格式化个人信息
    """
    status = "学生" if is_student else "非学生"
    return f"姓名: {name}, 年龄: {age}, 状态: {status}"


# =============================================================================
# 练习 2: 容器类型
# =============================================================================

def find_average_score(scores):
    """
    TODO: 添加类型注解
    计算分数列表的平均值
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def group_students_by_grade(students):
    """
    TODO: 添加类型注解
    students 格式: [{"name": "Alice", "grade": "A"}, ...]
    返回格式: {"A": ["Alice", ...], "B": [...]}
    """
    groups = {}
    for student in students:
        grade = student["grade"]
        if grade not in groups:
            groups[grade] = []
        groups[grade].append(student["name"])
    return groups


# =============================================================================
# 练习 3: TypedDict 设计
# =============================================================================

# TODO: 设计一个学生信息的 TypedDict，包含：
# - name: 姓名 (str)
# - age: 年龄 (int)
# - grades: 各科成绩 (Dict[str, int])
# - email: 邮箱 (可选, str)
# - phone: 电话 (可选, str)

class StudentInfo(TypedDict):
    # TODO: 在这里定义字段
    pass


# TODO: 设计一个任务状态的 TypedDict，包含：
# - task_id: 任务ID (str)
# - title: 任务标题 (str)
# - status: 状态，只能是 "pending", "in_progress", "completed", "failed"
# - priority: 优先级，1-5 的整数
# - assignee: 负责人 (可选, str)
# - due_date: 截止日期 (可选, str)

class TaskState(TypedDict):
    # TODO: 在这里定义字段
    pass


# =============================================================================
# 练习 4: 状态更新函数
# =============================================================================

def update_student_grade(student, subject, new_grade):
    """
    TODO: 添加类型注解并实现功能
    更新学生某科目的成绩，返回新的学生信息
    """
    # TODO: 实现功能
    pass


def mark_task_completed(task):
    """
    TODO: 添加类型注解并实现功能
    将任务标记为完成状态
    """
    # TODO: 实现功能
    pass


# =============================================================================
# 练习 5: LangGraph 风格的状态管理
# =============================================================================

# TODO: 设计一个简单聊天机器人的状态 TypedDict，包含：
# - user_input: 用户输入 (str)
# - bot_response: 机器人响应 (str)
# - conversation_count: 对话轮数 (int)
# - user_mood: 用户情绪 ("happy", "neutral", "sad", "angry")
# - bot_mode: 机器人模式 ("helpful", "creative", "factual")

class SimpleChatState(TypedDict):
    # TODO: 在这里定义字段
    pass


def greeting_node(state):
    """
    TODO: 添加类型注解并实现
    处理问候的节点，生成欢迎消息
    """
    # TODO: 实现功能
    pass


def mood_analysis_node(state):
    """
    TODO: 添加类型注解并实现
    分析用户情绪的节点（简单的关键词匹配）
    """
    # TODO: 实现功能
    # 提示: 可以检查 user_input 中的关键词
    # "happy", "great", "good" -> "happy"
    # "sad", "bad", "terrible" -> "sad"
    # "angry", "mad", "frustrated" -> "angry"
    # 其他 -> "neutral"
    pass


def response_generator_node(state):
    """
    TODO: 添加类型注解并实现
    根据用户情绪和机器人模式生成响应
    """
    # TODO: 实现功能
    pass


# =============================================================================
# 测试函数 (请勿修改)
# =============================================================================

def test_exercises():
    """测试所有练习的实现"""
    print("🧪 开始测试...")

    # 测试基础函数
    try:
        bmi = calculate_bmi(70, 1.75)
        print(f"✅ BMI 计算: {bmi:.2f}")
    except Exception as e:
        print(f"❌ BMI 计算错误: {e}")

    try:
        info = format_person_info("Alice", 25, True)
        print(f"✅ 个人信息: {info}")
    except Exception as e:
        print(f"❌ 个人信息格式化错误: {e}")

    # 测试容器函数
    try:
        avg = find_average_score([85, 90, 78, 92])
        print(f"✅ 平均分: {avg}")
    except Exception as e:
        print(f"❌ 平均分计算错误: {e}")

    print("\n🎯 完成所有 TODO 后运行测试查看结果!")


if __name__ == "__main__":
    test_exercises()


# =============================================================================
# 答案提示 (完成练习后可以查看)
# =============================================================================

"""
答案提示:

1. calculate_bmi:
   def calculate_bmi(weight: float, height: float) -> float:

2. format_person_info:
   def format_person_info(name: str, age: int, is_student: bool) -> str:

3. find_average_score:
   def find_average_score(scores: List[int]) -> float:

4. group_students_by_grade:
   def group_students_by_grade(students: List[Dict[str, str]]) -> Dict[str, List[str]]:

5. StudentInfo:
   class StudentInfo(TypedDict):
       name: str
       age: int
       grades: Dict[str, int]
       email: NotRequired[str]
       phone: NotRequired[str]

6. TaskState:
   class TaskState(TypedDict):
       task_id: str
       title: str
       status: Literal["pending", "in_progress", "completed", "failed"]
       priority: int
       assignee: NotRequired[str]
       due_date: NotRequired[str]

记住: 这些只是参考答案，你的实现可能有所不同但仍然正确！
"""