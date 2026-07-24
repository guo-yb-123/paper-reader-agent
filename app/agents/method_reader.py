"""
Method Reader Agent - 算法工程师视角
- 输入：方法论相关章节内容
- 输出：方法部分结构化总结（核心创新点、模型架构、算法逻辑、与前人工作区别）
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm
from app.services.section_utils import collect_section_contents

METHOD_SYSTEM_PROMPT = """你是一位资深算法工程师，专门负责深入分析学术论文的技术实现细节。

你的任务是精读论文的方法论部分，从技术实现的角度进行深度解析：

## 分析维度

### 1. 核心创新点
- 论文提出了什么新方法/新架构？核心创新是什么？
- 这个创新解决了什么技术难题？为什么之前的方法做不到？
- 创新的灵感来源是什么？是否有理论支撑？

### 2. 模型/算法架构
- 整体架构是怎样的？（请用文字描述清楚模块组成）
- 每个模块的功能和输入输出是什么？
- 数据在模型中如何流动？训练和推理流程分别是什么？
- 关键的超参数有哪些？如何设置的？

### 3. 关键技术细节
- 算法的数学原理是什么？核心公式的含义？
- 有哪些 trick 或实现细节值得注意？
- 计算复杂度如何？有什么优化策略？
- 训练过程有什么特殊设计（如 loss 函数、优化器、学习率策略）？

### 4. 与前人工作的技术对比
- 与前人方法在架构上有哪些本质区别？
- 改进是渐进式的还是颠覆式的？
- 该方法的优势和潜在不足分别是什么？
- 是否存在一些隐含的假设或限制条件？

## 输出格式
请用清晰的中文输出，包含以上四个维度的技术分析。
对关键公式和算法，请用 Markdown 格式呈现（代码块或数学公式）。
重要技术细节请用粗体标注。"""


def run_method_reader(
    paper_structure: dict,
    reading_plan: dict,
) -> str:
    """
    Method Reader：精读论文方法论部分。

    Args:
        paper_structure: 解析后的论文结构
        reading_plan: Planner 输出的阅读计划

    Returns:
        方法部分结构化总结（Markdown 格式文本）
    """
    logger.info("Method Reader 开始分析...")

    # 提取分配给 Method Reader 的章节内容
    method_plan = reading_plan.get("reading_plan", {}).get("method_reader", {})
    assigned_sections = method_plan.get("sections", [])
    focus_points = method_plan.get("focus_points", [])
    expected_output = method_plan.get("expected_output", "")

    section_contents = collect_section_contents(paper_structure, assigned_sections)

    focus_text = "\n".join(f"- {p}" for p in focus_points) if focus_points else "（无特殊关注点）"

    user_message = f"""请分析以下论文的方法论部分：

## 论文信息
- 标题：{paper_structure.get('title', 'Unknown')}

## 重点关注
{focus_text}

## 预期输出
{expected_output}

## 章节内容
{section_contents}

请按照系统提示中的四个维度，进行深度技术分析。"""

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=METHOD_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"Method Reader 分析完成，输出 {len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"Method Reader 执行失败: {e}")
        return f"## 方法分析失败\n\n错误: {e}\n\n请检查 API 配置。"
