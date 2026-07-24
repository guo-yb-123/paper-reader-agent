"""
Background Reader Agent - 领域研究者视角
- 输入：引言、相关工作等章节内容
- 输出：研究背景与相关工作总结（研究问题、领域现状、本文定位）
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm
from app.services.section_utils import collect_section_contents

BACKGROUND_SYSTEM_PROMPT = """你是一位资深的领域研究者，专门负责分析学术论文的研究背景与相关工作。

你的任务是精读论文的引言、背景和相关工作部分，从以下几个方面进行深度分析：

## 分析维度

### 1. 研究问题与动机
- 这篇论文要解决什么问题？为什么这个问题重要？
- 作者如何论证问题的紧迫性和研究价值？
- 问题的难点和挑战在哪里？

### 2. 领域现状与演进
- 该领域的发展脉络是怎样的？经历了哪些关键阶段？
- 当前主流方法有哪些？各自的优缺点是什么？
- 存在哪些尚未被充分解决的问题？

### 3. 相关工作分析
- 与本文最相关的前人工作有哪些？具体做了什么？
- 这些工作与本文的关键区别在哪里？
- 作者如何建立自己的工作在领域中的位置？

### 4. 本文贡献与定位
- 论文的核心贡献是什么？如何超越前人工作？
- 贡献之间的逻辑关系是怎样的？
- 该工作在领域发展中处于什么位置？

## 输出格式
请用清晰的中文输出，包含以上四个维度的分析。
使用 Markdown 格式，必要时使用列表和表格增强可读性。"""


def run_background_reader(
    paper_structure: dict,
    reading_plan: dict,
) -> str:
    """
    Background Reader：精读论文背景与相关工作。

    Args:
        paper_structure: 解析后的论文结构
        reading_plan: Planner 输出的阅读计划

    Returns:
        背景与相关工作总结（Markdown 格式文本）
    """
    logger.info("Background Reader 开始分析...")

    # 提取分配给 Background Reader 的章节内容
    bg_plan = reading_plan.get("reading_plan", {}).get("background_reader", {})
    assigned_sections = bg_plan.get("sections", [])
    focus_points = bg_plan.get("focus_points", [])
    expected_output = bg_plan.get("expected_output", "")

    # 从论文结构中获取对应章节的内容
    section_contents = collect_section_contents(paper_structure, assigned_sections)

    # 构建 focus points 文本
    focus_text = "\n".join(f"- {p}" for p in focus_points) if focus_points else "（无特殊关注点）"

    # 构建用户消息
    user_message = f"""请分析以下论文的研究背景与相关工作：

## 论文信息
- 标题：{paper_structure.get('title', 'Unknown')}

## 重点关注
{focus_text}

## 预期输出
{expected_output}

## 章节内容
{section_contents}

请按照系统提示中的四个维度，进行深度分析。"""

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=BACKGROUND_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"Background Reader 分析完成，输出 {len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"Background Reader 执行失败: {e}")
        return f"## 背景分析失败\n\n错误: {e}\n\n请检查 API 配置。"



