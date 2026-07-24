"""
Experiment Reader Agent - 实验分析师视角
- 输入：实验相关章节内容
- 输出：实验结果结构化总结（数据集、评价指标、基线对比、消融实验结论）
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm
from app.services.section_utils import collect_section_contents

EXPERIMENT_SYSTEM_PROMPT = """你是一位严谨的实验分析师，专门负责评估学术论文的实验设计与结果可信度。

你的任务是精读论文的实验部分，从实验设计的科学性和结果的有效性角度进行深度评估：

## 分析维度

### 1. 实验设置
- 使用了哪些数据集？数据集规模和特点是什么？
- 数据预处理和划分方式是怎样的？
- 评价指标有哪些？选择这些指标是否合理？
- 硬件环境和超参数设置如何？

### 2. 基线对比
- 对比了哪些基线方法？基线选择是否全面、公平？
- 主要结果是什么？性能提升幅度有多大？
- 提升是否具有统计显著性？有没有做显著性检验？
- 结果在不同的指标/数据集上是否一致？

### 3. 消融实验
- 做了哪些消融实验？每个消融实验验证了什么假设？
- 各组件对最终性能的贡献如何？
- 消融结论是否合理？有没有遗漏重要的消融项？
- 作者如何解释消融结果的因果逻辑？

### 4. 结果可信度评估
- 实验设计是否存在缺陷或偏差？
- 结果是否支撑论文的核心主张？
- 是否存在 cherry-picking（选择性报告结果）的嫌疑？
- 实验的可复现性如何？是否提供了足够的复现信息？

## 输出格式
请用清晰的中文输出，包含以上四个维度的分析。
使用 Markdown 格式，包含表格对比实验数据时请使用 Markdown 表格。
对可信度存疑的地方，请明确指出并说明原因。"""


def run_experiment_reader(
    paper_structure: dict,
    reading_plan: dict,
) -> str:
    """
    Experiment Reader：精读论文实验部分。

    Args:
        paper_structure: 解析后的论文结构
        reading_plan: Planner 输出的阅读计划

    Returns:
        实验结果结构化总结（Markdown 格式文本）
    """
    logger.info("Experiment Reader 开始分析...")

    exp_plan = reading_plan.get("reading_plan", {}).get("experiment_reader", {})
    assigned_sections = exp_plan.get("sections", [])
    focus_points = exp_plan.get("focus_points", [])
    expected_output = exp_plan.get("expected_output", "")

    section_contents = collect_section_contents(paper_structure, assigned_sections)

    focus_text = "\n".join(f"- {p}" for p in focus_points) if focus_points else "（无特殊关注点）"

    user_message = f"""请分析以下论文的实验部分：

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
            SystemMessage(content=EXPERIMENT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"Experiment Reader 分析完成，输出 {len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"Experiment Reader 执行失败: {e}")
        return f"## 实验分析失败\n\n错误: {e}\n\n请检查 API 配置。"
