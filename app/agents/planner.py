"""
Planner Agent - 科研论文阅读规划师
- 输入：论文标题、摘要、目录结构
- 输出：JSON 格式的阅读计划，为三个 Reader 分配章节和关注重点
- 角色：制定高效精读策略，确保各 Reader 分工明确、不遗漏重点
"""
import json
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm
from app.services.section_utils import extract_toc


# ============================================================
# Planner 系统提示词
# ============================================================

PLANNER_SYSTEM_PROMPT = """你是一位资深的科研论文阅读规划师。你的任务是为一篇学术论文制定高效的精读策略。

你需要将论文的各章节分配给三个专业的 Reader Agent：

1. **Method Reader（方法阅读器）**：算法工程师视角
   - 负责：方法论、模型架构、算法设计、实现细节
   - 关注：核心创新点、技术方案、数学公式、伪代码、与前人方法的区别

2. **Experiment Reader（实验阅读器）**：实验分析师视角
   - 负责：实验设置、数据集、评估指标、基线对比、消融实验
   - 关注：实验设计合理性、结果有效性、统计显著性、可复现性

3. **Background Reader（背景阅读器）**：领域研究者视角
   - 负责：引言、背景、相关工作、问题定义
   - 关注：研究动机、领域现状、本文定位、贡献总结

请根据论文的结构和内容，制定详细的阅读计划。

## 输出格式
必须严格输出以下 JSON 格式（不要输出其他内容）：

```json
{
  "paper_topic": "一句话概括论文主题",
  "reading_plan": {
    "method_reader": {
      "sections": ["分配给 Method Reader 的章节标题列表"],
      "focus_points": ["重点关注的技术问题列表"],
      "expected_output": "预期的分析输出描述"
    },
    "experiment_reader": {
      "sections": ["分配给 Experiment Reader 的章节标题列表"],
      "focus_points": ["重点关注的实验问题列表"],
      "expected_output": "预期的分析输出描述"
    },
    "background_reader": {
      "sections": ["分配给 Background Reader 的章节标题列表"],
      "focus_points": ["重点关注的背景问题列表"],
      "expected_output": "预期的分析输出描述"
    }
  },
  "overall_strategy": "整体阅读策略说明，包括各 Reader 之间的协作关系"
}
```

## 分配原则
- 每个章节可以分配给多个 Reader（不同视角看同一内容有价值）
- Abstract 应分配给所有三个 Reader
- Method Reader 应重点阅读方法/模型/算法相关章节
- Experiment Reader 应重点阅读实验/评估/结果相关章节
- Background Reader 应重点阅读引言/背景/相关工作章节
- 如果论文结构不标准，请根据章节标题语义灵活判断
"""


# ============================================================
# Planner Agent 函数
# ============================================================

def run_planner(
    paper_title: str,
    abstract: str,
    sections: list[dict],
) -> dict:
    """
    运行 Planner Agent，生成阅读计划。

    Args:
        paper_title: 论文标题
        abstract: 论文摘要
        sections: 章节树列表（一级章节，含 title/level/content/children）

    Returns:
        阅读计划 dict（含 method_reader/experiment_reader/background_reader 分配）
    """
    logger.info("Planner Agent 开始制定阅读计划...")

    # 提取目录（所有章节标题，带层级）
    toc = extract_toc(sections)
    toc_text = "\n".join(f"{'  ' * (item['level'] - 1)}- {item['title']}" for item in toc)

    # 构建用户消息
    user_message = f"""请为以下论文制定精读计划：

## 论文标题
{paper_title}

## 摘要
{abstract[:2000]}

## 章节目录
{toc_text}

请根据上述信息，输出 JSON 格式的阅读计划。"""

    # 调用 LLM
    llm = get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = invoke_llm(llm, messages)
        raw_output = response.content
        logger.info(f"Planner 原始输出长度: {len(raw_output)} 字符")

        # 解析 JSON（处理可能的 markdown 代码块包裹）
        plan = _parse_planner_output(raw_output)
        logger.success("Planner Agent 阅读计划生成完成")
        return plan

    except json.JSONDecodeError as e:
        logger.error(f"Planner 输出 JSON 解析失败: {e}")
        logger.debug(f"原始输出: {raw_output[:500]}")
        # 返回降级计划
        return _fallback_plan(paper_title, sections)
    except Exception as e:
        logger.error(f"Planner Agent 执行失败: {e}")
        return _fallback_plan(paper_title, sections)


def _parse_planner_output(raw: str) -> dict:
    """从 LLM 输出中解析 JSON，兼容 markdown 代码块"""
    # 移除可能的 markdown 代码块标记
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    # 找到 JSON 对象的起止位置
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def _fallback_plan(paper_title: str, sections: list[dict]) -> dict:
    """
    降级计划：当 LLM 输出解析失败时，基于章节标题关键词自动分配。
    """
    logger.warning("使用降级策略生成阅读计划")

    method_keywords = ["method", "model", "architecture", "algorithm", "approach", "framework", "training", "inference", "implementation", "loss", "optimization", "network", "layer", "encoder", "decoder", "transformer", "attention"]
    experiment_keywords = ["experiment", "evaluation", "result", "dataset", "baseline", "ablation", "performance", "comparison", "metric", "accuracy", "benchmark"]
    background_keywords = ["introduction", "background", "related work", "motivation", "contribution", "problem", "survey", "literature"]

    all_titles = extract_toc(sections)

    method_sections = []
    experiment_sections = []
    background_sections = []

    for item in all_titles:
        title_lower = item["title"].lower()
        assigned = False

        for kw in method_keywords:
            if kw in title_lower:
                method_sections.append(item["title"])
                assigned = True
                break

        for kw in experiment_keywords:
            if kw in title_lower:
                experiment_sections.append(item["title"])
                assigned = True
                break

        for kw in background_keywords:
            if kw in title_lower:
                background_sections.append(item["title"])
                assigned = True
                break

        if not assigned:
            # 默认分配给 Background Reader
            background_sections.append(item["title"])

    return {
        "paper_topic": paper_title,
        "reading_plan": {
            "method_reader": {
                "sections": method_sections or [item["title"] for item in all_titles],
                "focus_points": ["核心算法与模型架构", "技术实现细节", "与前人方法的区别"],
                "expected_output": "方法部分结构化总结"
            },
            "experiment_reader": {
                "sections": experiment_sections or [item["title"] for item in all_titles],
                "focus_points": ["数据集与评估指标", "基线对比结果", "消融实验结论"],
                "expected_output": "实验部分结构化总结"
            },
            "background_reader": {
                "sections": background_sections or [item["title"] for item in all_titles],
                "focus_points": ["研究问题与动机", "领域现状与相关工作", "本文定位与贡献"],
                "expected_output": "背景与相关工作总结"
            }
        },
        "overall_strategy": "（降级模式：基于关键词自动分配章节）"
    }
