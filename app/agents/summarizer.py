"""
Summarizer Agent - 科技编辑
- 输入：三个 Reader 总结 + Critic 批判
- 输出：完整的 Markdown 格式精读报告
- 角色：整合所有分析内容，生成统一格式的最终报告
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm


SUMMARIZER_SYSTEM_PROMPT = """你是一位资深科技编辑，专门负责将多位专家的分析整合为一份完整、易读的论文精读报告。

你的输入包括：
- 背景分析（Background Reader）
- 方法分析（Method Reader）
- 实验分析（Experiment Reader）
- 批判性审阅（Critic）

你需要将这些内容整合为一份结构化的 Markdown 精读报告。

## 报告结构要求

请严格按照以下章节组织报告：

### 📄 论文信息
- 标题、作者（如有）、发表信息

### 📌 一句话速览
用一句话概括这篇论文做了什么、效果如何。

### 🎯 研究背景与动机
整合 Background Reader 的分析，概述：
- 研究问题是什么
- 为什么这个问题重要
- 领域现状和本文定位

### 🔧 方法与技术
整合 Method Reader 的分析，概述：
- 核心方法 / 模型架构
- 关键技术创新点
- 技术实现要点

### 📊 实验与评估
整合 Experiment Reader 的分析，概述：
- 实验设置（数据集、指标、基线）
- 主要结果与对比
- 消融实验关键结论

### ⚠️ 批判性审视
整合 Critic 的审阅意见，概述：
- 论文的主要局限性和潜在问题
- 实验设计的不足
- 声明与证据的匹配度

### 📝 总结与思考
- 论文的核心贡献总结
- 对该方法的个人思考
- 值得跟进的相关工作或改进方向

## 格式要求
- 全文使用 Markdown 格式
- 适当使用表格、列表、引用等元素增强可读性
- 关键结论用粗体标注
- 保持专业、客观的学术语气
- 报告长度控制在 2000-4000 字（中文）"""


def run_summarizer(
    background_summary: str,
    method_summary: str,
    experiment_summary: str,
    critique: str,
    paper_title: str = "",
    paper_info: dict | None = None,
) -> str:
    """
    Summarizer Agent：整合所有分析，生成最终精读报告。

    Args:
        background_summary: Background Reader 的分析
        method_summary: Method Reader 的分析
        experiment_summary: Experiment Reader 的分析
        critique: Critic Agent 的批判性审阅
        paper_title: 论文标题
        paper_info: 论文基本信息（可选，含作者、摘要等）

    Returns:
        完整的 Markdown 格式精读报告
    """
    logger.info("Summarizer Agent 开始整合最终报告...")

    # 控制各部分的输入长度
    def trim(text: str, limit: int = 3000) -> str:
        return text[:limit] + ("...(截断)" if len(text) > limit else "")

    bg_text = trim(background_summary, 2500)
    method_text = trim(method_summary, 2500)
    exp_text = trim(experiment_summary, 2500)
    critique_text = trim(critique, 2000)

    # 构建论文信息段
    info_lines = [f"- 标题：**{paper_title or '未提供'}**"]
    if paper_info:
        if paper_info.get("authors"):
            info_lines.append(f"- 作者：{paper_info['authors']}")
        if paper_info.get("abstract"):
            abstract_short = paper_info["abstract"][:500]
            info_lines.append(f"- 摘要：{abstract_short}")
    paper_info_text = "\n".join(info_lines)

    user_message = f"""请将以下四份分析报告整合为一份完整的论文精读报告：

## 论文基本信息
{paper_info_text}

---

## 1. 背景分析（Background Reader）
{bg_text}

---

## 2. 方法分析（Method Reader）
{method_text}

---

## 3. 实验分析（Experiment Reader）
{exp_text}

---

## 4. 批判性审阅（Critic）
{critique_text}

---

请按照系统提示中的报告结构，整合以上内容，生成一份完整、专业的 Markdown 格式精读报告。"""

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"Summarizer Agent 报告生成完成，输出 {len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"Summarizer Agent 执行失败: {e}")
        return f"## 精读报告生成失败\n\n错误: {e}"
