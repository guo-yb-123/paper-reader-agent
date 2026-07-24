"""
对话问答处理器
- 根据 Router 的路由决策，调用对应专家 Agent 回答用户问题
- 每个专家基于自己的分析结果 + 论文原文回答问题
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm


# ============================================================
# 各专家的对话系统提示词
# ============================================================

EXPERT_PROMPTS = {
    "background_expert": """你是论文背景分析专家。你之前已经对这篇论文的研究背景、相关工作、问题动机进行了深度分析。

现在用户向你提问，请基于你的分析知识和论文原文，给出准确、专业的回答。

回答要求：
- 优先引用你之前的分析结论
- 如果问题超出你的分析范围，基于论文原文回答
- 保持客观、学术的语气
- 引用论文中的具体内容来支撑你的回答""",

    "method_expert": """你是论文方法分析专家。你之前已经对这篇论文的模型架构、算法设计、技术实现进行了深度分析。

现在用户向你提问，请基于你的分析知识和论文原文，给出准确、专业的回答。

回答要求：
- 用清晰的技术语言解释复杂的算法和架构
- 如果涉及公式或算法，用文字描述清楚
- 对比前人的方法时，明确指出本质区别
- 引用论文中的具体内容来支撑你的回答""",

    "experiment_expert": """你是论文实验分析专家。你之前已经对这篇论文的实验设计、数据集、评估指标、消融实验进行了深度分析。

现在用户向你提问，请基于你的分析知识和论文原文，给出准确、专业的回答。

回答要求：
- 用数据说话，给出具体的数字和对比结果
- 评估实验设计的合理性时给出明确的判断标准
- 如果发现实验设计问题，直接指出
- 引用论文中的具体内容来支撑你的回答""",

    "critic_expert": """你是论文批判性审阅专家。你之前已经对这篇论文的创新性、实验充分性、局限性进行了批判性分析。

现在用户向你提问，请基于你的审阅意见和论文原文，给出坦诚、犀利的回答。

回答要求：
- 不回避问题，直接指出论文的不足
- 给出有建设性的改进建议
- 如果某个方面确实做得好，也公正地给予肯定
- 引用论文中的具体内容来支撑你的批判观点""",

    "summarizer": """你是论文综述专家。你之前已经整合了各方分析，生成了完整的论文精读报告。

现在用户向你提问，请基于完整的论文理解，给出全面、平衡的回答。

回答要求：
- 从多个维度综合回答（背景、方法、实验）
- 既肯定论文的优点，也不回避其局限
- 给出有建设性的总结和思考
- 引用论文中的具体内容来支撑你的回答""",

    "general": """你是一位热心的学术助手。用户正在阅读一篇论文并提出了问题。

请基于论文内容，给出准确、有帮助的回答。

回答要求：
- 如果论文中有相关信息，直接引用回答
- 如果问题超出论文范围，如实说明
- 保持客观、学术的语气""",
}


# ============================================================
# 专家问答函数
# ============================================================

def answer_question(
    route: str,
    question: str,
    paper_structure: dict,
    analysis_data: dict,
    chat_history: list[dict] = None,
    retrieved_chunks: list[dict] = None,
) -> str:
    """
    根据路由决策，调用对应专家回答用户问题。

    Args:
        route: Router 的路由决策
        question: 用户问题
        paper_structure: 论文结构（含全部章节内容）
        analysis_data: 各专家的分析结果
        chat_history: 对话历史
        retrieved_chunks: pgvector 语义检索结果 [{"content": "...", "section_title": "...", "similarity": 0.95}]

    Returns:
        专家的回答文本
    """
    logger.info(f"问答处理器: route={route}, question='{question[:60]}...'")

    # 获取对应的系统提示词
    system_prompt = EXPERT_PROMPTS.get(route, EXPERT_PROMPTS["general"])

    # 获取该专家的分析结果
    expert_analysis = _get_expert_analysis(route, analysis_data)

    # 获取论文相关内容（优先使用向量检索结果）
    paper_context = _get_paper_context(question, paper_structure, retrieved_chunks)

    # 构建历史
    history_text = ""
    if chat_history:
        recent = chat_history[-8:]
        history_text = "\n".join(
            f"{'用户' if h.get('role') == 'user' else '助手'}: {h.get('content', '')[:300]}"
            for h in recent
        )

    # 构建消息
    user_message = f"""## 你的分析结果
{expert_analysis[:4000] if expert_analysis else "（尚未进行专项分析，请基于论文原文回答）"}

## 论文原文相关内容
{paper_context[:3000]}

## 对话历史
{history_text or "（无历史）"}

## 用户问题
{question}

请基于以上信息，给出准确、专业的回答。"""

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"专家 {route} 回答完成，{len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"专家 {route} 回答失败: {e}")
        return f"抱歉，回答过程中出现错误: {e}"


def _get_expert_analysis(route: str, analysis_data: dict) -> str:
    """根据路由获取对应专家的分析结果"""
    route_to_field = {
        "background_expert": "background_summary",
        "method_expert": "method_summary",
        "experiment_expert": "experiment_summary",
        "critic_expert": "critique",
        "summarizer": "final_report",
    }
    field = route_to_field.get(route)
    if field:
        return analysis_data.get(field, "")
    # general: 返回所有分析
    parts = []
    for key, label in [
        ("background_summary", "背景分析"),
        ("method_summary", "方法分析"),
        ("experiment_summary", "实验分析"),
        ("critique", "批判性审阅"),
    ]:
        val = analysis_data.get(key, "")
        if val:
            parts.append(f"## {label}\n{val[:1000]}")
    return "\n\n".join(parts) if parts else ""


def _get_paper_context(
    question: str,
    paper_structure: dict,
    retrieved_chunks: list[dict] = None,
) -> str:
    """
    获取与用户问题最相关的论文内容。

    优先使用 pgvector 语义检索结果（retrieved_chunks），
    如果向量检索不可用，降级为关键词匹配。

    Args:
        question: 用户问题
        paper_structure: 论文结构
        retrieved_chunks: pgvector 语义检索结果（由 API 层预先检索）

    Returns:
        拼接后的相关文本
    """
    if not paper_structure:
        return "（无论文内容）"

    # ── 优先：使用向量检索结果 ──
    if retrieved_chunks:
        parts = []
        for chunk in retrieved_chunks[:5]:
            similarity = chunk.get("similarity", 0)
            section = chunk.get("section_title", "")
            content = chunk.get("content", "")
            if content.strip():
                relevance_label = "★" if similarity >= 0.8 else "☆"
                parts.append(
                    f"### {relevance_label} {section} (相关度: {similarity:.0%})\n"
                    f"{content[:1500]}\n"
                )
        if parts:
            logger.info(f"使用向量检索结果: {len(parts)} 个文本块")
            return "\n\n".join(parts)

    # ── 降级：关键词匹配 ──
    logger.info("向量检索结果为空，降级为关键词匹配")

    parts = []
    abstract = paper_structure.get("abstract", "")
    if abstract:
        parts.append(f"### Abstract\n{abstract[:1000]}\n")

    sections = paper_structure.get("sections", [])
    question_lower = question.lower()

    for section in sections:
        title = section.get("title", "").lower()
        content = section.get("content", "")
        if any(kw in question_lower or kw in title for kw in
               ["intro", "background", "related", "method", "model", "architecture",
                "experiment", "result", "conclusion", "背景", "方法", "模型", "实验", "结论"]):
            if content and content.strip():
                parts.append(f"### {section.get('title', '')}\n{content[:1500]}\n")
            for child in section.get("children", []):
                child_content = child.get("content", "")
                if child_content.strip():
                    parts.append(f"#### {child.get('title', '')}\n{child_content[:800]}\n")

    if not parts:
        toc = [f"- {s.get('title', '')}" for s in sections]
        parts.append("## 论文章节目录\n" + "\n".join(toc))

    return "\n".join(parts[:5])

