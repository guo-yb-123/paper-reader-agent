"""
Router Agent - 对话路由分发器
- 输入：用户问题 + 论文上下文
- 输出：路由决策（将问题分配给最合适的 Agent 回答）
- 角色：理解用户意图，精准匹配专家
"""
import json
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm


ROUTER_SYSTEM_PROMPT = """你是一个智能对话路由器，负责分析用户关于论文的问题，并将其路由到最合适的专家 Agent。

## 可用的专家 Agent

1. **background_expert**（背景专家）
   - 擅长：研究背景、相关工作、问题动机、领域现状、论文贡献定位
   - 匹配问题：这篇论文的研究背景是什么？为什么要做这个研究？相关工作有哪些？该领域的现状如何？

2. **method_expert**（方法专家）
   - 擅长：模型架构、算法设计、技术实现、数学原理、核心创新
   - 匹配问题：这个方法是怎么实现的？模型结构是什么？用了什么算法？有什么创新点？

3. **experiment_expert**（实验专家）
   - 擅长：实验设置、数据集、评估指标、基线对比、消融实验、结果可信度
   - 匹配问题：实验用了什么数据集？效果怎么样？和哪些方法做了对比？消融实验说明了什么？

4. **critic_expert**（批判专家）
   - 擅长：指出论文问题、评估创新性、质疑实验充分性、分析局限性
   - 匹配问题：这篇论文有什么问题？方法有什么缺陷？实验是否充分？结论是否可靠？

5. **summarizer**（综述专家）
   - 擅长：整体总结、快速概览、多维度综合回答
   - 匹配问题：总结一下这篇论文？这篇论文讲了什么？给我一个概览？

## 路由规则
- 如果问题涉及多个维度，选择最核心的维度
- 如果问题非常宽泛（如"这篇论文怎么样"），路由到 summarizer
- 如果问题与论文完全无关，路由到 "general"

## 输出格式
必须严格输出 JSON：
```json
{
  "route": "background_expert | method_expert | experiment_expert | critic_expert | summarizer | general",
  "reason": "简短说明路由原因",
  "confidence": "high | medium | low"
}
```"""


def run_router(
    question: str,
    paper_structure: dict = None,
    chat_history: list[dict] = None,
) -> dict:
    """
    分析用户问题并确定路由目标。

    Args:
        question: 用户当前问题
        paper_structure: 论文结构（提供上下文）
        chat_history: 历史对话（可选）

    Returns:
        {"route": "...", "reason": "...", "confidence": "..."}
    """
    logger.info(f"Router Agent: 分析问题 '{question[:80]}...'")

    # 构建上下文
    context_parts = []
    if paper_structure:
        title = paper_structure.get("title", "")
        if title:
            context_parts.append(f"论文标题: {title}")
        sections = paper_structure.get("sections", [])
        if sections:
            toc = ", ".join(s.get("title", "") for s in sections[:10])
            context_parts.append(f"章节: {toc}")

    context = "\n".join(context_parts) if context_parts else "（无上下文）"

    # 历史对话（取最近3轮）
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # 最多3轮(6条)
        history_text = "\n".join(
            f"{'用户' if h.get('role') == 'user' else '助手'}: {h.get('content', '')[:200]}"
            for h in recent
        )

    user_message = f"""## 论文上下文
{context}

## 对话历史
{history_text or "（无历史）"}

## 用户问题
{question}

请判断该问题应路由到哪个专家 Agent。"""

    try:
        llm = get_llm(temperature=0)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        raw = response.content.strip()

        # 解析 JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        logger.info(f"Router 决策: {result.get('route')} (信心: {result.get('confidence')})")
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Router JSON 解析失败: {e}，使用关键词降级路由")

        # 降级：基于关键词的简单路由（中英双语）
        question_lower = question.lower()
        if any(kw in question_lower for kw in [
            "背景", "动机", "相关", "现状", "为什么",
            "background", "motivation", "related", "why", "context", "intro",
        ]):
            route = "background_expert"
        elif any(kw in question_lower for kw in [
            "方法", "模型", "算法", "架构", "实现", "怎么", "如何",
            "method", "architecture", "algorithm", "how", "approach", "framework",
        ]):
            route = "method_expert"
        elif any(kw in question_lower for kw in [
            "实验", "数据", "结果", "评估", "对比", "消融",
            "experiment", "result", "dataset", "ablation", "baseline", "evaluation", "performance",
        ]):
            route = "experiment_expert"
        elif any(kw in question_lower for kw in [
            "问题", "缺陷", "局限", "不足", "批判",
            "critic", "limitation", "weakness", "flaw", "drawback", "issue",
        ]):
            route = "critic_expert"
        elif any(kw in question_lower for kw in [
            "总结", "概括", "概览",
            "summarize", "overview", "summary", "tldr",
        ]):
            route = "summarizer"
        else:
            route = "general"

        return {"route": route, "reason": "关键词降级路由", "confidence": "low"}
