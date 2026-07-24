"""
LangGraph 工作流组装模块
- 分析工作流：Planner → 并行 Readers → Critic → Summarizer
- 对话工作流：Router → 对应 Agent → 回答
"""
from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from app.core.state import AnalysisState, ChatState


# ============================================================
# 节点函数
# ============================================================

def planner_node(state: AnalysisState) -> dict:
    """
    Planner 节点：制定论文阅读计划。
    分析工作流的入口节点。
    """
    from app.agents.planner import run_planner

    paper_structure = state.get("paper_structure") or {}
    logger.info(f"[Workflow] Planner 节点开始, paper_id={state.get('paper_id')}")

    try:
        title = paper_structure.get("title", "Unknown")
        abstract = paper_structure.get("abstract", "")
        sections = paper_structure.get("sections", [])

        reading_plan = run_planner(
            paper_title=title,
            abstract=abstract,
            sections=sections,
        )

        return {"reading_plan": reading_plan}

    except Exception as e:
        logger.error(f"[Workflow] Planner 节点失败: {e}")
        return {"error": str(e)}


def background_reader_node(state: AnalysisState) -> dict:
    """
    Background Reader 节点：精读背景与相关工作。
    并行分支之一，由 Planner 分配章节后执行。
    """
    from app.agents.background_reader import run_background_reader

    logger.info("[Workflow] Background Reader 节点开始")

    try:
        paper_structure = state.get("paper_structure") or {}
        reading_plan = state.get("reading_plan", {})

        result = run_background_reader(paper_structure, reading_plan)
        return {"background_summary": result}

    except Exception as e:
        logger.error(f"[Workflow] Background Reader 节点失败: {e}")
        return {"background_summary": f"分析失败: {e}"}


def method_reader_node(state: AnalysisState) -> dict:
    """
    Method Reader 节点：精读方法论部分。
    并行分支之二。
    """
    from app.agents.method_reader import run_method_reader

    logger.info("[Workflow] Method Reader 节点开始")

    try:
        paper_structure = state.get("paper_structure") or {}
        reading_plan = state.get("reading_plan", {})

        result = run_method_reader(paper_structure, reading_plan)
        return {"method_summary": result}

    except Exception as e:
        logger.error(f"[Workflow] Method Reader 节点失败: {e}")
        return {"method_summary": f"分析失败: {e}"}


def experiment_reader_node(state: AnalysisState) -> dict:
    """
    Experiment Reader 节点：精读实验部分。
    并行分支之三。
    """
    from app.agents.experiment_reader import run_experiment_reader

    logger.info("[Workflow] Experiment Reader 节点开始")

    try:
        paper_structure = state.get("paper_structure") or {}
        reading_plan = state.get("reading_plan", {})

        result = run_experiment_reader(paper_structure, reading_plan)
        return {"experiment_summary": result}

    except Exception as e:
        logger.error(f"[Workflow] Experiment Reader 节点失败: {e}")
        return {"experiment_summary": f"分析失败: {e}"}


def critic_node(state: AnalysisState) -> dict:
    """
    Critic 节点：批判性审阅三个 Reader 的总结。
    汇聚节点：等三个 Reader 都完成后执行。
    """
    from app.agents.critic import run_critic

    logger.info("[Workflow] Critic 节点开始")

    try:
        paper_structure = state.get("paper_structure") or {}
        title = paper_structure.get("title", "")
        background_summary = state.get("background_summary", "")
        method_summary = state.get("method_summary", "")
        experiment_summary = state.get("experiment_summary", "")

        if not any([background_summary, method_summary, experiment_summary]):
            logger.warning("[Workflow] Critic: 所有 Reader 输出为空，跳过审阅")
            return {"critique": "（Reader 分析结果为空，无法进行批判性审阅）"}

        critique = run_critic(
            background_summary=background_summary,
            method_summary=method_summary,
            experiment_summary=experiment_summary,
            paper_title=title,
        )
        return {"critique": critique}

    except Exception as e:
        logger.error(f"[Workflow] Critic 节点失败: {e}")
        return {"critique": f"批判性审阅失败: {e}"}


def summarizer_node(state: AnalysisState) -> dict:
    """
    Summarizer 节点：整合所有输出，生成最终 Markdown 报告。
    """
    from app.agents.summarizer import run_summarizer

    logger.info("[Workflow] Summarizer 节点开始")

    try:
        paper_structure = state.get("paper_structure") or {}
        title = paper_structure.get("title", "")
        background_summary = state.get("background_summary", "")
        method_summary = state.get("method_summary", "")
        experiment_summary = state.get("experiment_summary", "")
        critique = state.get("critique", "")

        if not any([background_summary, method_summary, experiment_summary]):
            logger.warning("[Workflow] Summarizer: 所有 Reader 输出为空")
            return {"final_report": "（分析数据不足，无法生成报告）"}

        final_report = run_summarizer(
            background_summary=background_summary,
            method_summary=method_summary,
            experiment_summary=experiment_summary,
            critique=critique,
            paper_title=title,
            paper_info={
                "authors": paper_structure.get("authors", ""),
                "abstract": paper_structure.get("abstract", ""),
            },
        )
        return {"final_report": final_report}

    except Exception as e:
        logger.error(f"[Workflow] Summarizer 节点失败: {e}")
        return {"final_report": f"报告生成失败: {e}"}


# ============================================================
# 路由函数
# ============================================================

def after_planner_router(state: AnalysisState) -> list:
    """
    Planner 之后的路由逻辑。
    使用 LangGraph Send API 实现三个 Reader 的并行分发。

    注意：Send 的第二个参数是传递给目标节点的状态更新。
    目标节点会继承当前累积状态 + 此更新。
    """
    if state.get("error"):
        logger.error(f"[Workflow] Planner 出错，终止流程: {state['error']}")
        return []

    logger.info("[Workflow] Planner 完成，并行分发到三个 Reader")

    # 将完整状态传给每个 Reader（确保 paper_structure 和 reading_plan 不丢失）
    return [
        Send("background_reader", {
            "paper_structure": state.get("paper_structure"),
            "reading_plan": state.get("reading_plan"),
        }),
        Send("method_reader", {
            "paper_structure": state.get("paper_structure"),
            "reading_plan": state.get("reading_plan"),
        }),
        Send("experiment_reader", {
            "paper_structure": state.get("paper_structure"),
            "reading_plan": state.get("reading_plan"),
        }),
    ]


# ============================================================
# 工作流构建
# ============================================================

def build_analysis_workflow() -> StateGraph:
    """
    构建论文分析工作流。

    流程：
    planner_node
        ├─[Send]→ background_reader  ─┐
        ├─[Send]→ method_reader       ─┤─→ critic_node → summarizer_node → END
        └─[Send]→ experiment_reader   ─┘

    全部 6 个 Agent 已实现：
    - Planner ✅
    - 并行 Readers (Background / Method / Experiment) ✅
    - Critic ✅
    - Summarizer ✅
    """
    workflow = StateGraph(AnalysisState)

    # 添加所有节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("background_reader", background_reader_node)
    workflow.add_node("method_reader", method_reader_node)
    workflow.add_node("experiment_reader", experiment_reader_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("summarizer", summarizer_node)

    # 入口
    workflow.set_entry_point("planner")

    # Planner → 并行 Readers（通过 Send fan-out）
    workflow.add_conditional_edges(
        "planner",
        after_planner_router,
    )

    # 三个 Reader 汇聚到 Critic
    # （LangGraph 自动处理汇聚：所有 Send 目标都完成后才进入 critic）
    workflow.add_edge("background_reader", "critic")
    workflow.add_edge("method_reader", "critic")
    workflow.add_edge("experiment_reader", "critic")

    # Critic → Summarizer → END
    workflow.add_edge("critic", "summarizer")
    workflow.add_edge("summarizer", END)

    logger.info("分析工作流构建完成：Planner → 并行 Readers → Critic → Summarizer")
    return workflow


# ============================================================
# 对话工作流节点
# ============================================================

def router_node(state: ChatState) -> dict:
    """
    Router 节点：分析用户问题，决定路由到哪个专家。
    """
    from app.agents.router import run_router

    logger.info("[Chat Workflow] Router 节点开始")

    try:
        question = state.get("current_question", "")
        paper_structure = state.get("paper_structure") or {}
        messages = state.get("messages", [])

        # 从 messages 中提取对话历史
        chat_history = [
            {"role": m.get("role", m.type if hasattr(m, 'type') else "user"),
             "content": m.get("content", str(m))}
            for m in (messages or [])
        ]

        router_decision = run_router(
            question=question,
            paper_structure=paper_structure,
            chat_history=chat_history,
        )
        return {"router_decision": router_decision}

    except Exception as e:
        logger.error(f"[Chat Workflow] Router 节点失败: {e}")
        return {"router_decision": {"route": "general", "reason": str(e), "confidence": "low"}}


def _make_expert_node(route: str, label: str):
    """
    工厂函数：为指定专家创建 LangGraph 节点函数。
    每个专家节点结构相同但硬编码其专长路由，确保路由分发真正生效。
    """
    def expert_node(state: ChatState) -> dict:
        from app.agents.chat_handler import answer_question

        logger.info(f"[Chat Workflow] {label} 节点开始 (route={route})")

        try:
            question = state.get("current_question", "")
            paper_structure = state.get("paper_structure") or {}

            analysis_data = {
                "background_summary": state.get("background_summary", ""),
                "method_summary": state.get("method_summary", ""),
                "experiment_summary": state.get("experiment_summary", ""),
                "critique": state.get("critique", ""),
                "final_report": state.get("final_report", ""),
            }

            messages = state.get("messages", [])
            chat_history = [
                {"role": m.get("role", m.type if hasattr(m, 'type') else "user"),
                 "content": m.get("content", str(m))}
                for m in (messages or [])
            ]

            retrieved_chunks = state.get("retrieved_chunks", [])

            answer = answer_question(
                route=route,
                question=question,
                paper_structure=paper_structure,
                analysis_data=analysis_data,
                chat_history=chat_history,
                retrieved_chunks=retrieved_chunks,
            )
            return {"answer": answer}

        except Exception as e:
            logger.error(f"[Chat Workflow] {label} 节点失败: {e}")
            return {"answer": f"抱歉，回答失败: {e}"}

    return expert_node


# 为每个专家创建独立节点（chat_ 前缀避免与分析工作流节点的变量名冲突）
chat_background_expert_node = _make_expert_node("background_expert", "Background Expert")
chat_method_expert_node = _make_expert_node("method_expert", "Method Expert")
chat_experiment_expert_node = _make_expert_node("experiment_expert", "Experiment Expert")
chat_critic_expert_node = _make_expert_node("critic_expert", "Critic Expert")
chat_summarizer_node = _make_expert_node("summarizer", "Summarizer")
chat_general_node = _make_expert_node("general", "General")


def after_router(state: ChatState) -> str:
    """
    Router 之后的路由分发：根据 Router 决策将问题分发给对应专家节点。

    路由映射:
      background_expert → background_expert_node
      method_expert      → method_expert_node
      experiment_expert  → experiment_expert_node
      critic_expert      → critic_expert_node
      summarizer         → summarizer_node
      general / 其他      → general_node
    """
    decision = state.get("router_decision", {})
    route = decision.get("route", "general")
    logger.info(f"[Chat Workflow] 路由决策: {route} → 对应专家节点")
    return route


def build_chat_workflow() -> StateGraph:
    """
    构建对话问答工作流。

    流程:
    router_node
        ├─→ chat_background_expert_node → END
        ├─→ chat_method_expert_node      → END
        ├─→ chat_experiment_expert_node  → END
        ├─→ chat_critic_expert_node      → END
        ├─→ chat_summarizer_node         → END
        └─→ chat_general_node            → END

    每个专家节点是独立的 LangGraph node，Router 通过 conditional_edges
    将问题分发给对应专家，不同专家使用不同的系统提示词回答问题。
    """
    workflow = StateGraph(ChatState)

    # 注册所有节点
    workflow.add_node("router", router_node)
    workflow.add_node("background_expert", chat_background_expert_node)
    workflow.add_node("method_expert", chat_method_expert_node)
    workflow.add_node("experiment_expert", chat_experiment_expert_node)
    workflow.add_node("critic_expert", chat_critic_expert_node)
    workflow.add_node("summarizer", chat_summarizer_node)
    workflow.add_node("general", chat_general_node)

    workflow.set_entry_point("router")

    # Router → 各专家（条件路由）
    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "background_expert": "background_expert",
            "method_expert": "method_expert",
            "experiment_expert": "experiment_expert",
            "critic_expert": "critic_expert",
            "summarizer": "summarizer",
            "general": "general",
        },
    )

    # 所有专家节点 → END
    workflow.add_edge("background_expert", END)
    workflow.add_edge("method_expert", END)
    workflow.add_edge("experiment_expert", END)
    workflow.add_edge("critic_expert", END)
    workflow.add_edge("summarizer", END)
    workflow.add_edge("general", END)

    logger.info("对话工作流构建完成：Router → 6 个独立专家节点")
    return workflow
