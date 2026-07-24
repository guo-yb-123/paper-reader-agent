"""
对话问答接口
- POST /api/chat/{paper_id}：向论文提问，返回专家回答
- GET /api/chat/{paper_id}/history：获取对话历史
"""
import json
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.core.database import get_db
from app.schemas.models import (
    Paper, AnalysisReport, ChatHistory,
    ChatRequest, ChatResponse,
)
from app.graph.workflow import build_chat_workflow
from app.services.vector_store import search_similar_chunks

router = APIRouter()


@router.post("/chat/{paper_id}", response_model=ChatResponse, summary="论文问答")
async def chat_with_paper(
    paper_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    向论文提问，由合适的专家 Agent 回答。

    - Router Agent 自动判断问题类型并路由到对应专家
    - 支持多轮对话（通过 history 字段传递历史）
    - 返回专家回答和路由决策信息
    """
    # 检查论文
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    if paper.status not in ["completed", "analyzing"]:
        raise HTTPException(
            status_code=400,
            detail=f"论文尚未分析（当前状态: {paper.status}），请先触发分析",
        )

    # 获取解析结果
    parsed_json_path = paper.parsed_json_path
    paper_structure = {}
    if parsed_json_path:
        if os.path.exists(parsed_json_path):
            with open(parsed_json_path, "r", encoding="utf-8") as f:
                paper_structure = json.load(f)
        else:
            logger.warning(f"解析缓存文件不存在: {parsed_json_path}")

    # 获取分析报告
    report = db.query(AnalysisReport).filter(
        AnalysisReport.paper_id == paper_id
    ).first()

    # ── 语义检索：使用 pgvector 查找与问题最相关的文本块 ──
    retrieved_chunks = []
    try:
        retrieved_chunks = search_similar_chunks(
            query=request.question,
            paper_id=paper_id,
            db=db,
            top_k=5,
        )
        logger.info(f"向量检索返回 {len(retrieved_chunks)} 条相关文本块")
    except Exception as e:
        logger.warning(f"向量检索失败，降级使用全文匹配: {e}")
        # 检索失败不阻塞对话，降级为空列表（handler 会使用 paper_structure 兜底）

    # 构建对话状态
    chat_state = {
        "paper_id": paper_id,
        "paper_structure": paper_structure or {
            "title": paper.title,
            "abstract": paper.abstract,
            "sections": [],
        },
        "background_summary": report.background_summary if report else "",
        "method_summary": report.method_summary if report else "",
        "experiment_summary": report.experiment_summary if report else "",
        "critique": report.critique if report else "",
        "final_report": report.final_report if report else "",
        "current_question": request.question,
        "messages": request.history or [],
        "retrieved_chunks": retrieved_chunks,
    }

    # 运行对话工作流
    try:
        chat_workflow = build_chat_workflow()
        app = chat_workflow.compile()
        result = app.invoke(chat_state, {"recursion_limit": 10})

        answer = result.get("answer", "抱歉，暂时无法回答这个问题。")
        router_decision = result.get("router_decision", {})

        # 保存对话历史
        db.add(ChatHistory(
            paper_id=paper_id,
            role="user",
            content=request.question,
            router_decision=None,
        ))
        db.add(ChatHistory(
            paper_id=paper_id,
            role="assistant",
            content=answer,
            router_decision=router_decision.get("route", ""),
        ))
        db.commit()

        return ChatResponse(
            paper_id=paper_id,
            question=request.question,
            answer=answer,
            router_decision=router_decision.get("route", ""),
        )

    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话处理失败: {e}")


@router.get("/chat/{paper_id}/history", summary="获取对话历史")
async def get_chat_history(paper_id: str, db: Session = Depends(get_db)):
    """获取指定论文的对话历史记录"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    history = (
        db.query(ChatHistory)
        .filter(ChatHistory.paper_id == paper_id)
        .order_by(ChatHistory.created_at.asc())
        .limit(100)
        .all()
    )

    return {
        "paper_id": paper_id,
        "history": [
            {
                "id": str(h.id),
                "role": h.role,
                "content": h.content,
                "router_decision": h.router_decision,
                "created_at": h.created_at.isoformat() if h.created_at else "",
            }
            for h in history
        ],
        "total": len(history),
    }
