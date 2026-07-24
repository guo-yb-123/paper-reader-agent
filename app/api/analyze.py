"""
分析接口
- POST /api/analyze/{paper_id}：触发异步论文分析任务
- GET /api/task/{task_id}：查询任务状态与进度
- GET /api/report/{paper_id}：获取分析报告
"""
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.core.database import get_db
from app.schemas.models import (
    Paper, AnalysisReport,
    AnalyzeResponse, TaskStatusResponse, TaskStatus, ReportResponse,
)
from app.tasks.celery_app import celery_app

router = APIRouter()


@router.post("/analyze/{paper_id}", response_model=AnalyzeResponse, summary="触发论文分析")
async def trigger_analysis(paper_id: str, db: Session = Depends(get_db)):
    """
    触发异步论文分析任务。

    - 检查论文是否存在
    - 提交 Celery 异步任务
    - 返回 task_id 用于追踪任务进度
    """
    # 检查论文
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    if paper.status in ("analyzing", "parsing"):
        raise HTTPException(status_code=400, detail="论文正在分析中，请勿重复提交")
    if paper.status == "completed":
        raise HTTPException(status_code=400, detail="论文已分析完成，请查看报告")

    if not paper.pdf_path or not os.path.exists(paper.pdf_path):
        raise HTTPException(status_code=400, detail="PDF 文件丢失，请重新上传")

    # 提交异步任务（状态由 Celery 任务内部更新）
    task = celery_app.send_task(
        "analyze_paper",
        args=[paper_id],
        kwargs={},
    )

    logger.info(f"论文 {paper_id} 分析任务已提交: task_id={task.id}")
    return AnalyzeResponse(
        paper_id=paper_id,
        task_id=task.id,
        status="PENDING",
        message="分析任务已提交，请通过 task_id 查询进度",
    )


@router.get("/task/{task_id}", response_model=TaskStatusResponse, summary="查询任务状态")
async def get_task_status(task_id: str):
    """
    查询 Celery 任务的执行状态和进度。

    状态包括：PENDING / PROCESSING / SUCCESS / FAILURE
    进度百分比见 progress 字段。
    """
    try:
        result = celery_app.AsyncResult(task_id)
    except Exception:
        raise HTTPException(status_code=404, detail="任务不存在")

    response_data = {
        "task_id": task_id,
        "paper_id": "",
        "status": TaskStatus.PENDING,
        "progress": 0,
        "message": "任务等待中...",
        "error": None,
        "result": None,
    }

    if result.state == "PENDING":
        pass
    elif result.state == "PROCESSING":
        info = result.info or {}
        response_data["status"] = TaskStatus.PROCESSING
        response_data["progress"] = info.get("progress", 0)
        response_data["message"] = info.get("message", "正在处理...")
        response_data["paper_id"] = info.get("paper_id", "")
    elif result.state == "SUCCESS":
        response_data["status"] = TaskStatus.SUCCESS
        response_data["progress"] = 100
        response_data["message"] = "分析完成"
        response_data["result"] = result.result
        if isinstance(result.result, dict):
            response_data["paper_id"] = result.result.get("paper_id", "")
    elif result.state == "FAILURE":
        response_data["status"] = TaskStatus.FAILURE
        response_data["error"] = str(result.info) if result.info else "未知错误"
        response_data["message"] = "分析失败"
    else:
        response_data["status"] = TaskStatus.PROCESSING
        response_data["message"] = f"未知状态: {result.state}"

    return TaskStatusResponse(**response_data)


@router.get("/report/{paper_id}", response_model=ReportResponse, summary="获取分析报告")
async def get_report(paper_id: str, db: Session = Depends(get_db)):
    """
    获取论文的完整分析报告。

    如果分析尚未完成，返回当前状态。
    如果分析完成，返回完整报告（final_report 为 Markdown 格式）。
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    report = db.query(AnalysisReport).filter(
        AnalysisReport.paper_id == paper_id
    ).first()

    response = ReportResponse(
        paper_id=paper_id,
        title=paper.title or paper.original_filename,
        status=paper.status,
    )

    if report:
        response.reading_plan = report.reading_plan
        response.method_summary = report.method_summary
        response.experiment_summary = report.experiment_summary
        response.background_summary = report.background_summary
        response.critique = report.critique
        response.final_report = report.final_report

    return response
