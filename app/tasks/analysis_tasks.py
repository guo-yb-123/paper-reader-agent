"""
分析任务定义模块
- 定义 Celery 异步任务：完整的论文分析流水线
- 任务状态追踪与进度上报
"""
import os
from app.tasks.celery_app import celery_app
from app.core.logger import logger
from app.core.database import _get_session_local
from app.core.config import settings
from app.schemas.models import Paper, AnalysisReport
from app.services.pdf_parser import parse_pdf
from app.services.vector_store import index_paper
from app.services.converters import parsed_paper_to_dict


@celery_app.task(bind=True, name="analyze_paper")
def analyze_paper_task(self, paper_id: str) -> dict:
    """
    论文分析异步任务 — 完整的 LangGraph 分析流水线。

    流程:
    1. 解析 PDF → 结构化论文
    2. 运行 LangGraph 工作流 (Planner → 并行 Readers → Critic → Summarizer)
    3. 将结果持久化到 AnalysisReport 表
    4. 更新 paper.status

    Args:
        paper_id: 论文 UUID

    Returns:
        包含分析结果的字典
    """
    logger.info(f"[Task] 开始分析论文 {paper_id}，Celery task_id: {self.request.id}")

    db = None
    try:
        db = _get_session_local()()
        # ── 阶段 1: 获取论文记录 + 幂等保护 ──
        paper = db.query(Paper).filter(Paper.id == paper_id).first()
        if not paper:
            raise ValueError(f"论文不存在: {paper_id}")

        if paper.status == "analyzing":
            logger.warning(f"[Task] 论文 {paper_id} 正在分析中，跳过重复执行")
            return {
                "paper_id": paper_id,
                "status": "skipped",
                "message": "论文正在分析中，跳过重复触发",
            }

        pdf_path = paper.pdf_path
        if not pdf_path:
            raise ValueError(f"论文 {paper_id} 没有关联的 PDF 文件")

        # 标记为分析中（防止并发重复执行）
        paper.status = "analyzing"
        db.commit()

        # ── 阶段 2: PDF 解析 ──
        self.update_state(
            state="PROCESSING",
            meta={"progress": 5, "message": "正在解析 PDF...", "paper_id": paper_id},
        )

        parsed = parse_pdf(paper_id, pdf_path)
        paper_structure = parsed_paper_to_dict(parsed)

        # 回填论文元信息 + 缓存路径（C-1 修复：让聊天流程能加载论文结构）
        paper.title = parsed.title or paper.title
        paper.abstract = parsed.abstract or paper.abstract
        paper.page_count = parsed.page_count
        paper.parsed_json_path = os.path.join(
            settings.parsed_storage_dir, f"{paper_id}.json"
        )
        paper.status = "parsed"
        db.commit()

        logger.info(f"[Task] PDF 解析完成: {parsed.page_count} 页, "
                    f"标题: {parsed.title[:80] if parsed.title else 'N/A'}")

        # ── 阶段 3: LangGraph 分析工作流 ──
        paper.status = "analyzing"
        db.commit()

        from app.graph.workflow import build_analysis_workflow

        initial_state = {
            "paper_id": paper_id,
            "pdf_path": pdf_path,
            "paper_structure": paper_structure,
        }

        workflow = build_analysis_workflow()
        app = workflow.compile()

        self.update_state(
            state="PROCESSING",
            meta={"progress": 20, "message": "Planner Agent 正在制定阅读计划...", "paper_id": paper_id},
        )

        # 执行工作流（同步调用，因为 LangGraph 内部会管理状态流转）
        result = app.invoke(initial_state, {"recursion_limit": 50})

        # 检查是否有错误
        if result.get("error"):
            logger.error(f"[Task] 工作流执行出错: {result['error']}")
            raise RuntimeError(result["error"])

        logger.info(f"[Task] LangGraph 工作流执行完成")

        # ── 阶段 4: 向量化入库（C-2 修复）──
        self.update_state(
            state="PROCESSING",
            meta={"progress": 90, "message": "正在生成向量索引...", "paper_id": paper_id},
        )

        try:
            # 分块 → 生成 embedding → 入库 pgvector
            chunk_count = index_paper(paper_id, paper_structure, db)
            logger.success(f"[Task] 向量化入库完成: {chunk_count} 个文本块")

            # 清理旧向量数据（仅在入库成功后清理，保证原子性）
            from app.schemas.models import DocumentChunk
            old_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.paper_id == paper_id,
                DocumentChunk.chunk_index >= chunk_count
            ).delete()
            if old_chunks:
                logger.info(f"[Task] 清理旧向量数据: {old_chunks} 条")
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"[Task] 向量化入库失败（非致命）: {e}")
            # 向量化失败不阻塞分析报告生成

        # ── 阶段 5: 持久化分析结果 ──
        self.update_state(
            state="PROCESSING",
            meta={"progress": 95, "message": "正在保存分析结果...", "paper_id": paper_id},
        )

        report = db.query(AnalysisReport).filter(
            AnalysisReport.paper_id == paper_id
        ).first()

        if not report:
            report = AnalysisReport(paper_id=paper_id)
            db.add(report)

        report.reading_plan = result.get("reading_plan")
        report.background_summary = result.get("background_summary")
        report.method_summary = result.get("method_summary")
        report.experiment_summary = result.get("experiment_summary")
        report.critique = result.get("critique")
        report.final_report = result.get("final_report")

        paper.status = "completed"
        db.commit()

        logger.success(f"[Task] 论文 {paper_id} 分析完成，报告已保存")
        return {
            "paper_id": paper_id,
            "status": "completed",
            "message": "分析完成",
            "report_length": len(result.get("final_report", "")) if result.get("final_report") else 0,
        }

    except Exception as e:
        logger.exception(f"[Task] 论文 {paper_id} 分析失败: {e}")
        # 用新 session 更新失败状态，避免原连接已断导致二次异常被吞
        try:
            db.rollback()
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if paper:
                paper.status = "failed"
                db.commit()
        except Exception as db_err:
            logger.error(f"[Task] 无法更新失败状态: {db_err}")
        self.update_state(
            state="FAILURE",
            meta={"progress": 0, "message": str(e)[:500], "paper_id": paper_id},
        )
        raise

    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
