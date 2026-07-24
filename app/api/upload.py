"""
PDF 上传接口
- POST /api/upload：上传 PDF 文件，返回 paper_id
- GET /api/papers：获取论文列表
- DELETE /api/papers/{paper_id}：删除论文
"""
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.core.database import get_db
from app.schemas.models import Paper, UploadResponse, PaperListItem, PaperListResponse
from app.services.storage import save_pdf, delete_pdf, delete_parsed_cache

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, summary="上传PDF论文")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF 论文文件"),
    db: Session = Depends(get_db),
):
    """
    上传一篇 PDF 论文。

    - 文件保存到 data/pdfs/ 目录（UUID 重命名）
    - 在数据库中创建论文记录
    - 返回 paper_id 供后续操作使用
    """
    # 校验文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 读取文件内容
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="文件为空")
        if len(content) > 100 * 1024 * 1024:  # 限制 100MB
            raise HTTPException(status_code=400, detail="文件大小超过 100MB 限制")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件读取失败: {e}")
        raise HTTPException(status_code=500, detail="文件读取失败")

    # 保存文件
    stored_filename, pdf_path = save_pdf(content, file.filename)

    # 创建数据库记录
    paper_id = uuid.uuid4()
    paper = Paper(
        id=paper_id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        pdf_path=pdf_path,
        status="uploaded",
    )

    try:
        db.add(paper)
        db.commit()
        db.refresh(paper)
    except Exception as e:
        # 回滚：删除已保存的文件
        delete_pdf(stored_filename)
        logger.error(f"数据库写入失败: {e}")
        raise HTTPException(status_code=500, detail="数据库写入失败")

    logger.success(f"论文上传成功: {paper_id} - {file.filename}")
    return UploadResponse(
        paper_id=str(paper_id),
        filename=file.filename,
        status="uploaded",
        message="论文上传成功，可以使用 paper_id 触发分析",
    )


@router.get("/papers", response_model=PaperListResponse, summary="获取论文列表")
async def list_papers(db: Session = Depends(get_db)):
    """获取所有已上传的论文列表，按创建时间倒序"""
    papers = (
        db.query(Paper)
        .order_by(Paper.created_at.desc())
        .limit(50)
        .all()
    )

    return PaperListResponse(
        papers=[
            PaperListItem(
                paper_id=str(p.id),
                title=p.title or p.original_filename,
                status=p.status,
                created_at=p.created_at.isoformat() if p.created_at else "",
            )
            for p in papers
        ],
        total=len(papers),
    )


@router.delete("/papers/{paper_id}", summary="删除论文")
async def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """删除论文及关联的所有数据（文件、缓存、向量、报告）"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    # 删除文件
    delete_pdf(paper.stored_filename)
    delete_parsed_cache(paper_id)

    # 删除数据库记录（级联删除 chunks、report、chat_history）
    db.delete(paper)
    db.commit()

    logger.info(f"论文已删除: {paper_id}")
    return {"paper_id": paper_id, "message": "论文及相关数据已删除"}
