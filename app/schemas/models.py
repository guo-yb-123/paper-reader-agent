"""
数据模型模块
- Pydantic 模型：API 请求/响应
- SQLAlchemy 模型：数据库表结构
"""
import uuid
from datetime import datetime, UTC
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.core.config import settings


# ============================================================
# SQLAlchemy 声明基类
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# 数据库模型
# ============================================================

class Paper(Base):
    """论文主表"""
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename = Column(String(512), nullable=False, comment="原始文件名")
    stored_filename = Column(String(512), nullable=False, comment="存储文件名(UUID)")
    title = Column(String(1024), nullable=True, comment="论文标题")
    authors = Column(String(1024), nullable=True, comment="作者")
    abstract = Column(Text, nullable=True, comment="摘要")
    pdf_path = Column(String(1024), nullable=False, comment="PDF存储路径")
    parsed_json_path = Column(String(1024), nullable=True, comment="解析结果JSON路径")
    page_count = Column(Integer, nullable=True, comment="页数")
    status = Column(
        String(32), default="uploaded",
        comment="状态: uploaded/parsing/parsed/analyzing/completed/failed"
    )
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=datetime.utcnow)

    # 关联
    chunks = relationship("DocumentChunk", back_populates="paper", cascade="all, delete-orphan")
    report = relationship("AnalysisReport", back_populates="paper", uselist=False, cascade="all, delete-orphan")


class DocumentChunk(Base):
    """文档分块表（pgvector）"""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_title = Column(String(1024), nullable=True, comment="所属章节标题")
    chunk_index = Column(Integer, nullable=False, comment="块序号")
    content = Column(Text, nullable=False, comment="文本内容")
    # pgvector 向量字段 — 维度由 VECTOR_DIMENSION 配置决定
    embedding = Column(
        Vector(settings.vector_dimension), nullable=True,
        comment=f"文本向量({settings.vector_dimension}维)",
    )
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    paper = relationship("Paper", back_populates="chunks")


class AnalysisReport(Base):
    """分析报告表"""
    __tablename__ = "analysis_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), unique=True, nullable=False)
    reading_plan = Column(JSON, nullable=True, comment="Planner阅读计划")
    method_summary = Column(Text, nullable=True, comment="方法部分总结")
    experiment_summary = Column(Text, nullable=True, comment="实验部分总结")
    background_summary = Column(Text, nullable=True, comment="背景部分总结")
    critique = Column(Text, nullable=True, comment="批判性分析")
    final_report = Column(Text, nullable=True, comment="最终Markdown报告")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=datetime.utcnow)

    paper = relationship("Paper", back_populates="report")


class ChatHistory(Base):
    """对话历史表"""
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(32), nullable=False, comment="user/assistant")
    content = Column(Text, nullable=False, comment="对话内容")
    router_decision = Column(String(64), nullable=True, comment="Router路由决策")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


# ============================================================
# Pydantic 请求/响应模型
# ============================================================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


# --- 上传相关 ---

class UploadResponse(BaseModel):
    """PDF上传响应"""
    paper_id: str
    filename: str
    status: str
    message: str


# --- 分析相关 ---

class AnalyzeRequest(BaseModel):
    """触发分析请求"""
    paper_id: str


class AnalyzeResponse(BaseModel):
    """触发分析响应"""
    paper_id: str
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""
    task_id: str
    paper_id: str
    status: TaskStatus
    progress: Optional[int] = Field(default=0, ge=0, le=100, description="进度百分比")
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


# --- 报告相关 ---

class ReportResponse(BaseModel):
    """报告查询响应"""
    paper_id: str
    title: Optional[str] = None
    status: str
    final_report: Optional[str] = None
    reading_plan: Optional[dict] = None
    method_summary: Optional[str] = None
    experiment_summary: Optional[str] = None
    background_summary: Optional[str] = None
    critique: Optional[str] = None


# --- 对话相关 ---

class ChatRequest(BaseModel):
    """对话请求"""
    paper_id: str
    question: str = Field(..., min_length=1, max_length=5000)
    history: Optional[list[dict]] = Field(default=[], description="对话历史")


class ChatResponse(BaseModel):
    """对话响应"""
    paper_id: str
    question: str
    answer: str
    router_decision: Optional[str] = None


# --- 论文列表 ---

class PaperListItem(BaseModel):
    """论文列表项"""
    paper_id: str
    title: Optional[str] = None
    status: str
    created_at: str


class PaperListResponse(BaseModel):
    """论文列表响应"""
    papers: list[PaperListItem]
    total: int
