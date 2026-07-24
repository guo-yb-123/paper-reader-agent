"""Initial migration: create all core tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-23
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建所有核心表"""

    # ── pgvector 扩展 ──
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── papers 论文主表 ──
    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("original_filename", sa.String(512), nullable=False,
                  comment="原始文件名"),
        sa.Column("stored_filename", sa.String(512), nullable=False,
                  comment="存储文件名(UUID)"),
        sa.Column("title", sa.String(1024), nullable=True, comment="论文标题"),
        sa.Column("authors", sa.String(1024), nullable=True, comment="作者"),
        sa.Column("abstract", sa.Text, nullable=True, comment="摘要"),
        sa.Column("pdf_path", sa.String(1024), nullable=False, comment="PDF存储路径"),
        sa.Column("parsed_json_path", sa.String(1024), nullable=True,
                  comment="解析结果JSON路径"),
        sa.Column("page_count", sa.Integer, nullable=True, comment="页数"),
        sa.Column("status", sa.String(32), server_default="uploaded",
                  comment="状态: uploaded/parsing/parsed/analyzing/completed/failed"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
    )

    # ── document_chunks 文档分块表（pgvector）──
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_title", sa.String(1024), nullable=True,
                  comment="所属章节标题"),
        sa.Column("chunk_index", sa.Integer, nullable=False, comment="块序号"),
        sa.Column("content", sa.Text, nullable=False, comment="文本内容"),
        sa.Column("embedding", postgresql.ARRAY(sa.Float), nullable=True,
                  comment="文本向量(1536维)"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
    )
    # 索引：按论文检索
    op.create_index("ix_document_chunks_paper_id", "document_chunks", ["paper_id"])

    # ── analysis_reports 分析报告表 ──
    op.create_table(
        "analysis_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), unique=True,
                  nullable=False),
        sa.Column("reading_plan", postgresql.JSON, nullable=True,
                  comment="Planner阅读计划"),
        sa.Column("method_summary", sa.Text, nullable=True,
                  comment="方法部分总结"),
        sa.Column("experiment_summary", sa.Text, nullable=True,
                  comment="实验部分总结"),
        sa.Column("background_summary", sa.Text, nullable=True,
                  comment="背景部分总结"),
        sa.Column("critique", sa.Text, nullable=True, comment="批判性分析"),
        sa.Column("final_report", sa.Text, nullable=True,
                  comment="最终Markdown报告"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
    )

    # ── chat_history 对话历史表 ──
    op.create_table(
        "chat_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, comment="user/assistant"),
        sa.Column("content", sa.Text, nullable=False, comment="对话内容"),
        sa.Column("router_decision", sa.String(64), nullable=True,
                  comment="Router路由决策"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
    )
    # 索引：按论文和时间查询对话历史
    op.create_index("ix_chat_history_paper_id", "chat_history",
                    ["paper_id", "created_at"])


def downgrade() -> None:
    """删除所有核心表"""
    op.drop_table("chat_history")
    op.drop_table("analysis_reports")
    op.drop_index("ix_document_chunks_paper_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("papers")
