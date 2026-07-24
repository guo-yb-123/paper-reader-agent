"""
数据库初始化模块
- 创建 SQLAlchemy 引擎和会话工厂（延迟初始化）
- 提供 get_db 依赖注入
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logger import logger

# 延迟初始化（避免导入时就连接数据库）
_engine = None
_SessionLocal = None


def _get_engine():
    """获取或创建数据库引擎（懒加载）"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def _get_session_local():
    """获取或创建会话工厂"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
        )
    return _SessionLocal


def init_db() -> None:
    """
    初始化数据库：创建所有表。

    开发环境：使用 SQLAlchemy create_all 自动建表。
    生产环境：推荐使用 Alembic 迁移。
        cd paper-reader-agent
        alembic upgrade head
    """
    from app.schemas.models import Base

    try:
        engine = _get_engine()
        # 确保 pgvector 扩展已启用
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
        logger.info("数据库扩展已就绪 (vector, uuid-ossp)")

        # 创建所有表（开发用；生产请用 alembic upgrade head）
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建完成（开发模式 create_all）")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


def get_db() -> Session:
    """FastAPI 依赖注入：获取数据库会话"""
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()
