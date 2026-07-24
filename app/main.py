"""
FastAPI 应用入口
- 创建 FastAPI 实例
- 注册路由
- 配置启动/关闭事件
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, detect_available_model
from app.core.logger import setup_logger, logger
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # --- 启动时 ---
    setup_logger()
    logger.info("=" * 60)
    logger.info("多Agent科研论文阅读系统 启动中...")
    logger.info("=" * 60)

    # 初始化数据库
    init_db()

    # 自动检测可用模型
    try:
        available_model = await detect_available_model()
        logger.success(f"LLM 模型已就绪: {available_model}")
    except RuntimeError as e:
        logger.error(f"模型检测失败，应用将以降级模式运行: {e}")
        # 不阻止应用启动，允许后续手动配置

    logger.info("应用启动完成，等待请求...")
    yield

    # --- 关闭时 ---
    logger.info("应用正在关闭...")
    # 清理资源（如果有需要）


# 创建 FastAPI 应用
app = FastAPI(
    title="多Agent科研论文阅读系统",
    description="基于 LangGraph + 通义千问 的多Agent协作论文精读系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 根路由 ---
@app.get("/")
async def root():
    """健康检查 & 基本信息"""
    return {
        "service": "Paper Reader Agent",
        "version": "0.1.0",
        "status": "running",
        "llm_model": settings.available_llm_model or "未检测",
    }


@app.get("/health")
async def health_check():
    """健康检查端点（供 Docker 健康检查使用）"""
    return {"status": "healthy"}


# --- 注册各模块路由 ---
from app.api import upload, analyze, chat
app.include_router(upload.router, prefix="/api", tags=["上传与管理"])
app.include_router(analyze.router, prefix="/api", tags=["分析与报告"])
app.include_router(chat.router, prefix="/api", tags=["对话问答"])
