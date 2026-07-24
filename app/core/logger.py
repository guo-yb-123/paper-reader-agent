"""
日志配置模块
- 基于 loguru 的全局日志方案
- 同时输出到控制台和文件，按天切割
- 所有 Agent 执行、API 请求、任务流转统一打日志
"""
import sys
import os
from loguru import logger
from app.core.config import settings


def setup_logger() -> None:
    """初始化全局日志配置"""
    # 移除默认 handler
    logger.remove()

    # --- 控制台输出 ---
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )

    # --- 文件输出（按天切割） ---
    os.makedirs(settings.log_dir, exist_ok=True)
    log_file = os.path.join(settings.log_dir, "paper_reader_{time:YYYY-MM-DD}.log")

    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        level="DEBUG",
        rotation="00:00",  # 每天午夜切割
        retention="30 days",  # 保留 30 天
        compression="gz",  # 旧日志压缩
        encoding="utf-8",
    )

    # --- 错误日志单独记录 ---
    error_log_file = os.path.join(settings.log_dir, "error_{time:YYYY-MM-DD}.log")
    logger.add(
        error_log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}\n{exception}"
        ),
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info("日志系统初始化完成")
