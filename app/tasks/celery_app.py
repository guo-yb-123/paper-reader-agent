"""
Celery 应用配置模块
- Broker: Redis
- Result Backend: Redis
- 任务序列化: JSON
"""
from celery import Celery
from app.core.config import settings
from app.core.logger import logger

# 创建 Celery 应用实例
celery_app = Celery(
    "paper_reader",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.analysis_tasks"],  # 自动发现任务模块
)

# Celery 配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # 任务完成后才确认，防止 worker 崩溃丢失任务
    worker_prefetch_multiplier=1,  # 每次只预取一个任务
    task_soft_time_limit=600,  # 软超时 10 分钟
    task_time_limit=900,  # 硬超时 15 分钟
    result_expires=86400,  # 结果保留 24 小时
)

logger.info(f"Celery 应用初始化完成，broker: {settings.celery_broker_url}")
