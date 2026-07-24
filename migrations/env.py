"""
Alembic 迁移环境配置
- 从项目 Settings 读取数据库 URL
- 关联 SQLAlchemy Base metadata 以支持 autogenerate
"""
from logging.config import fileConfig
from alembic import context

# 加载项目配置（通过环境变量/.env）
from app.core.config import settings

# Alembic Config 对象
config = context.config

# 用项目配置覆盖 alembic.ini 中的 sqlalchemy.url
config.set_main_option("sqlalchemy.url", settings.database_url)

# 设置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 导入所有模型，确保 Base.metadata 包含全部表
from app.schemas.models import Base  # noqa: E402, F401

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    离线模式：生成 SQL 脚本而非直接连接数据库。
    用法: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    在线模式：直接连接数据库执行迁移。
    用法: alembic upgrade head
    """
    from sqlalchemy import create_engine

    connectable = create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
