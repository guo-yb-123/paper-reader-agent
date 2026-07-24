# ============================================================
# 多Agent科研论文阅读系统 - Docker镜像
# ============================================================
FROM python:3.11-slim

LABEL project="paper-reader-agent"
LABEL description="Multi-Agent Research Paper Reading System"

# 注意：psycopg2-binary 自带 libpq，无需安装系统级数据库驱动
# python:3.11-slim 已包含 Python 运行时所有必要组件

# 工作目录
WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .

# 升级 pip + 使用国内镜像加速（如果在国内）
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
        --retries 5 \
        --timeout 120

# 复制项目代码
COPY . .

# 创建数据和日志目录
RUN mkdir -p data/pdfs data/parsed logs

# 创建非 root 用户
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000 8501

# 默认启动 FastAPI（可通过 docker-compose command 覆盖）
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]
