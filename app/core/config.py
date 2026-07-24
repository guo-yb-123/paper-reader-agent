"""
核心配置模块
- 使用 pydantic-settings 从环境变量/.env 加载配置
- 启动时自动检测可用的 LLM 模型
"""
from typing import Optional
from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用全局配置"""

    # --- LLM API ---
    llm_api_key: str = Field(
        default="",
        alias="LLM_API_KEY",
        validation_alias="LLM_API_KEY",
    )
    # 向后兼容：如果设置了 DASHSCOPE_API_KEY 但没有 LLM_API_KEY，使用前者
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")

    # --- 模型配置 ---
    llm_model_candidates: str = Field(
        default="qwen-turbo,qwen-plus",
        alias="LLM_MODEL_CANDIDATES",
    )
    embedding_provider: str = Field(
        default="dashscope",
        alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="text-embedding-v2",
        alias="EMBEDDING_MODEL",
    )
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="LLM_BASE_URL",
    )

    # --- PostgreSQL ---
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="paper_reader", alias="POSTGRES_USER")
    postgres_password: str = Field(default="paper_reader_pass", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="paper_reader_db", alias="POSTGRES_DB")

    # --- Redis ---
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # --- Celery ---
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND"
    )

    # --- 应用配置 ---
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")

    # --- 文件存储 ---
    pdf_storage_dir: str = Field(default="./data/pdfs", alias="PDF_STORAGE_DIR")
    parsed_storage_dir: str = Field(default="./data/parsed", alias="PARSED_STORAGE_DIR")

    # --- 向量配置 ---
    vector_chunk_size: int = Field(default=500, alias="VECTOR_CHUNK_SIZE")
    vector_top_k: int = Field(default=5, alias="VECTOR_TOP_K")
    vector_dimension: int = Field(default=1536, alias="VECTOR_DIMENSION")

    # --- 运行时检测出的可用模型（非环境变量） ---
    available_llm_model: Optional[str] = None

    @property
    def effective_api_key(self) -> str:
        """优先用 LLM_API_KEY，回退到 DASHSCOPE_API_KEY（向后兼容）"""
        return self.llm_api_key or self.dashscope_api_key

    @property
    def database_url(self) -> str:
        """拼接 PostgreSQL 连接字符串"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def candidate_models(self) -> list[str]:
        """解析候选模型列表"""
        return [m.strip() for m in self.llm_model_candidates.split(",") if m.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# 全局单例
settings = Settings()


async def detect_available_model() -> str:
    """
    自动检测可用的 LLM 模型。
    按候选列表依次发送测试请求，返回第一个可用的模型名。
    如果全部不可用，抛出 RuntimeError。

    检测结果会缓存到 settings.available_llm_model。
    """
    if settings.available_llm_model is not None:
        return settings.available_llm_model

    # 延迟导入，避免循环依赖
    from langchain_openai import ChatOpenAI

    candidates = settings.candidate_models
    logger.info(f"开始检测可用 LLM 模型，候选列表: {candidates}")

    for model_name in candidates:
        try:
            logger.info(f"正在测试模型: {model_name} ...")
            llm = ChatOpenAI(
                model=model_name,
                base_url=settings.llm_base_url,
                api_key=settings.effective_api_key,
                max_tokens=5,  # 只发极少 token 节省额度
                temperature=0,
            )
            # 最简单的测试请求
            response = await llm.ainvoke("hi")
            if response and response.content:
                settings.available_llm_model = model_name
                logger.success(f"模型 {model_name} 可用，已选为主模型")
                return model_name
        except Exception as e:
            logger.warning(f"模型 {model_name} 不可用: {str(e)[:200]}")

    # 全部失败
    error_msg = (
        f"所有候选模型均不可用，已测试: {candidates}。"
        f"请检查 API Key 是否正确，或模型额度是否耗尽。"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def _detect_model_sync() -> None:
    """
    同步版本：依次测试候选模型，找到第一个可用的。
    用于 Celery Worker 等无法运行 async 的上下文。
    """
    from langchain_openai import ChatOpenAI

    candidates = settings.candidate_models
    logger.info(f"[Sync Detect] 开始检测可用 LLM 模型，候选列表: {candidates}")

    for model_name in candidates:
        try:
            llm = ChatOpenAI(
                model=model_name,
                base_url=settings.llm_base_url,
                api_key=settings.effective_api_key,
                max_tokens=5,
                temperature=0,
            )
            response = llm.invoke("hi")
            if response and response.content:
                settings.available_llm_model = model_name
                logger.success(f"[Sync Detect] 模型 {model_name} 可用")
                return
        except Exception as e:
            logger.warning(f"[Sync Detect] 模型 {model_name} 不可用: {str(e)[:200]}")

    raise RuntimeError(
        f"所有候选模型均不可用，已测试: {candidates}。"
        f"请检查 DASHSCOPE_API_KEY 或网络连接。"
    )


def get_llm(**kwargs):
    """
    获取配置好的 ChatOpenAI 实例。

    首次调用时自动检测可用模型（同步版本，兼容 Celery Worker）。
    如果没有任何可用模型，抛出 RuntimeError。
    """
    from langchain_openai import ChatOpenAI

    if settings.available_llm_model is None:
        _detect_model_sync()

    return ChatOpenAI(
        model=settings.available_llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.effective_api_key,
        temperature=kwargs.pop("temperature", 0.1),
        **kwargs,
    )


def get_embeddings():
    """
    获取 Embeddings 实例。

    - local: 使用本地 HuggingFace 模型（免费，需 sentence-transformers）
    - dashscope: 使用通义千问 Embedding API
    """
    if settings.embedding_provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "本地 Embedding 需要 sentence-transformers。请安装:\n"
                "  pip install langchain-huggingface sentence-transformers"
            )
        logger.info("使用本地 Embedding 模型: BAAI/bge-small-zh-v1.5")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # dashscope / 默认：DashScope 原生 Embedding API
    # OpenAI 兼容模式的 /embeddings 端点与 DashScope 格式不兼容，
    # 因此使用 DashScope 原生 text-embedding API
    from langchain_core.embeddings import Embeddings
    import httpx
    from typing import List

    class DashScopeEmbeddings(Embeddings):
        """DashScope 原生 Embedding 封装，实现 LangChain Embeddings 接口"""

        def __init__(self, model: str, api_key: str):
            self.model = model
            self._api_key = api_key
            self._client = httpx.Client(
                base_url="https://dashscope.aliyuncs.com",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """批量生成文本向量"""
            if not texts:
                return []
            # DashScope 单次最多 25 条，分批处理
            all_vectors = []
            batch_size = 25
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self._client.post(
                    "/api/v1/services/embeddings/text-embedding/text-embedding",
                    json={
                        "model": self.model,
                        "input": {"texts": batch},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("code"):
                    raise RuntimeError(
                        f"Embedding API error: {data.get('code')} - {data.get('message')}"
                    )
                embeddings = data["output"]["embeddings"]
                all_vectors.extend([e["embedding"] for e in embeddings])
            return all_vectors

        def embed_query(self, text: str) -> List[float]:
            """生成单条查询向量"""
            return self.embed_documents([text])[0]

    logger.info(f"使用 DashScope 原生 Embedding API，模型: {settings.embedding_model}")
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_api_key,
    )


def invoke_llm(llm, messages):
    """
    带指数退避重试的 LLM 同步调用。

    用法：代替 llm.invoke(messages)，自动重试瞬时网络错误。
    对额度耗尽、认证失败等硬错误不重试，直接抛出。
    """
    from app.core.retry_utils import invoke_llm_with_retry

    return invoke_llm_with_retry(llm, messages)
