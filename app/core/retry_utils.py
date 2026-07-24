"""
LLM 调用重试工具模块
- 利用 tenacity 为 LLM 调用添加指数退避重试
- 处理常见瞬时错误：rate limit、超时、连接中断
"""
from loguru import logger

# 重试配置
MAX_RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 2       # 最小等待 2 秒
RETRY_MAX_WAIT = 30      # 最大等待 30 秒

# 可重试的异常类型（瞬时错误，重试有望恢复）
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,              # 网络层面的连接重置等
)


def _is_retryable(exception: Exception) -> bool:
    """
    判断异常是否值得重试。
    对已知的不可恢复错误（额度耗尽、认证失败等）不重试。
    """
    error_msg = str(exception).lower()

    # 不可重试的错误特征
    non_retryable_patterns = [
        "insufficient_quota",
        "quota exceeded",
        "invalid api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "not found",           # 模型不存在
        "invalid_request_error",  # 参数错误，重试徒劳
        "context length",
        "maximum context",
        "token limit",         # prompt 超长，重试不会变短
    ]

    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            logger.warning(f"[Retry] 检测到不可重试错误: {pattern}，跳过重试")
            return False

    return True


def invoke_llm_with_retry(llm, messages):
    """
    带指数退避重试的 LLM 调用。

    用法（替代 llm.invoke(messages)）:
        response = invoke_llm_with_retry(llm, messages)

    对 ConnectionError / TimeoutError 等瞬时错误自动重试最多 3 次。
    对额度耗尽、认证失败等硬错误不重试，直接抛出。
    """
    last_exception = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return llm.invoke(messages)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < MAX_RETRY_ATTEMPTS:
                wait_time = min(RETRY_MIN_WAIT * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
                logger.warning(
                    f"[Retry] LLM 调用失败 (第 {attempt}/{MAX_RETRY_ATTEMPTS} 次): "
                    f"{str(e)[:200]}。{wait_time}s 后重试..."
                )
                import time
                time.sleep(wait_time)
            else:
                logger.error(f"[Retry] LLM 调用重试耗尽 ({MAX_RETRY_ATTEMPTS} 次全部失败)")
                raise
        except Exception as e:
            # 硬错误：检查是否可重试
            if _is_retryable(e) and attempt < MAX_RETRY_ATTEMPTS:
                wait_time = min(RETRY_MIN_WAIT * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
                logger.warning(
                    f"[Retry] LLM 调用异常 (第 {attempt}/{MAX_RETRY_ATTEMPTS} 次): "
                    f"{str(e)[:200]}。{wait_time}s 后重试..."
                )
                import time
                time.sleep(wait_time)
                last_exception = e
            else:
                raise

    # 理论上不会到达这里
    raise last_exception  # type: ignore[misc]
