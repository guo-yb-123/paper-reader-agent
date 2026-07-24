"""
文件存储服务
- PDF 文件保存（UUID 重命名防冲突）
- 文件读取与清理
"""
import os
import uuid
from loguru import logger
from app.core.config import settings


def save_pdf(file_content: bytes, original_filename: str) -> tuple[str, str]:
    """
    保存上传的 PDF 文件。

    Args:
        file_content: 文件字节内容
        original_filename: 原始文件名

    Returns:
        (stored_filename, pdf_path): 存储文件名和完整路径
    """
    # 生成唯一文件名
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "pdf"
    stored_filename = f"{uuid.uuid4().hex}.{ext}"

    # 确保存储目录存在
    os.makedirs(settings.pdf_storage_dir, exist_ok=True)

    # 写入文件
    pdf_path = os.path.join(settings.pdf_storage_dir, stored_filename)
    with open(pdf_path, "wb") as f:
        f.write(file_content)

    logger.info(f"PDF 已保存: {original_filename} → {stored_filename}")
    return stored_filename, pdf_path


def get_pdf_path(stored_filename: str) -> str:
    """获取 PDF 文件的完整路径"""
    return os.path.join(settings.pdf_storage_dir, stored_filename)


def delete_pdf(stored_filename: str) -> bool:
    """删除 PDF 文件"""
    pdf_path = get_pdf_path(stored_filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        logger.info(f"PDF 已删除: {stored_filename}")
        return True
    return False


def delete_parsed_cache(paper_id: str) -> bool:
    """删除解析缓存"""
    cache_path = os.path.join(settings.parsed_storage_dir, f"{paper_id}.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        return True
    return False
