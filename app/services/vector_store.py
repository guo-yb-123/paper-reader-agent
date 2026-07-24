"""
向量存储与检索服务
- 将论文章节按段落分块（每块约500字）
- 调用 Embedding 模型生成向量
- 存入 pgvector，支持语义相似度检索
"""
from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings, get_embeddings
from app.schemas.models import DocumentChunk
from app.services.converters import parsed_paper_to_dict


# ============================================================
# 文本分块
# ============================================================

def split_text_into_chunks(
    text: str,
    section_title: str,
    chunk_size: int = None,
    overlap: int = 100,
    start_index: int = 0,
) -> list[dict]:
    """
    将文本按段落分块，每块约 chunk_size 字。
    尽量保留段落边界，避免在句子中间切断。

    Args:
        text: 要分块的文本
        section_title: 所属章节标题
        chunk_size: 每块最大字符数（默认从配置读取）
        overlap: 块之间的重叠字符数
        start_index: 起始块序号（用于跨章节统一编号）

    Returns:
        [{"content": "...", "section_title": "...", "chunk_index": 0}, ...]
    """
    if chunk_size is None:
        chunk_size = settings.vector_chunk_size  # 默认 500

    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    chunks = []
    current_chunk = ""
    chunk_index = start_index

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 情况1：当前块 + 新段落 <= chunk_size，直接追加
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = (current_chunk + "\n" + para).strip() if current_chunk else para
            continue

        # 情况2：当前块已有内容，先保存
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "section_title": section_title,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            # 用 overlap 衔接
            overlap_text = current_chunk[-overlap:] if overlap > 0 and len(current_chunk) > overlap else ""
            current_chunk = (overlap_text + "\n" + para).strip() if overlap_text else para
        else:
            # 情况3：current_chunk 为空，但单个段落超过 chunk_size → 强制截断
            current_chunk = para

        # 情况4：当前块仍超过 chunk_size，循环截断
        while len(current_chunk) > chunk_size:
            cut_pos = current_chunk[:chunk_size].rfind("\n")
            if cut_pos < chunk_size // 2:
                cut_pos = chunk_size

            chunks.append({
                "content": current_chunk[:cut_pos].strip(),
                "section_title": section_title,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            # 下一块带 overlap
            start = max(0, cut_pos - overlap)
            current_chunk = current_chunk[start:].strip()

    # 保存最后一块
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "section_title": section_title,
            "chunk_index": chunk_index,
        })

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """将文本分割为段落列表"""
    # 先按双换行分
    if "\n\n" in text:
        paragraphs = text.split("\n\n")
    else:
        paragraphs = text.split("\n")

    # 合并过短的段落
    merged = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        if len(para) < 50 and buffer:
            buffer += " " + para
        elif len(para) < 50:
            buffer = para
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(para)
    if buffer:
        merged.append(buffer)

    return merged if merged else [text]


# ============================================================
# 论文全文分块
# ============================================================

def chunk_paper(paper_structure: dict) -> list[dict]:
    """
    对解析后的论文结构进行全文分块。
    遍历所有章节，对每个章节的正文进行分块。

    Args:
        paper_structure: parse_pdf 返回的 ParsedPaper 或缓存中的 dict

    Returns:
        所有文本块的列表（chunk_index 全局统一编号）
    """
    all_chunks = []
    global_index = 0

    # 摘要单独分块
    abstract = paper_structure.get("abstract", "")
    if abstract:
        abstract_chunks = split_text_into_chunks(
            abstract, "Abstract", start_index=global_index
        )
        all_chunks.extend(abstract_chunks)
        global_index += len(abstract_chunks)

    # 递归处理章节
    sections = paper_structure.get("sections", [])
    _chunk_sections(sections, all_chunks, global_index)

    logger.info(f"论文分块完成: 共 {len(all_chunks)} 个文本块")
    return all_chunks


def _chunk_sections(
    sections: list[dict],
    all_chunks: list[dict],
    global_index: int,
) -> int:
    """
    递归处理章节树，为每个章节的正文分块。
    返回更新后的全局索引。
    """
    for section in sections:
        title = section.get("title", "")
        content = section.get("content", "")

        if content.strip():
            chunks = split_text_into_chunks(
                content, title, start_index=global_index
            )
            all_chunks.extend(chunks)
            global_index += len(chunks)

        # 递归子章节
        children = section.get("children", [])
        if children:
            global_index = _chunk_sections(children, all_chunks, global_index)

    return global_index


# ============================================================
# 向量化与存储
# ============================================================

def embed_and_store(
    paper_id: str,
    chunks: list[dict],
    db: Session,
) -> int:
    """
    对文本块生成 Embedding 向量并存入数据库。

    Args:
        paper_id: 论文ID
        chunks: 文本块列表
        db: 数据库会话

    Returns:
        存储的块数量
    """
    if not chunks:
        logger.warning(f"论文 {paper_id} 没有可存储的文本块")
        return 0

    # 获取 embedding 模型
    embeddings_client = get_embeddings()
    logger.info(f"使用 Embedding 模型: {settings.embedding_model}")

    # 批量生成向量（减少 API 调用次数）
    texts = [c["content"] for c in chunks]
    try:
        logger.info(f"正在为 {len(texts)} 个文本块生成向量...")
        vectors = embeddings_client.embed_documents(texts)
        logger.success(f"向量生成完成: {len(vectors)} 个")
    except Exception as e:
        logger.error(f"Embedding 生成失败: {e}")
        raise

    # 批量插入数据库
    chunk_objects = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_obj = DocumentChunk(
            paper_id=paper_id,
            section_title=chunk.get("section_title", ""),
            chunk_index=chunk["chunk_index"],
            content=chunk["content"],
            embedding=vector,
        )
        chunk_objects.append(chunk_obj)

    try:
        db.add_all(chunk_objects)
        db.commit()
        logger.success(f"成功存储 {len(chunk_objects)} 个文本块到向量数据库")
    except Exception as e:
        db.rollback()
        logger.error(f"向量存储失败: {e}")
        raise

    return len(chunk_objects)


# ============================================================
# 相似度检索
# ============================================================

def search_similar_chunks(
    query: str,
    paper_id: str,
    db: Session,
    top_k: int = None,
) -> list[dict]:
    """
    根据查询文本进行语义检索，返回最相关的文本块。

    Args:
        query: 查询文本
        paper_id: 限定在指定论文内检索
        db: 数据库会话
        top_k: 返回结果数（默认从配置读取）

    Returns:
        [{"content": "...", "section_title": "...", "similarity": 0.95}, ...]
    """
    if top_k is None:
        top_k = settings.vector_top_k  # 默认 5

    # 生成查询向量
    embeddings_client = get_embeddings()
    try:
        query_vector = embeddings_client.embed_query(query)
    except Exception as e:
        logger.error(f"查询向量生成失败: {e}")
        raise

    # 使用 pgvector 的余弦相似度运算符 <=> 检索
    # 注意：pgvector 的 <=> 返回余弦距离，(1 - 距离) = 相似度
    try:
        result = db.execute(
            text("""
                SELECT
                    id,
                    section_title,
                    content,
                    chunk_index,
                    1 - (embedding <=> :query_vector) AS similarity
                FROM document_chunks
                WHERE paper_id = :paper_id
                    AND embedding IS NOT NULL
                ORDER BY embedding <=> :query_vector
                LIMIT :top_k
            """),
            {
                "query_vector": query_vector,
                "paper_id": paper_id,
                "top_k": top_k,
            },
        )

        rows = result.fetchall()
        results = [
            {
                "content": row.content,
                "section_title": row.section_title,
                "similarity": round(row.similarity, 4),
            }
            for row in rows
        ]

        logger.info(f"检索完成: 查询='{query[:50]}...', 返回 {len(results)} 条结果")
        return results

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        raise


def search_all_papers(
    query: str,
    db: Session,
    top_k: int = 5,
) -> list[dict]:
    """
    跨论文检索（不限 paper_id），用于全局搜索。

    Args:
        query: 查询文本
        db: 数据库会话
        top_k: 返回结果数

    Returns:
        包含 paper_id 的检索结果列表
    """
    embeddings_client = get_embeddings()
    query_vector = embeddings_client.embed_query(query)

    result = db.execute(
        text("""
            SELECT
                dc.id,
                dc.paper_id,
                p.title AS paper_title,
                dc.section_title,
                dc.content,
                1 - (dc.embedding <=> :query_vector) AS similarity
            FROM document_chunks dc
            JOIN papers p ON dc.paper_id = p.id
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> :query_vector
            LIMIT :top_k
        """),
        {"query_vector": query_vector, "top_k": top_k},
    )

    rows = result.fetchall()
    return [
        {
            "paper_id": str(row.paper_id),
            "paper_title": row.paper_title,
            "section_title": row.section_title,
            "content": row.content,
            "similarity": round(row.similarity, 4),
        }
        for row in rows
    ]


# ============================================================
# 向量化完整流程
# ============================================================

def index_paper(paper_id: str, paper_structure: dict, db: Session) -> int:
    """
    对论文执行完整的向量化入库流程：
    1. 全文分块
    2. 生成 Embedding
    3. 存入 pgvector

    Args:
        paper_id: 论文ID
        paper_structure: 解析后的论文结构（dict 或 ParsedPaper）
        db: 数据库会话

    Returns:
        入库的文本块数量
    """
    logger.info(f"开始向量化论文 {paper_id}")

    # 如果是 ParsedPaper 对象，转为 dict
    if hasattr(paper_structure, '__dataclass_fields__'):
        paper_structure = parsed_paper_to_dict(paper_structure)

    # 1. 分块
    chunks = chunk_paper(paper_structure)
    if not chunks:
        logger.warning(f"论文 {paper_id} 分块结果为空")
        return 0

    # 2 + 3. 向量化 + 存储
    count = embed_and_store(paper_id, chunks, db)

    logger.success(f"论文 {paper_id} 向量化完成，共 {count} 个文本块")
    return count


