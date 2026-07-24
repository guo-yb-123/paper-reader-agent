"""
PDF 结构化解析服务
- 使用 PyMuPDF (fitz) 提取文本、字体大小、加粗、页码等格式信息
- 根据字体大小分布自动判断标题层级
- 正则辅助识别章节标题（编号标题 + 命名标题）
- 自动提取论文标题和摘要
- 构建树形章节结构，支持 JSON 缓存
"""
import os
import json
import re
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter
import fitz  # PyMuPDF
from loguru import logger

from app.core.config import settings


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TextBlock:
    """文本块（一个段落或标题）"""
    text: str
    font_size: float
    is_bold: bool
    page_num: int
    bbox: tuple  # 边界框 (x0, y0, x1, y1)


@dataclass
class Section:
    """章节节点（树形结构）"""
    title: str
    level: int                # 1/2/3 级标题
    page_num: int
    content: str = ""         # 该章节下的正文（不含子章节）
    children: list["Section"] = field(default_factory=list)
    font_size: float = 0.0


@dataclass
class ParsedPaper:
    """解析完成的论文结构"""
    paper_id: str
    title: str
    authors: str
    abstract: str
    sections: list[Section]   # 一级章节列表
    page_count: int
    raw_text: str             # 全文纯文本（备用）


# ============================================================
# 标题检测正则
# ============================================================

# 编号标题：1. / 1.1 / 1.1.1 / 2. / I. / A. 等
NUMBERED_HEADING_PATTERNS = [
    re.compile(r'^\s*(\d+(?:\.\d+)*)\.?\s+(.+)'),       # 1. Introduction / 1.1 Method
    re.compile(r'^\s*([IVX]+)\.?\s+(.+)'),                # I. Introduction
    re.compile(r'^\s*([A-Z])\.?\s+(.+)'),                  # A. Method
]

# 命名标题关键词（按层级）
NAMED_HEADINGS_L1 = [
    'abstract', 'introduction', 'related work', 'background',
    'method', 'methodology', 'approach', 'experiment', 'evaluation',
    'results', 'discussion', 'conclusion', 'references', 'acknowledgment',
    'appendix', 'supplementary',
]

NAMED_HEADINGS_L2 = [
    'dataset', 'implementation', 'training', 'inference',
    'baseline', 'ablation', 'hyperparameter', 'evaluation metric',
    'model architecture', 'loss function', 'optimization',
    'data preprocessing', 'feature extraction',
]


def _is_heading_text(text: str) -> bool:
    """判断文本是否为章节标题"""
    text_clean = text.strip().lower().rstrip('.')

    # 编号标题
    for pat in NUMBERED_HEADING_PATTERNS:
        if pat.match(text.strip()):
            return True

    # 命名标题
    for heading in NAMED_HEADINGS_L1 + NAMED_HEADINGS_L2:
        if text_clean == heading or text_clean.startswith(heading):
            return True

    return False


def _guess_level(text: str, font_size: float, font_thresholds: dict) -> int:
    """
    综合判断标题层级。
    结合字体大小阈值和文本特征。
    """
    text_clean = text.strip().lower().rstrip('.')

    # 按文本内容判断
    if text_clean in NAMED_HEADINGS_L1:
        return 1
    if text_clean in NAMED_HEADINGS_L2:
        return 2

    # 按编号深度判断
    for pat in NUMBERED_HEADING_PATTERNS:
        m = pat.match(text.strip())
        if m:
            num_part = m.group(1)
            depth = num_part.count('.') + 1
            return min(depth, 3)

    # 按字体大小判断
    if font_size >= font_thresholds.get('h1', 16):
        return 1
    elif font_size >= font_thresholds.get('h2', 13):
        return 2
    elif font_size >= font_thresholds.get('h3', 11):
        return 3

    return 2


# ============================================================
# PDF 解析核心
# ============================================================

class PDFParser:
    """PDF 结构化解析器"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc: Optional[fitz.Document] = None
        self.blocks: list[TextBlock] = []
        self.font_sizes: list[float] = []
        self.font_thresholds: dict = {}

    # -------- 步骤1：提取文本块 --------

    def _extract_blocks(self) -> None:
        """
        从 PDF 中提取所有文本块，记录字体大小、是否加粗、页码等信息。
        使用 page.get_text("dict") 获取结构化文本。
        """
        logger.info(f"开始提取文本块: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(self.doc, start=1):
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # 非文本块（图片等），跳过
                    continue
                for line in block.get("lines", []):
                    line_text_parts = []
                    max_font_size = 0.0
                    is_bold = False

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        line_text_parts.append(text)

                        font_size = span.get("size", 10.0)
                        max_font_size = max(max_font_size, font_size)
                        self.font_sizes.append(font_size)

                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or "heavy" in font_name or "black" in font_name:
                            is_bold = True

                    full_text = " ".join(line_text_parts).strip()
                    if not full_text:
                        continue

                    bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
                    self.blocks.append(TextBlock(
                        text=full_text,
                        font_size=round(max_font_size, 1),
                        is_bold=is_bold,
                        page_num=page_num,
                        bbox=bbox,
                    ))

        logger.info(f"提取完成: {len(self.blocks)} 个文本块, {len(self.doc)} 页")

    # -------- 步骤2：分析字体大小分布 --------

    def _analyze_font_sizes(self) -> None:
        """
        分析字体大小分布，自动确定各级标题的字体阈值。
        策略：统计所有字体大小，找出分布规律。
        - 最大字号 → 论文标题
        - 较大的字号（前10%）→ 一级标题
        - 中等字号 → 二级标题
        - 正文 → 最常见的字号
        """
        if not self.font_sizes:
            return

        counter = Counter(round(s, 1) for s in self.font_sizes)
        most_common = counter.most_common()
        logger.info(f"字体大小分布 (top 10): {most_common[:10]}")

        # 正文字体：出现频率最高的字号
        body_size = most_common[0][0] if most_common else 10.0

        # 收集所有大于正文字体的字号
        larger_sizes = sorted(
            [s for s in set(self.font_sizes) if s > body_size + 0.5],
            reverse=True,
        )

        # 一级标题阈值：比正文大 4pt 以上
        h1_size = body_size + 4.0
        # 二级标题阈值：比正文大 2pt 以上
        h2_size = body_size + 2.0
        # 三级标题阈值：比正文大 0.5pt 以上
        h3_size = body_size + 0.5

        # 如果有分离的大字号数据，用实际数据调整
        if len(larger_sizes) >= 3:
            h1_size = larger_sizes[0]  # 最大字号 → 一级标题
            h2_size = larger_sizes[min(1, len(larger_sizes) - 1)]
            h3_size = larger_sizes[min(2, len(larger_sizes) - 1)]
        elif len(larger_sizes) == 2:
            h1_size = larger_sizes[0]
            h2_size = larger_sizes[1]
        elif len(larger_sizes) == 1:
            h1_size = larger_sizes[0]

        self.font_thresholds = {
            'body': body_size,
            'h1': h1_size,
            'h2': h2_size,
            'h3': h3_size,
        }
        logger.info(f"字体阈值: body={body_size}, h1>={h1_size}, h2>={h2_size}, h3>={h3_size}")

    # -------- 步骤3：提取论文标题 --------

    def _extract_title(self) -> str:
        """
        提取论文标题：取全文最大字号的文本（通常出现在第一页顶部）。
        """
        if not self.blocks:
            return ""

        # 只看第一页的前几个块
        first_page_blocks = [b for b in self.blocks if b.page_num == 1]
        if not first_page_blocks:
            first_page_blocks = self.blocks[:10]

        # 找最大字体
        max_size = max(b.font_size for b in first_page_blocks)
        title_candidates = [
            b for b in first_page_blocks
            if b.font_size >= max_size - 0.5 and len(b.text) > 5
        ]

        if title_candidates:
            title = " ".join(b.text for b in title_candidates[:3])
            # 截断过长的标题
            if len(title) > 300:
                title = title_candidates[0].text
            logger.info(f"提取到标题: {title[:100]}...")
            return title.strip()

        return ""

    # -------- 步骤4：提取摘要 --------

    def _extract_abstract(self) -> str:
        """
        提取摘要：找到 "Abstract" 或 "摘要" 标题，提取其后直到下一个标题的文本。
        支持中英双语论文。
        """
        abstract_started = False
        abstract_parts = []
        abstract_triggers = ['abstract', '摘要']

        for block in self.blocks:
            text_clean = block.text.strip().lower().rstrip('.')

            if any(text_clean == t or text_clean.startswith(t) for t in abstract_triggers):
                abstract_started = True
                continue

            if abstract_started:
                # 遇到下一个标题就停止
                if _is_heading_text(block.text) and block.font_size >= self.font_thresholds.get('h3', 11):
                    break
                abstract_parts.append(block.text)

        abstract = " ".join(abstract_parts).strip()
        # 截断过长的摘要
        if len(abstract) > 3000:
            abstract = abstract[:3000] + "..."

        logger.info(f"提取到摘要: {len(abstract)} 字符")
        return abstract

    # -------- 步骤5：构建章节树 --------

    def _build_section_tree(self) -> list[Section]:
        """
        根据标题检测和字体层级，构建树形章节结构。
        返回一级章节列表，每个一级章节下递归嵌套子章节。
        """
        if not self.blocks:
            return []

        # 第一步：找出所有标题块
        heading_indices: list[tuple[int, int]] = []  # [(block_index, level), ...]

        for i, block in enumerate(self.blocks):
            if _is_heading_text(block.text):
                level = _guess_level(block.text, block.font_size, self.font_thresholds)
                heading_indices.append((i, level))

        if not heading_indices:
            # 没有检测到标题，整个文档作为一个章节
            full_text = "\n".join(b.text for b in self.blocks)
            return [Section(
                title="全文",
                level=1,
                page_num=1,
                content=full_text,
            )]

        # 第二步：按标题分块，构建章节树
        sections = []
        stack: list[Section] = []  # 用栈管理嵌套关系

        for idx, (block_idx, level) in enumerate(heading_indices):
            block = self.blocks[block_idx]

            # 确定此标题的正文范围：从当前块到下一个标题块
            next_block_idx = (
                heading_indices[idx + 1][0] if idx + 1 < len(heading_indices)
                else len(self.blocks)
            )

            # 收集正文（不含标题块本身）
            content_parts = []
            for j in range(block_idx + 1, next_block_idx):
                # 跳过子标题（它们会被作为子节点处理）
                is_sub_heading = any(
                    h_idx == j for h_idx, _ in heading_indices
                )
                if not is_sub_heading:
                    content_parts.append(self.blocks[j].text)

            section = Section(
                title=block.text.strip(),
                level=min(level, 3),
                page_num=block.page_num,
                content="\n".join(content_parts).strip(),
                font_size=block.font_size,
            )

            # 用栈确定父节点
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                stack[-1].children.append(section)
            else:
                sections.append(section)

            stack.append(section)

        logger.info(
            f"构建章节树完成: {len(sections)} 个一级章节, "
            f"共 {self._count_sections(sections)} 个章节节点"
        )
        return sections

    @staticmethod
    def _count_sections(sections: list[Section]) -> int:
        """递归统计章节总数"""
        count = len(sections)
        for s in sections:
            count += PDFParser._count_sections(s.children)
        return count

    # -------- 主流程 --------

    def parse(self, paper_id: str) -> ParsedPaper:
        """
        执行完整的 PDF 解析流程。

        Args:
            paper_id: 论文ID

        Returns:
            ParsedPaper 结构化论文对象
        """
        logger.info(f"开始解析论文 {paper_id}: {self.pdf_path}")

        # 步骤1：提取文本块
        self._extract_blocks()

        # 步骤2：分析字体大小分布
        self._analyze_font_sizes()

        # 步骤3：提取标题
        title = self._extract_title()

        # 步骤4：提取摘要
        abstract = self._extract_abstract()

        # 步骤5：构建章节树
        sections = self._build_section_tree()

        # 全文纯文本
        raw_text = "\n".join(b.text for b in self.blocks)

        parsed = ParsedPaper(
            paper_id=paper_id,
            title=title,
            authors="",  # 作者提取较复杂，后续可增强
            abstract=abstract,
            sections=sections,
            page_count=len(self.doc) if self.doc else 0,
            raw_text=raw_text,
        )

        # 关闭文档
        if self.doc:
            self.doc.close()

        logger.success(f"论文 {paper_id} 解析完成: {parsed.page_count} 页, "
                       f"{self._count_sections(sections)} 个章节")
        return parsed


# ============================================================
# 对外服务接口
# ============================================================

def parse_pdf(paper_id: str, pdf_path: str, force_reparse: bool = False) -> ParsedPaper:
    """
    解析 PDF 论文，支持 JSON 缓存。

    Args:
        paper_id: 论文ID
        pdf_path: PDF 文件路径
        force_reparse: 是否强制重新解析（忽略缓存）

    Returns:
        ParsedPaper 结构化论文对象
    """
    # 检查缓存
    cache_path = _get_cache_path(paper_id)
    if not force_reparse and os.path.exists(cache_path):
        logger.info(f"从缓存加载解析结果: {cache_path}")
        return _load_from_cache(cache_path)

    # 执行解析
    parser = PDFParser(pdf_path)
    parsed = parser.parse(paper_id)

    # 保存缓存
    _save_to_cache(parsed, cache_path)

    return parsed


def _get_cache_path(paper_id: str) -> str:
    """获取解析结果缓存路径"""
    os.makedirs(settings.parsed_storage_dir, exist_ok=True)
    return os.path.join(settings.parsed_storage_dir, f"{paper_id}.json")


def _save_to_cache(parsed: ParsedPaper, cache_path: str) -> None:
    """将解析结果保存为 JSON"""
    try:
        data = {
            "paper_id": parsed.paper_id,
            "title": parsed.title,
            "authors": parsed.authors,
            "abstract": parsed.abstract,
            "sections": _sections_to_dict(parsed.sections),
            "page_count": parsed.page_count,
            "raw_text": parsed.raw_text,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"解析结果已缓存: {cache_path}")
    except Exception as e:
        logger.error(f"缓存保存失败: {e}")


def _load_from_cache(cache_path: str) -> ParsedPaper:
    """从 JSON 缓存加载解析结果"""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ParsedPaper(
        paper_id=data["paper_id"],
        title=data["title"],
        authors=data.get("authors", ""),
        abstract=data["abstract"],
        sections=_dict_to_sections(data["sections"]),
        page_count=data["page_count"],
        raw_text=data.get("raw_text", ""),
    )


def _sections_to_dict(sections: list[Section]) -> list[dict]:
    """递归将 Section 列表转为 dict（用于 JSON 序列化）"""
    result = []
    for s in sections:
        result.append({
            "title": s.title,
            "level": s.level,
            "page_num": s.page_num,
            "content": s.content,
            "font_size": s.font_size,
            "children": _sections_to_dict(s.children),
        })
    return result


def _dict_to_sections(data: list[dict]) -> list[Section]:
    """递归将 dict 转为 Section 列表"""
    sections = []
    for d in data:
        sections.append(Section(
            title=d["title"],
            level=d["level"],
            page_num=d["page_num"],
            content=d["content"],
            font_size=d.get("font_size", 0.0),
            children=_dict_to_sections(d.get("children", [])),
        ))
    return sections
