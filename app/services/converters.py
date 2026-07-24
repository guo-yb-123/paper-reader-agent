"""
数据结构转换工具模块
- 提供跨模块共享的 dataclass / dict 转换函数
- 避免各模块之间重复定义
"""


def parsed_paper_to_dict(parsed) -> dict:
    """将 ParsedPaper dataclass 转为 workflow state 可用的 dict"""
    def _section_to_dict(section):
        return {
            "title": section.title,
            "level": section.level,
            "page_num": section.page_num,
            "content": section.content,
            "children": [_section_to_dict(c) for c in section.children],
        }

    return {
        "paper_id": parsed.paper_id,
        "title": parsed.title,
        "authors": getattr(parsed, "authors", ""),
        "abstract": parsed.abstract,
        "sections": [_section_to_dict(s) for s in parsed.sections],
        "page_count": parsed.page_count,
        "raw_text": parsed.raw_text,
    }
