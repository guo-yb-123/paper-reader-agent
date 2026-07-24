"""
论文章节工具模块
- 提供跨 Agent 共享的论文结构遍历函数
- 避免各 Agent 之间互相导入
"""


def extract_toc(sections: list[dict]) -> list[dict]:
    """递归提取目录结构（所有层级的标题）"""
    toc = []
    for s in sections:
        toc.append({
            "title": s.get("title", ""),
            "level": s.get("level", 1),
        })
        children = s.get("children", [])
        if children:
            toc.extend(extract_toc(children))
    return toc


def collect_section_contents(paper_structure: dict, assigned_sections: list[str]) -> str:
    """
    从论文结构中收集指定章节的完整内容（含子章节）。

    Args:
        paper_structure: 解析后的论文结构 dict
        assigned_sections: Planner 分配的章节标题列表

    Returns:
        拼接后的章节内容文本
    """
    sections = paper_structure.get("sections", [])

    result_parts = []

    # Abstract 特殊处理
    abstract = paper_structure.get("abstract", "")
    if abstract and "abstract" in [s.lower() for s in assigned_sections]:
        result_parts.append(f"### Abstract\n\n{abstract}\n")

    # 递归收集匹配章节
    for section in sections:
        _collect_recursive(section, assigned_sections, result_parts)

    if not result_parts:
        # 如果没有匹配到任何章节，返回全部内容
        for section in sections:
            _append_section_full(section, result_parts)

    return "\n\n".join(result_parts)


def _collect_recursive(
    section: dict,
    assigned_sections: list[str],
    result: list[str],
) -> None:
    """递归收集匹配的章节内容"""
    title = section.get("title", "")

    # 模糊匹配
    is_assigned = any(
        assigned_title.lower() in title.lower() or title.lower() in assigned_title.lower()
        for assigned_title in assigned_sections
    )

    if is_assigned:
        _append_section_full(section, result)
    else:
        for child in section.get("children", []):
            _collect_recursive(child, assigned_sections, result)


def _append_section_full(section: dict, result: list[str]) -> None:
    """将章节及其子章节的完整内容追加到结果（递归最多3层）"""
    title = section.get("title", "")
    content = section.get("content", "")

    lines = [f"## {title}\n"]
    if content.strip():
        lines.append(content)
    lines.append("")

    for child in section.get("children", []):
        child_title = child.get("title", "")
        child_content = child.get("content", "")
        lines.append(f"### {child_title}\n")
        if child_content.strip():
            lines.append(child_content)
        lines.append("")

        for grandchild in child.get("children", []):
            gc_title = grandchild.get("title", "")
            gc_content = grandchild.get("content", "")
            lines.append(f"#### {gc_title}\n")
            if gc_content.strip():
                lines.append(gc_content)
            lines.append("")

    result.append("\n".join(lines))
