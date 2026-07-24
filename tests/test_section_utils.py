"""
测试论文章节工具模块
"""
import pytest
from app.services.section_utils import extract_toc, collect_section_contents


class TestExtractToc:
    """测试目录提取"""

    def test_empty_sections(self):
        assert extract_toc([]) == []

    def test_flat_sections(self):
        sections = [
            {"title": "Introduction", "level": 1, "content": "", "children": []},
            {"title": "Method", "level": 1, "content": "", "children": []},
        ]
        toc = extract_toc(sections)
        assert len(toc) == 2
        assert toc[0] == {"title": "Introduction", "level": 1}
        assert toc[1] == {"title": "Method", "level": 1}

    def test_nested_sections(self):
        sections = [
            {
                "title": "Method",
                "level": 1,
                "content": "",
                "children": [
                    {"title": "Architecture", "level": 2, "content": "", "children": []},
                    {"title": "Training", "level": 2, "content": "", "children": [
                        {"title": "Loss Function", "level": 3, "content": "", "children": []},
                    ]},
                ],
            },
        ]
        toc = extract_toc(sections)
        # 递归提取所有层级标题：Method, Architecture, Training, Loss Function = 4
        assert len(toc) == 4
        titles = [t["title"] for t in toc]
        assert titles == ["Method", "Architecture", "Training", "Loss Function"]

    def test_missing_fields(self):
        sections = [{"title": "Test"}]
        toc = extract_toc(sections)
        assert toc[0]["level"] == 1  # default


class TestCollectSectionContents:
    """测试章节内容收集"""

    def make_structure(self, title="Test Paper", abstract="Test abstract", sections=None):
        return {
            "title": title,
            "abstract": abstract,
            "sections": sections or [],
        }

    def test_collect_abstract(self):
        paper = self.make_structure(abstract="This is an abstract.")
        result = collect_section_contents(paper, ["abstract"])
        assert "Abstract" in result
        assert "This is an abstract" in result

    def test_collect_by_section_title(self):
        paper = self.make_structure(sections=[
            {
                "title": "Method",
                "level": 1,
                "content": "Method content here.",
                "children": [],
            },
            {
                "title": "Experiment",
                "level": 1,
                "content": "Experiment content here.",
                "children": [],
            },
        ])
        result = collect_section_contents(paper, ["Method"])
        assert "Method content here" in result
        assert "Experiment content here" not in result

    def test_fuzzy_match(self):
        """测试模糊匹配：分配 "Method" 应匹配 "Proposed Method" """
        paper = self.make_structure(sections=[
            {
                "title": "Proposed Method",
                "level": 1,
                "content": "Proposed content.",
                "children": [],
            },
        ])
        result = collect_section_contents(paper, ["Method"])
        assert "Proposed content" in result

    def test_no_match_returns_all(self):
        """没有匹配到任何章节时，返回全部内容"""
        paper = self.make_structure(sections=[
            {
                "title": "Introduction",
                "level": 1,
                "content": "Intro content.",
                "children": [],
            },
        ])
        result = collect_section_contents(paper, ["Nonexistent"])
        assert "Intro content" in result
