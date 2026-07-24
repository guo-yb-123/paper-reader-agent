"""
LangGraph 工作流状态定义
- 定义分析工作流和对话工作流的共享状态
"""
from typing import TypedDict, Optional


class AnalysisState(TypedDict):
    """论文分析工作流状态"""
    paper_id: str
    pdf_path: str
    paper_structure: Optional[dict]  # PDF解析后的结构化论文树
    reading_plan: Optional[dict]     # Planner输出的阅读计划
    method_summary: Optional[str]    # Method Reader 输出
    experiment_summary: Optional[str]  # Experiment Reader 输出
    background_summary: Optional[str]  # Background Reader 输出
    critique: Optional[str]          # Critic 输出
    final_report: Optional[str]      # Summarizer 最终Markdown报告
    # 错误追踪
    error: Optional[str]


class ChatState(TypedDict):
    """对话工作流状态"""
    paper_id: str
    paper_structure: Optional[dict]
    # 分析结果（对话前已通过分析工作流生成）
    background_summary: Optional[str]
    method_summary: Optional[str]
    experiment_summary: Optional[str]
    critique: Optional[str]
    final_report: Optional[str]
    # 对话相关
    messages: list[dict]                   # 对话历史 [{"role": "user/assistant", "content": "..."}]
    current_question: str                   # 当前用户问题
    router_decision: Optional[dict]         # Router的路由决策 {"route": "...", "reason": "..."}
    answer: Optional[str]                   # 回答结果
    # 语义检索结果（在 API 层通过 pgvector 检索填充）
    retrieved_chunks: Optional[list[dict]]  # [{"content": "...", "section_title": "...", "similarity": 0.95}]
