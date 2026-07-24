"""
端到端集成测试
- Mock LLM 响应，验证完整的 LangGraph 分析工作流和对话工作流
- 测试 6 个 Agent 的完整协作链路: Planner → Readers → Critic → Summarizer
- 测试对话工作流: Router → Expert 路由分发
"""
import json
import pytest
from unittest.mock import patch, MagicMock


# ============================================================
# Mock 工具
# ============================================================

# Mock LLM 响应内容
MOCK_PLANNER_OUTPUT = json.dumps({
    "paper_topic": "A Novel Deep Learning Approach for NLP",
    "reading_plan": {
        "method_reader": {
            "sections": ["Method", "Model Architecture"],
            "focus_points": ["核心算法", "模型结构"],
            "expected_output": "方法部分结构化总结",
        },
        "experiment_reader": {
            "sections": ["Experiment", "Results"],
            "focus_points": ["数据集", "基线对比"],
            "expected_output": "实验部分结构化总结",
        },
        "background_reader": {
            "sections": ["Introduction", "Related Work"],
            "focus_points": ["研究动机", "领域现状"],
            "expected_output": "背景与相关工作总结",
        },
    },
    "overall_strategy": "各Reader并行精读各自分配的章节",
})

MOCK_BACKGROUND_OUTPUT = """## 研究背景与相关工作总结
### 1. 研究问题与动机
本文研究深度学习在NLP领域的应用...

### 2. 领域现状
当前主流方法包括BERT、GPT等预训练模型...

### 3. 本文贡献
提出了一个新的注意力机制..."""

MOCK_METHOD_OUTPUT = """## 方法部分结构化总结
### 1. 核心创新点
提出了Multi-Head Sparse Attention机制...

### 2. 模型架构
整体采用Encoder-Decoder架构...

### 3. 关键技术细节
使用残差连接和层归一化..."""

MOCK_EXPERIMENT_OUTPUT = """## 实验结果结构化总结
### 1. 实验设置
使用GLUE、SuperGLUE等标准数据集...

### 2. 基线对比
在7个任务上超越BERT-large...

### 3. 消融实验
注意力头数从12减少到8仍有竞争力..."""

MOCK_CRITIQUE_OUTPUT = """## 批判性审阅报告
### 🔴高风险
- 缺乏在大规模数据集上的验证
### 🟡中风险
- 消融实验对关键超参数的覆盖不足
### 总体评分: 7/10"""

MOCK_FINAL_REPORT = """# 论文精读报告

## 📄 论文信息
- 标题：A Novel Deep Learning Approach for NLP

## 📌 一句话速览
本文提出了Multi-Head Sparse Attention...

## 🎯 研究背景与动机
深度学习在NLP领域的应用...

## 🔧 方法与技术
提出了Multi-Head Sparse Attention机制...

## 📊 实验与评估
在GLUE、SuperGLUE等标准数据集上...

## ⚠️ 批判性审视
缺乏大规模验证...

## 📝 总结与思考
本文贡献了一个新颖的注意力机制..."""

MOCK_ROUTER_OUTPUT = json.dumps({
    "route": "method_expert",
    "reason": "问题涉及模型架构和方法",
    "confidence": "high",
})

MOCK_CHAT_ANSWER = """这篇论文的核心创新是提出了Multi-Head Sparse Attention机制。
该机制通过稀疏化注意力矩阵，将计算复杂度从 O(n²) 降低到 O(n log n)，
同时保持了与标准注意力机制相当的性能。

具体来说：
1. **稀疏注意力模式**：只计算最相关的 token 对之间的注意力
2. **多尺度稀疏**：在不同层使用不同的稀疏模式
3. **可学习稀疏模式**：稀疏模式通过可训练参数动态调整"""


# ============================================================
# 辅助函数
# ============================================================

def _make_mock_llm(response_text: str):
    """创建返回指定文本的 Mock LLM"""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ============================================================
# 测试：分析工作流 (6 Agent 全链路)
# ============================================================

class TestAnalysisWorkflowE2E:
    """端到端测试：Planner → Readers → Critic → Summarizer"""

    @pytest.fixture
    def mock_llms(self):
        """
        Mock 所有 LLM 调用。

        分析工作流中共调用 6 次 LLM:
        - Planner (1次)
        - Background Reader (1次)
        - Method Reader (1次)
        - Experiment Reader (1次)
        - Critic (1次)
        - Summarizer (1次)

        使用 side_effect 按顺序返回不同内容。
        """
        mock_llm_instances = [
            _make_mock_llm(MOCK_PLANNER_OUTPUT),
            _make_mock_llm(MOCK_BACKGROUND_OUTPUT),
            _make_mock_llm(MOCK_METHOD_OUTPUT),
            _make_mock_llm(MOCK_EXPERIMENT_OUTPUT),
            _make_mock_llm(MOCK_CRITIQUE_OUTPUT),
            _make_mock_llm(MOCK_FINAL_REPORT),
        ]
        return mock_llm_instances

    @pytest.fixture
    def paper_structure(self):
        """Mock 论文结构（模拟 PDF 解析后的结构化数据）"""
        return {
            "paper_id": "test-uuid-001",
            "title": "A Novel Deep Learning Approach for NLP",
            "authors": "Zhang et al.",
            "abstract": "We propose a novel attention mechanism for NLP tasks...",
            "sections": [
                {
                    "title": "Introduction",
                    "level": 1,
                    "page_num": 1,
                    "content": "Deep learning has revolutionized NLP...",
                    "children": [],
                },
                {
                    "title": "Related Work",
                    "level": 1,
                    "page_num": 2,
                    "content": "Previous work includes BERT, GPT...",
                    "children": [],
                },
                {
                    "title": "Method",
                    "level": 1,
                    "page_num": 3,
                    "content": "Our proposed method consists of...",
                    "children": [
                        {
                            "title": "Model Architecture",
                            "level": 2,
                            "page_num": 4,
                            "content": "The architecture includes encoder and decoder...",
                            "children": [],
                        },
                    ],
                },
                {
                    "title": "Experiment",
                    "level": 1,
                    "page_num": 6,
                    "content": "We evaluate on GLUE benchmark...",
                    "children": [],
                },
                {
                    "title": "Results",
                    "level": 1,
                    "page_num": 8,
                    "content": "Our method achieves SOTA on 7 tasks...",
                    "children": [],
                },
            ],
            "page_count": 10,
            "raw_text": "Full paper text...",
        }

    def test_full_analysis_workflow_executes_end_to_end(
        self, mock_llms, paper_structure
    ):
        """
        验证完整的分析工作流端到端执行。

        流程:
        1. Planner 生成阅读计划
        2. 三个 Reader 并行分析
        3. Critic 批判性审阅
        4. Summarizer 生成最终报告

        预期:
        - 所有 6 个节点的输出都有值
        - 没有错误字段
        - 最终报告包含预期内容
        """
        from app.graph.workflow import build_analysis_workflow

        # Mock get_llm，按顺序返回预设的 LLM 实例
        llm_iter = iter(mock_llms)

        with patch("app.agents.planner.get_llm", return_value=next(llm_iter)), \
             patch("app.agents.background_reader.get_llm", return_value=next(llm_iter)), \
             patch("app.agents.method_reader.get_llm", return_value=next(llm_iter)), \
             patch("app.agents.experiment_reader.get_llm", return_value=next(llm_iter)), \
             patch("app.agents.critic.get_llm", return_value=next(llm_iter)), \
             patch("app.agents.summarizer.get_llm", return_value=next(llm_iter)):

            workflow = build_analysis_workflow()
            app = workflow.compile()

            initial_state = {
                "paper_id": "test-uuid-001",
                "pdf_path": "/tmp/test.pdf",
                "paper_structure": paper_structure,
            }

            result = app.invoke(initial_state, {"recursion_limit": 50})

            # ── 断言 ──

            # 1. 没有错误
            assert result.get("error") is None or result.get("error") == "", \
                f"工作流执行出错: {result.get('error')}"

            # 2. Planner 产出阅读计划
            reading_plan = result.get("reading_plan")
            assert reading_plan is not None
            assert "reading_plan" in reading_plan
            assert "method_reader" in reading_plan["reading_plan"]
            assert len(reading_plan["reading_plan"]["method_reader"]["sections"]) > 0

            # 3. 三个 Reader 都有输出
            assert result.get("background_summary") is not None
            assert len(result["background_summary"]) > 50
            assert "研究背景" in result["background_summary"]

            assert result.get("method_summary") is not None
            assert len(result["method_summary"]) > 50
            assert "核心创新" in result["method_summary"]

            assert result.get("experiment_summary") is not None
            assert len(result["experiment_summary"]) > 50
            assert "实验" in result["experiment_summary"]

            # 4. Critic 有审阅输出
            assert result.get("critique") is not None
            assert len(result["critique"]) > 50
            assert "风险" in result["critique"]

            # 5. Summarizer 生成最终报告
            assert result.get("final_report") is not None
            assert len(result["final_report"]) > 100
            assert "论文精读报告" in result["final_report"]

    def test_workflow_handles_planner_failure(self, paper_structure):
        """
        验证 Planner 异常时的容错：LLM 调用失败后 run_planner 返回降级计划，
        工作流不会中断，而是继续执行（使用关键词分配的降级计划）。
        """
        from app.graph.workflow import build_analysis_workflow

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM 服务不可用")

        with patch("app.agents.planner.get_llm", return_value=mock_llm):
            workflow = build_analysis_workflow()
            app = workflow.compile()

            initial_state = {
                "paper_id": "test-uuid-001",
                "pdf_path": "/tmp/test.pdf",
                "paper_structure": paper_structure,
            }

            result = app.invoke(initial_state, {"recursion_limit": 50})

            # Planner 失败后会返回降级计划（关键词分配），而非设置 error 字段
            # 工作流继续运行才是正确的容错行为
            reading_plan = result.get("reading_plan")
            assert reading_plan is not None
            assert "reading_plan" in reading_plan
            # 降级模式应有 overall_strategy 标记
            assert "overall_strategy" in reading_plan

    def test_reader_failure_does_not_block_critic(self, paper_structure):
        """
        验证单个 Reader 失败不阻塞 Critic 汇聚：
        Method Reader 异常，但 Background/Experiment 正常，
        Critic 仍应收到部分结果并给出审阅。
        """
        from app.graph.workflow import build_analysis_workflow

        # Planner LLM
        mock_planner_llm = _make_mock_llm(MOCK_PLANNER_OUTPUT)
        # Background Reader: 正常
        mock_bg_llm = _make_mock_llm(MOCK_BACKGROUND_OUTPUT)
        # Method Reader: 异常
        mock_method_llm = MagicMock()
        mock_method_llm.invoke.side_effect = ConnectionError("网络超时")
        # Experiment Reader: 正常
        mock_exp_llm = _make_mock_llm(MOCK_EXPERIMENT_OUTPUT)
        # Critic: 正常
        mock_critic_llm = _make_mock_llm(MOCK_CRITIQUE_OUTPUT)
        # Summarizer: 正常
        mock_summarizer_llm = _make_mock_llm(MOCK_FINAL_REPORT)

        with patch("app.agents.planner.get_llm", return_value=mock_planner_llm), \
             patch("app.agents.background_reader.get_llm", return_value=mock_bg_llm), \
             patch("app.agents.method_reader.get_llm", return_value=mock_method_llm), \
             patch("app.agents.experiment_reader.get_llm", return_value=mock_exp_llm), \
             patch("app.agents.critic.get_llm", return_value=mock_critic_llm), \
             patch("app.agents.summarizer.get_llm", return_value=mock_summarizer_llm):

            workflow = build_analysis_workflow()
            app = workflow.compile()

            initial_state = {
                "paper_id": "test-uuid-001",
                "pdf_path": "/tmp/test.pdf",
                "paper_structure": paper_structure,
            }

            result = app.invoke(initial_state, {"recursion_limit": 50})

            # Method Reader 应有错误标记
            assert "分析失败" in result.get("method_summary", "")

            # Background 和 Experiment 应正常
            assert "研究背景" in result.get("background_summary", "")
            assert "实验" in result.get("experiment_summary", "")

            # Critic 不应阻塞——即使 Method 失败，仍有 BG 和 Exp 的输出
            assert result.get("critique") is not None
            assert len(result["critique"]) > 20

            # Summarizer 应生成报告（基于部分可用数据）
            assert result.get("final_report") is not None
            assert len(result["final_report"]) > 50


# ============================================================
# 测试：对话工作流 (Router → Expert)
# ============================================================

class TestChatWorkflowE2E:
    """端到端测试：Router → Expert Agent 对话路由"""

    def test_chat_workflow_routes_to_correct_expert(self):
        """
        验证 Router 将问题正确路由到对应专家并返回答案。
        """
        from app.graph.workflow import build_chat_workflow

        mock_router_llm = _make_mock_llm(MOCK_ROUTER_OUTPUT)
        mock_expert_llm = _make_mock_llm(MOCK_CHAT_ANSWER)

        with patch("app.agents.router.get_llm", return_value=mock_router_llm), \
             patch("app.agents.chat_handler.get_llm", return_value=mock_expert_llm):

            workflow = build_chat_workflow()
            app = workflow.compile()

            chat_state = {
                "paper_id": "test-uuid-001",
                "paper_structure": {"title": "Test Paper", "sections": []},
                "background_summary": "背景分析...",
                "method_summary": "方法分析...",
                "experiment_summary": "实验分析...",
                "critique": "批判审阅...",
                "final_report": "完整报告...",
                "current_question": "这个模型的架构是什么？",
                "messages": [],
                "retrieved_chunks": [],
            }

            result = app.invoke(chat_state, {"recursion_limit": 10})

            # 路由正确
            router_decision = result.get("router_decision", {})
            assert router_decision.get("route") == "method_expert"

            # 专家回答了问题
            answer = result.get("answer", "")
            assert "Multi-Head Sparse Attention" in answer
            assert len(answer) > 100

    def test_chat_workflow_router_fallback_on_bad_json(self):
        """
        验证 Router LLM 返回非法 JSON 时降级到关键词路由。
        问题包含"方法"关键词，应降级到 method_expert。
        """
        from app.graph.workflow import build_chat_workflow

        # Router 返回非法 JSON，触发降级
        mock_router_llm = _make_mock_llm("not valid json at all {{{bad")
        mock_expert_llm = _make_mock_llm(MOCK_CHAT_ANSWER)

        with patch("app.agents.router.get_llm", return_value=mock_router_llm), \
             patch("app.agents.chat_handler.get_llm", return_value=mock_expert_llm):

            workflow = build_chat_workflow()
            app = workflow.compile()

            chat_state = {
                "paper_id": "test-uuid-001",
                "paper_structure": {"title": "Test Paper", "sections": []},
                "background_summary": "",
                "method_summary": "",
                "experiment_summary": "",
                "critique": "",
                "final_report": "",
                "current_question": "这篇论文的方法是什么？",
                "messages": [],
                "retrieved_chunks": [],
            }

            result = app.invoke(chat_state, {"recursion_limit": 10})

            # 降级路由到 method_expert（因为问题包含"方法"）
            router_decision = result.get("router_decision", {})
            assert router_decision.get("route") in ("method_expert", "general")
            assert router_decision.get("confidence") == "low"

            # 答案仍然生成
            assert result.get("answer") is not None
            assert len(result.get("answer", "")) > 20

    def test_router_node_failure_falls_back_to_general(self):
        """
        验证 Router 节点完全崩溃时 fallback 到关键词降级路由。
        Router 的 run_router 内部 catch 所有异常并走关键词降级，
        不会设置 error，而是返回 matched route（或 general）。
        """
        from app.graph.workflow import build_chat_workflow

        # Router LLM 完全崩溃
        mock_router_llm = MagicMock()
        mock_router_llm.invoke.side_effect = RuntimeError("API 崩溃")
        mock_expert_llm = _make_mock_llm("这是一个通用回答。")

        with patch("app.agents.router.get_llm", return_value=mock_router_llm), \
             patch("app.agents.chat_handler.get_llm", return_value=mock_expert_llm):

            workflow = build_chat_workflow()
            app = workflow.compile()

            chat_state = {
                "paper_id": "test-uuid-001",
                "paper_structure": {"title": "Test Paper", "sections": []},
                "background_summary": "",
                "method_summary": "",
                "experiment_summary": "",
                "critique": "",
                "final_report": "",
                "current_question": "hello world",
                "messages": [],
                "retrieved_chunks": [],
            }

            result = app.invoke(chat_state, {"recursion_limit": 10})

            # Router 降级：问题不含任何关键词 → general
            router_decision = result.get("router_decision", {})
            assert router_decision.get("route") == "general"
            assert router_decision.get("confidence") == "low"

            # 专家仍然正常工作
            assert result.get("answer") is not None


# ============================================================
# 测试：重试机制
# ============================================================

class TestRetryMechanism:
    """测试 LLM 调用重试机制"""

    def test_retry_on_connection_error(self):
        """验证 ConnectionError 触发重试，最终成功"""
        from app.core.retry_utils import invoke_llm_with_retry

        mock_llm = MagicMock()
        # 前 2 次失败，第 3 次成功
        mock_response = MagicMock()
        mock_response.content = "success after retry"
        mock_llm.invoke.side_effect = [
            ConnectionError("连接超时"),
            ConnectionError("连接超时"),
            mock_response,
        ]

        response = invoke_llm_with_retry(mock_llm, ["test message"])
        assert response.content == "success after retry"
        assert mock_llm.invoke.call_count == 3

    def test_no_retry_on_quota_error(self):
        """验证额度耗尽不重试（直接抛出）"""
        from app.core.retry_utils import invoke_llm_with_retry

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Insufficient_quota: 额度已耗尽")

        with pytest.raises(Exception, match="Insufficient_quota"):
            invoke_llm_with_retry(mock_llm, ["test message"])

        # 硬错误不重试：只调用了 1 次
        assert mock_llm.invoke.call_count == 1

    def test_retry_exhausted_after_all_failures(self):
        """验证连续失败 3 次后抛出异常"""
        from app.core.retry_utils import invoke_llm_with_retry

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("持续网络故障")

        with pytest.raises(ConnectionError):
            invoke_llm_with_retry(mock_llm, ["test message"])

        assert mock_llm.invoke.call_count == 3


# ============================================================
# 测试：工作流节点汇聚（LangGraph barrier）
# ============================================================

class TestWorkflowBarrier:
    """测试 LangGraph 并行分支的 barrier 语义"""

    @pytest.fixture
    def paper_structure(self):
        """Mock 论文结构"""
        return {
            "paper_id": "test-uuid-001",
            "title": "A Novel Deep Learning Approach for NLP",
            "authors": "Zhang et al.",
            "abstract": "We propose a novel attention mechanism...",
            "sections": [
                {"title": "Introduction", "level": 1, "page_num": 1,
                 "content": "Deep learning has revolutionized NLP...", "children": []},
                {"title": "Method", "level": 1, "page_num": 3,
                 "content": "Our proposed method consists of...", "children": []},
                {"title": "Experiment", "level": 1, "page_num": 6,
                 "content": "We evaluate on GLUE benchmark...", "children": []},
            ],
            "page_count": 10,
            "raw_text": "Full paper text...",
        }

    def test_critic_waits_for_all_readers(self, paper_structure):
        """
        验证 Critic 在三个 Reader 全部完成后才执行。
        通过 Mock LLM 验证：Critic 收到的输入包含所有 Reader 的输出。
        """
        from app.graph.workflow import build_analysis_workflow

        mock_planner_llm = _make_mock_llm(MOCK_PLANNER_OUTPUT)
        mock_bg_llm = _make_mock_llm(MOCK_BACKGROUND_OUTPUT)
        mock_method_llm = _make_mock_llm(MOCK_METHOD_OUTPUT)
        mock_exp_llm = _make_mock_llm(MOCK_EXPERIMENT_OUTPUT)

        # 用 Mock 捕获 Critic 收到的 messages
        mock_critic_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_CRITIQUE_OUTPUT
        mock_critic_llm.invoke.return_value = mock_response

        mock_summ_llm = _make_mock_llm(MOCK_FINAL_REPORT)

        with patch("app.agents.planner.get_llm", return_value=mock_planner_llm), \
             patch("app.agents.background_reader.get_llm", return_value=mock_bg_llm), \
             patch("app.agents.method_reader.get_llm", return_value=mock_method_llm), \
             patch("app.agents.experiment_reader.get_llm", return_value=mock_exp_llm), \
             patch("app.agents.critic.get_llm", return_value=mock_critic_llm), \
             patch("app.agents.summarizer.get_llm", return_value=mock_summ_llm):

            workflow = build_analysis_workflow()
            app = workflow.compile()

            initial_state = {
                "paper_id": "test-uuid-001",
                "pdf_path": "/tmp/test.pdf",
                "paper_structure": paper_structure,
            }

            result = app.invoke(initial_state, {"recursion_limit": 50})

            # 验证 Critic 被调用了（不是被跳过）
            mock_critic_llm.invoke.assert_called_once()

            # 从 Critic 的调用参数中提取输入消息
            call_args = mock_critic_llm.invoke.call_args[0][0]
            human_message = call_args[1].content  # HumanMessage 是第二个元素

            # Critic 的输入应包含所有三个 Reader 的输出
            assert "Background Reader" in human_message or \
                   MOCK_BACKGROUND_OUTPUT[:50] in human_message
            assert "Method Reader" in human_message or \
                   MOCK_METHOD_OUTPUT[:50] in human_message
            assert "Experiment Reader" in human_message or \
                   MOCK_EXPERIMENT_OUTPUT[:50] in human_message

            # 最终报告正常
            assert result.get("final_report") is not None
