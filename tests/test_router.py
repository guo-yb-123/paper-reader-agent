"""
测试 Router Agent 降级路由逻辑
"""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.router import run_router


class TestRouterFallback:
    """测试 Router 的关键词降级路由（不依赖 LLM）"""

    def _mock_llm_failure(self):
        """模拟 LLM 调用失败（返回非法 JSON），触发降级路由"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "invalid json {{{"
        mock_llm.invoke.return_value = mock_response
        return patch("app.agents.router.get_llm", return_value=mock_llm)

    def test_background_route_chinese(self):
        with self._mock_llm_failure():
            result = run_router("这篇论文的研究背景是什么？")
        assert result["route"] == "background_expert"
        assert result["confidence"] == "low"

    def test_method_route_chinese(self):
        with self._mock_llm_failure():
            result = run_router("模型的架构是什么？")
        assert result["route"] == "method_expert"

    def test_experiment_route_chinese(self):
        with self._mock_llm_failure():
            result = run_router("实验用了什么数据集？")
        assert result["route"] == "experiment_expert"

    def test_critic_route_chinese(self):
        with self._mock_llm_failure():
            result = run_router("这篇论文有什么局限性？")
        assert result["route"] == "critic_expert"

    def test_summarizer_route_chinese(self):
        with self._mock_llm_failure():
            result = run_router("总结一下这篇论文")
        assert result["route"] == "summarizer"

    def test_general_fallback(self):
        with self._mock_llm_failure():
            result = run_router("xyzzy qwerty placeholder")
        assert result["route"] == "general"

    def test_english_method_keywords(self):
        with self._mock_llm_failure():
            result = run_router("What is the model architecture?")
        assert result["route"] == "method_expert"
