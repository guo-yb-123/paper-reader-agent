"""
FastAPI 接口集成测试
使用 TestClient 测试 API 端点（Mock 所有外部依赖）
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock


"""
FastAPI 接口集成测试
使用 TestClient 测试 API 端点（Mock 所有外部依赖）

策略：手动构建 FastAPI app，用 dependency_overrides 替换 get_db，
避开 lifespan 中的数据库连接。
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware


def _make_mock_db():
    """Create a mock DB session where all queries return None/empty."""
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.limit.return_value = q
    q.first.return_value = None
    q.all.return_value = []
    s = MagicMock()
    s.query.return_value = q
    return s


@pytest.fixture
def client():
    """构建一个不依赖数据库的测试 app"""
    from app.api import upload, analyze, chat

    app = FastAPI(title="Test")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/")
    async def root():
        return {"service": "Paper Reader Agent", "status": "running"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    # 注册路由
    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(analyze.router, prefix="/api", tags=["analyze"])
    app.include_router(chat.router, prefix="/api", tags=["chat"])

    # 用 dependency_overrides 替换 get_db
    from app.core.database import get_db

    def override_get_db():
        yield _make_mock_db()

    app.dependency_overrides[get_db] = override_get_db

    # Mock celery in all modules that use it
    with patch("app.api.analyze.celery_app.send_task"), \
         patch("app.api.analyze.celery_app.AsyncResult"):
        with TestClient(app) as c:
            yield c


class TestHealthCheck:
    """健康检查端点"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Paper Reader Agent"
        assert data["status"] == "running"

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestPaperEndpoints:
    """论文管理接口"""

    def test_get_papers_returns_list(self, client):
        response = client.get("/api/papers")
        assert response.status_code == 200
        data = response.json()
        assert "papers" in data
        assert "total" in data
        assert isinstance(data["papers"], list)

    def test_analyze_nonexistent_paper_returns_404(self, client):
        response = client.post("/api/analyze/nonexistent-id")
        assert response.status_code == 404

    def test_report_nonexistent_paper_returns_404(self, client):
        response = client.get("/api/report/nonexistent-id")
        assert response.status_code == 404


class TestErrorHandling:
    """异常响应格式"""

    def test_upload_without_file_returns_422(self, client):
        """无文件上传应返回 422 参数校验错误"""
        response = client.post("/api/upload")
        assert response.status_code == 422

    def test_chat_without_body_returns_422(self, client):
        """空 body 请求应返回 422"""
        response = client.post("/api/chat/test-id")
        assert response.status_code == 422

    def test_chat_empty_question_rejected(self, client):
        """空问题应被 Pydantic 校验拦截"""
        response = client.post("/api/chat/test-id", json={
            "paper_id": "test-id",
            "question": "",
        })
        assert response.status_code == 422
