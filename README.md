# 多 Agent 论文精读系统

基于 **LangGraph** 的多 Agent 协作论文精读系统。上传 PDF 论文，自动调度 6 个 AI Agent 分工协作，生成结构化精读报告，支持 pgvector 语义检索的多轮对话问答。

## 架构

```
用户上传 PDF → FastAPI → Celery 异步任务
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
         PDF 结构化解析   文本分块 + 向量化   报告持久化
              │               │               │
              ↓               ↓               ↑
    LangGraph 分析工作流      pgvector        │
              │                               │
    ┌─────────┼─────────┐                     │
    ↓         ↓         ↓                     │
  Planner → 3路并行 Reader → Critic → Summarizer
    (规划)  (背景/方法/实验)  (审稿)    (编辑)
              │                               │
              └───────────────────────────────┘
                              ↓
              对话工作流: Router → 5 个 Expert Agent
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
         背景专家         方法专家         实验专家
              ↓               ↓               ↓
         批判专家         综述专家         通用回答
              │               │               │
              └───────────────┼───────────────┘
                              ↓
              语义检索 (pgvector 向量 + 关键词降级)
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **LangGraph 并行编排** | `StateGraph` + `Send` API 实现 Planner → 3 路并行 Reader 的 Map-Reduce 模式 |
| **6 Agent 分工协作** | Planner（规划）、Background/Method/Experiment Reader（三维精读）、Critic（审稿）、Summarizer（整合） |
| **智能对话路由** | Router 自动判断问题类型，分发到 5 个专家 Agent 回答 |
| **pgvector 语义检索** | 向量相似度检索 + 关键词降级策略，双重保障 |
| **LLM 容错重试** | tenacity 指数退避重试，智能跳过额度耗尽等不可重试错误 |
| **模型自动检测** | 启动时按序探测候选模型，自动选择第一个可用 |
| **PDF 结构化解析** | PyMuPDF 字体大小分布分析，自动识别标题层级，构建章节树 |
| **异步任务** | Celery 异步分析 + 进度上报 + 幂等保护 + 超时控制 |
| **多模型支持** | 通义千问多模型候选，Embedding 支持本地免费模型 |
| **容器化部署** | Docker Compose 五服务编排（PostgreSQL+pgvector / Redis / API / Worker / 前端） |

## 六个 Agent 分工

| Agent | 角色 | 输入 | 输出 |
|-------|------|------|------|
| **Planner** | 阅读规划师 | 论文标题/摘要/目录 | JSON 阅读计划，分配章节 |
| **Background Reader** | 领域研究者 | 引言/背景/相关工作 | 研究动机、领域现状、本文定位 |
| **Method Reader** | 算法工程师 | 方法论/模型章节 | 核心创新、模型架构、技术细节 |
| **Experiment Reader** | 实验分析师 | 实验/评估章节 | 数据集、基线对比、消融结论 |
| **Critic** | 审稿人 | 三个 Reader 的全部输出 | 批判性清单（创新/实验/局限性） |
| **Summarizer** | 科技编辑 | 四个前置 Agent 输出 | 完整 Markdown 精读报告 |

## 快速开始

```bash
# 1. 配置环境变量（系统环境或 .env 文件）
LLM_API_KEY=你的DashScope_API_Key

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动数据库
docker-compose up -d postgres redis

# 4. 三个终端分别启动
# 终端1: API
uvicorn app.main:app --host 0.0.0.0 --port 8002

# 终端2: Worker
celery -A app.tasks.celery_app worker --loglevel=INFO --pool=solo -n paper_worker

# 终端3: 前端
streamlit run frontend/app.py

# 5. 浏览器打开
# 前端: http://localhost:8501
# API 文档: http://localhost:8002/docs
```

## 配置

`.env` 或系统环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | — | 通义千问 API Key |
| `LLM_MODEL_CANDIDATES` | `qwen-turbo,qwen-plus` | 候选模型，按序检测 |
| `LLM_BASE_URL` | `dashscope.aliyuncs.com/...` | API 地址 |
| `POSTGRES_HOST` | `localhost` | PostgreSQL 主机 |
| `POSTGRES_PORT` | `5434` | PostgreSQL 端口 |
| `REDIS_HOST` | `localhost` | Redis 主机 |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `APP_PORT` | `8002` | FastAPI 端口 |

## 项目结构

```
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── api/                     # 接口层
│   │   ├── upload.py            # PDF 上传/列表/删除
│   │   ├── analyze.py           # 分析触发/状态查询/报告获取
│   │   └── chat.py              # 对话问答/历史
│   ├── agents/                  # Agent 实现
│   │   ├── planner.py           # 阅读规划师
│   │   ├── background_reader.py # 背景精读
│   │   ├── method_reader.py     # 方法精读
│   │   ├── experiment_reader.py # 实验精读
│   │   ├── critic.py            # 批判性审阅
│   │   ├── summarizer.py        # 报告整合
│   │   ├── router.py            # 对话路由
│   │   └── chat_handler.py      # 问答处理
│   ├── graph/
│   │   └── workflow.py          # LangGraph 工作流（分析 + 对话）
│   ├── core/
│   │   ├── config.py            # pydantic-settings 配置
│   │   ├── database.py          # SQLAlchemy 引擎
│   │   ├── logger.py            # loguru 日志
│   │   ├── retry_utils.py       # LLM 重试机制（tenacity）
│   │   └── state.py             # LangGraph 状态定义
│   ├── services/
│   │   ├── pdf_parser.py        # PDF 结构化解析（PyMuPDF）
│   │   ├── vector_store.py      # 向量分块/存储/检索（pgvector）
│   │   ├── section_utils.py     # 章节遍历工具
│   │   ├── converters.py        # 数据结构转换
│   │   └── storage.py           # 文件存储
│   ├── schemas/
│   │   └── models.py            # ORM + Pydantic 模型
│   └── tasks/
│       ├── celery_app.py        # Celery 配置
│       └── analysis_tasks.py    # 分析任务流水线
├── frontend/
│   └── app.py                   # Streamlit 前端
├── tests/                       # 33 个测试用例
├── migrations/                  # Alembic 数据库迁移
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/upload` | 上传 PDF（multipart/form-data） |
| `GET` | `/api/papers` | 论文列表 |
| `DELETE` | `/api/papers/{id}` | 删除论文及关联数据 |
| `POST` | `/api/analyze/{id}` | 触发异步分析，返回 task_id |
| `GET` | `/api/task/{task_id}` | 查询分析进度 |
| `GET` | `/api/report/{id}` | 获取完整分析报告 |
| `POST` | `/api/chat/{id}` | 论文问答（自动路由专家） |
| `GET` | `/api/chat/{id}/history` | 对话历史 |

## 技术栈

- **框架**: FastAPI + LangGraph + Celery
- **LLM**: 通义千问，OpenAI 兼容接口，模型自动检测
- **Embedding**: DashScope text-embedding-v2 / 本地 HuggingFace 模型
- **PDF 解析**: PyMuPDF (fitz)，字体分布分析 + 章节树构建
- **数据库**: PostgreSQL + pgvector（向量），SQLAlchemy ORM
- **缓存/队列**: Redis（Celery Broker + Result Backend）
- **前端**: Streamlit
- **可观测性**: loguru 结构化日志 + 文件归档
- **部署**: Docker Compose 五服务编排

