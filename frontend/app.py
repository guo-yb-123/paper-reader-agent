"""
Streamlit 前端 - 多Agent科研论文阅读系统
- 侧边栏：论文列表、上传入口、API 状态
- 上传页：拖拽上传 PDF
- 报告页：分章节折叠展示精读报告
- 对话页：聊天界面，针对论文问答
"""
import os
import time
import requests
import streamlit as st

# ============================================================
# 配置
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8002")

st.set_page_config(
    page_title="论文精读系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 样式
# ============================================================

st.markdown("""
<style>
    .report-section {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
    }
    .stExpander {
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }
    .chat-user { color: #1a73e8; font-weight: bold; }
    .chat-assistant { color: #4CAF50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 会话状态初始化
# ============================================================

if "papers" not in st.session_state:
    st.session_state.papers = []
if "selected_paper_id" not in st.session_state:
    st.session_state.selected_paper_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_page" not in st.session_state:
    st.session_state.current_page = "上传论文"

# ============================================================
# API 工具函数
# ============================================================

def api_call(method: str, path: str, **kwargs) -> dict | None:
    """统一的 API 调用封装"""
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
        if resp.status_code >= 400:
            st.error(f"API 错误 ({resp.status_code}): {resp.json().get('detail', resp.text)[:300]}")
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接后端服务 ({API_BASE_URL})，请确保服务已启动")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


def refresh_papers():
    """刷新论文列表"""
    data = api_call("GET", "/api/papers")
    if data:
        st.session_state.papers = data.get("papers", [])


def upload_pdf(file) -> str | None:
    """上传 PDF 文件"""
    resp = api_call("POST", "/api/upload", files={"file": file})
    if resp:
        return resp.get("paper_id")
    return None


def trigger_analysis(paper_id: str) -> str | None:
    """触发论文分析"""
    resp = api_call("POST", f"/api/analyze/{paper_id}")
    if resp:
        return resp.get("task_id")
    return None


def get_task_status(task_id: str) -> dict | None:
    """查询任务状态"""
    return api_call("GET", f"/api/task/{task_id}")


def get_report(paper_id: str) -> dict | None:
    """获取分析报告"""
    return api_call("GET", f"/api/report/{paper_id}")


def send_message(paper_id: str, question: str, history: list) -> dict | None:
    """发送对话消息"""
    return api_call(
        "POST", f"/api/chat/{paper_id}",
        json={"paper_id": paper_id, "question": question, "history": history},
        timeout=120,
    )


def get_chat_history(paper_id: str) -> list:
    """获取对话历史"""
    data = api_call("GET", f"/api/chat/{paper_id}/history")
    if data:
        return data.get("history", [])
    return []


# ============================================================
# 侧边栏
# ============================================================

def render_sidebar():
    """渲染侧边栏：论文列表 + 导航"""
    with st.sidebar:
        st.title("📄 论文精读系统")

        # API 状态
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if resp.status_code == 200:
                st.success("🟢 后端服务正常")
            else:
                st.warning("🟡 后端服务异常")
        except Exception:
            st.error("🔴 后端服务未连接")

        st.divider()

        # 导航
        pages = ["上传论文", "分析报告", "对话问答"]
        if st.session_state.current_page not in pages:
            st.session_state.current_page = "上传论文"

        page = st.radio("导航", pages, key="nav_radio")
        st.session_state.current_page = page

        st.divider()

        # 论文列表
        st.subheader("📚 论文列表")
        if st.button("🔄 刷新列表", use_container_width=True):
            refresh_papers()
            st.rerun()

        if not st.session_state.papers:
            refresh_papers()

        for paper in st.session_state.papers:
            pid = paper.get("paper_id", "")
            title = paper.get("title", "未命名")[:40]
            status = paper.get("status", "")

            status_icon = {
                "uploaded": "📤", "parsing": "⏳", "parsed": "📋",
                "analyzing": "🔄", "completed": "✅", "failed": "❌",
            }.get(status, "❓")

            col1, col2 = st.columns([4, 1])
            with col1:
                label = f"{status_icon} {title}"
                if st.button(label, key=f"paper_{pid}", use_container_width=True,
                             help=f"状态: {status}"):
                    st.session_state.selected_paper_id = pid
                    st.session_state.chat_history[pid] = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{pid}", help="删除论文"):
                    api_call("DELETE", f"/api/papers/{pid}")
                    refresh_papers()
                    if st.session_state.selected_paper_id == pid:
                        st.session_state.selected_paper_id = None
                    st.rerun()

        st.divider()
        st.caption(f"当前选中: {st.session_state.selected_paper_id or '无'}")


# ============================================================
# 上传页
# ============================================================

def render_upload_page():
    """渲染 PDF 上传页面"""
    st.header("📤 上传论文 PDF")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "拖拽或点击上传 PDF 论文",
            type=["pdf"],
            help="支持英文和中文论文 PDF，文件大小限制 100MB",
        )

        if uploaded_file is not None:
            st.info(f"已选择: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("🚀 上传并开始分析", type="primary", use_container_width=True):
                with st.status("正在处理...", expanded=True) as status:
                    # 1. 上传
                    st.write("📤 上传 PDF...")
                    paper_id = upload_pdf(uploaded_file)
                    if not paper_id:
                        status.update(label="上传失败", state="error")
                        return
                    st.write(f"✅ 上传成功: `{paper_id}`")

                    # 2. 触发分析
                    st.write("🔬 触发分析任务...")
                    task_id = trigger_analysis(paper_id)
                    if not task_id:
                        status.update(label="分析触发失败", state="error")
                        return
                    st.write(f"📋 任务 ID: `{task_id}`")

                    # 3. 轮询进度
                    st.write("⏳ 等待分析完成...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for _ in range(120):  # 最多等待 10 分钟
                        task = get_task_status(task_id)
                        if not task:
                            break

                        tstatus = task.get("status", "")
                        progress = task.get("progress", 0)
                        message = task.get("message", "")

                        progress_bar.progress(progress / 100)
                        status_text.text(f"{message} ({progress}%)")

                        if tstatus == "SUCCESS":
                            status.update(label="分析完成！", state="complete")
                            refresh_papers()
                            st.session_state.selected_paper_id = paper_id
                            st.session_state.current_page = "分析报告"
                            time.sleep(0.5)
                            st.rerun()
                        elif tstatus == "FAILURE":
                            error = task.get("error", "未知错误")
                            status.update(label=f"分析失败: {error}", state="error")
                            return

                        time.sleep(3)

                    status.update(label="分析超时，请稍后查看报告", state="warning")

    with col2:
        st.subheader("📋 使用说明")
        st.markdown("""
        1. **上传 PDF**：拖拽或点击上传
        2. **自动分析**：系统自动解析论文并启动多 Agent 分析
        3. **查看报告**：分析完成后，在「分析报告」页查看
        4. **对话问答**：在「对话问答」页与论文对话

        ⏱️ 分析通常需要 3-8 分钟
        """)


# ============================================================
# 报告页
# ============================================================

def render_report_page():
    """渲染分析报告页面"""
    st.header("📊 分析报告")

    if not st.session_state.selected_paper_id:
        st.info("👈 请先在左侧边栏选择一篇论文")
        return

    paper_id = st.session_state.selected_paper_id
    report = get_report(paper_id)

    if not report:
        st.error("无法获取报告，请检查后端服务")
        return

    status = report.get("status", "")
    title = report.get("title", "未命名")

    st.subheader(f"📄 {title}")

    if status != "completed":
        st.warning(f"当前状态: **{status}** — 分析尚未完成，请等待或重新触发分析")

        if st.button("🔄 重新触发分析"):
            task_id = trigger_analysis(paper_id)
            if task_id:
                st.success(f"分析任务已提交: {task_id}")
                # 自动跳转轮询
                with st.spinner("等待分析完成..."):
                    for _ in range(120):
                        task = get_task_status(task_id)
                        if task and task.get("status") == "SUCCESS":
                            st.rerun()
                        time.sleep(3)
        return

    # 报告内容
    final_report = report.get("final_report", "")

    if final_report:
        # 展示完整的 Markdown 报告
        with st.expander("📝 完整精读报告", expanded=True):
            st.markdown(final_report)

    # 各分段详情（折叠）
    col1, col2 = st.columns(2)

    with col1:
        reading_plan = report.get("reading_plan", {})
        if reading_plan:
            with st.expander("📋 阅读计划"):
                st.json(reading_plan)

        method_summary = report.get("method_summary", "")
        if method_summary:
            with st.expander("🔧 方法分析 (Method Reader)"):
                st.markdown(method_summary)

        background_summary = report.get("background_summary", "")
        if background_summary:
            with st.expander("🎯 背景分析 (Background Reader)"):
                st.markdown(background_summary)

    with col2:
        experiment_summary = report.get("experiment_summary", "")
        if experiment_summary:
            with st.expander("📊 实验分析 (Experiment Reader)"):
                st.markdown(experiment_summary)

        critique = report.get("critique", "")
        if critique:
            with st.expander("⚠️ 批判性审阅 (Critic)"):
                st.markdown(critique)

    # 导出按钮
    if final_report:
        st.download_button(
            label="📥 下载完整报告 (Markdown)",
            data=final_report,
            file_name=f"{title[:30]}_精读报告.md",
            mime="text/markdown",
        )


# ============================================================
# 对话页
# ============================================================

def render_chat_page():
    """渲染对话问答页面"""
    st.header("💬 对话问答")

    if not st.session_state.selected_paper_id:
        st.info("👈 请先在左侧边栏选择一篇论文")
        return

    paper_id = st.session_state.selected_paper_id

    # 检查报告是否完成
    report = get_report(paper_id)
    if not report or report.get("status") != "completed":
        st.warning("论文分析尚未完成，部分回答可能不够精准")

    # 初始化历史
    if paper_id not in st.session_state.chat_history:
        st.session_state.chat_history[paper_id] = []

    # 显示对话历史
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history[paper_id]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            router = msg.get("router_decision", "")

            if role == "user":
                st.markdown(f"<div class='chat-user'>🙋 你</div> {content}", unsafe_allow_html=True)
            else:
                expert_label = f" (via {router})" if router else ""
                st.markdown(f"<div class='chat-assistant'>🤖 专家{expert_label}</div>", unsafe_allow_html=True)
                st.markdown(content)
            st.divider()

    # 输入框
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "输入你的问题",
                placeholder="例如：这篇论文的核心创新点是什么？实验设计有什么问题？",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("发送 🚀", use_container_width=True)

    if submitted and question.strip():
        # 添加到历史（用户）
        st.session_state.chat_history[paper_id].append({
            "role": "user", "content": question.strip(),
        })

        # 调用 API
        history_for_api = [
            {"role": h["role"], "content": h["content"]}
            for h in st.session_state.chat_history[paper_id][:-1]  # 不含当前问题
        ]

        with st.spinner("专家思考中..."):
            resp = send_message(paper_id, question.strip(), history_for_api)
            if resp:
                st.session_state.chat_history[paper_id].append({
                    "role": "assistant",
                    "content": resp.get("answer", "抱歉，回答失败"),
                    "router_decision": resp.get("router_decision", ""),
                })
            else:
                st.session_state.chat_history[paper_id].append({
                    "role": "assistant",
                    "content": "抱歉，回答失败，请检查后端服务。",
                    "router_decision": "",
                })

        st.rerun()

    # 建议问题
    with st.expander("💡 试试这些问题"):
        suggestions = [
            "这篇论文要解决什么问题？",
            "核心方法是什么？有什么创新？",
            "实验用了什么数据集？效果怎么样？",
            "这篇论文有什么局限性？",
            "给我总结一下这篇论文",
        ]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history[paper_id].append({
                        "role": "user", "content": sug,
                    })
                    with st.spinner("专家思考中..."):
                        resp = send_message(paper_id, sug, [
                            {"role": h["role"], "content": h["content"]}
                            for h in st.session_state.chat_history[paper_id][:-1]
                        ])
                        if resp:
                            st.session_state.chat_history[paper_id].append({
                                "role": "assistant",
                                "content": resp.get("answer", "抱歉"),
                                "router_decision": resp.get("router_decision", ""),
                            })
                    st.rerun()

    # 清空对话
    if st.session_state.chat_history.get(paper_id):
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_history[paper_id] = []
            st.rerun()


# ============================================================
# 主入口
# ============================================================

def main():
    render_sidebar()

    page = st.session_state.current_page
    if page == "上传论文":
        render_upload_page()
    elif page == "分析报告":
        render_report_page()
    elif page == "对话问答":
        render_chat_page()
    else:
        render_upload_page()


if __name__ == "__main__":
    main()
