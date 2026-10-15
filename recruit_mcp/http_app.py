"""招聘 Agent 的 HTTP 服务入口（FastAPI + MCP Streamable HTTP）。

把同一套招聘后端（SQLite / Chroma / 混合检索 / 大模型）在一个进程里同时暴露成：

- 普通 REST 接口（/api/...）：给浏览器、前端、Postman 等常规 HTTP 客户端调用；
- 标准 MCP 接口（/mcp）：走 2025 年版 Streamable HTTP 传输，
  任何 MCP 客户端（本项目 Agent / Claude Desktop / Cursor）连一个 URL 即可握手调用，
  一个服务可同时被多个网络客户端共享，不再是 stdio 那种“一个客户端拉起一个进程”。

启动（在仓库根目录）：
    python recruit_mcp/http_app.py
    # 或：uvicorn recruit_mcp.http_app:app --host 127.0.0.1 --port 8000

启动后：
    - 健康检查：GET http://127.0.0.1:8000/health
    - 接口文档：GET http://127.0.0.1:8000/docs
    - MCP 端点：http://127.0.0.1:8000/mcp
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# 本文件在 recruit_mcp/ 下，上一级即仓库根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
# 让两种启动方式（直接 python、uvicorn 模块）都能找到本目录与 recruit_graph 包
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 必须在 import recruit_server / recruit_graph 之前载入根目录 .env
load_dotenv(REPO_ROOT / ".env")

# 复用 recruit_server 里已经注册好 8 个工具的 FastMCP 实例
from recruit_server import server  # noqa: E402
from recruit_graph.candidate_store import (  # noqa: E402
    analyze_profile,
    delete_candidate as delete_candidate_row,
    list_candidates,
    save_candidate,
)
from recruit_graph.graph_tools import (  # noqa: E402
    analyze_resume_worker,
    match_resume_by_jd,
)
from recruit_graph.hybrid_retriever import hybrid_search  # noqa: E402
from recruit_graph.jd_store import get_jd, list_jds  # noqa: E402


# ===================== 请求体模型 =====================
class ResumeIn(BaseModel):
    """简历分析 / 初筛的请求体：resume_text 是简历原文。"""

    resume_text: str


class MatchIn(BaseModel):
    """人岗临时匹配的请求体：简历原文 + 一段 JD 原文。"""

    resume_text: str
    jd_text: str


# ===================== MCP 子应用（/mcp） =====================
# 子应用内部路由默认是 /mcp；这里它要被挂到外层 /mcp 前缀下，
# 挂载时 Starlette 会剥掉前缀，所以把“内部端点”改成根路径 "/"，
# 两段拼起来对外仍是标准的 http://host:8000/mcp。
server.settings.streamable_http_path = "/"
mcp_app = server.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """宿主应用的生命周期。

    关键坑：MCP 子应用一旦被 mount，它自带的 lifespan 就不再运行，
    而 Streamable HTTP 的会话管理器（session_manager）必须由 lifespan 启动，
    否则第一个请求会报 “Task group is not initialized”。
    所以这里由宿主 FastAPI 显式进入 session_manager.run()。
    """
    async with server.session_manager.run():
        yield


# ===================== FastAPI 主应用 =====================
app = FastAPI(
    title="recruit-agent 招聘智能体服务",
    description="同一套招聘后端，同时提供 REST 接口（/api）与 MCP 接口（/mcp）。",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 健康检查 ----------
@app.get("/health")
def health():
    """探活接口，返回服务状态与 MCP 端点。"""
    return {"status": "ok", "service": "recruit-agent", "mcp_endpoint": "/mcp"}


# ---------- 岗位库（只读） ----------
@app.get("/api/positions")
def api_list_positions():
    """列出全部在招岗位（编号、岗位名、录入时间），最新在前。"""
    return list_jds()


@app.get("/api/positions/{jd_id}")
def api_get_position(jd_id: int):
    """按编号查看一份岗位的完整 JD。"""
    jd = get_jd(jd_id)
    if not jd:
        raise HTTPException(status_code=404, detail=f"未找到编号为 {jd_id} 的岗位")
    return jd


@app.get("/api/search")
def api_search(
    query: str = Query(..., description="自然语言问题，如：哪个岗位要求会 RAG"),
    top_k: int = Query(3, ge=1, le=10, description="返回片段数量"),
    use_rerank: bool = Query(
        True, description="true=粗排+精排（默认，最准）；false=只粗排（更快）"
    ),
):
    """在岗位 JD 知识库做混合检索（BM25 + 向量 + RRF，可选 bge 精排）。"""
    docs = hybrid_search(query, top_k=top_k, use_rerank=use_rerank)
    return [{"id": doc.metadata.get("id"), "text": doc.page_content} for doc in docs]


# ---------- 简历分析（只读，不写库） ----------
@app.post("/api/analyze-resume")
def api_analyze_resume(body: ResumeIn):
    """只解析简历、抽取结构化画像，不写数据库。"""
    return analyze_profile(body.resume_text)


@app.post("/api/match-resume")
def api_match_resume(body: MatchIn):
    """把简历与一段给定 JD 临时对照，返回画像 + 匹配结果，不写数据库。"""
    profile_json = analyze_resume_worker.invoke({"resume_text": body.resume_text})
    match_json = match_resume_by_jd.invoke(
        {"profile_json": profile_json, "jd_text": body.jd_text}
    )
    return {"profile": json.loads(profile_json), "match": json.loads(match_json)}


# ---------- 业务闭环（写 SQLite） ----------
@app.post("/api/screen-candidate")
def api_screen_candidate(body: ResumeIn):
    """完整初筛：抽取画像 → 全量 JD 匹配打分 → 写入候选人库。"""
    return save_candidate(body.resume_text)


@app.get("/api/candidates")
def api_rank_candidates(min_score: int = Query(0, ge=0, description="最低匹配分")):
    """查看候选人匹配排名，按匹配分从高到低。"""
    return list_candidates(min_score=min_score)


@app.delete("/api/candidates/{candidate_id}")
def api_delete_candidate(candidate_id: int):
    """按编号删除一名候选人。"""
    deleted = delete_candidate_row(candidate_id)
    return {"candidate_id": candidate_id, "deleted": deleted}


# 最后把 MCP 子应用挂到 /mcp（精确路径，不影响上面的 /api 与 /docs）
app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    import uvicorn

    # 默认只监听本机回环地址；部署到服务器时再改成 0.0.0.0 并配置访问白名单
    uvicorn.run(app, host="127.0.0.1", port=8000)
