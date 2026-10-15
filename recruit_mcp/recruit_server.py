"""招聘业务 MCP Server（真实后端能力版）。

把 recruit_graph 里的真实招聘能力按 MCP 标准暴露成工具，任何 MCP 客户端
（Claude Desktop / 本项目的 Agent / 命令行）一次握手即可发现并调用：

- 岗位库（只读）：
  list_positions / get_position / search_jd_knowledge
- 简历分析（只读，不写库）：
  analyze_resume / match_resume_with_jd
- 业务闭环（写 SQLite）：
  screen_candidate / rank_candidates / delete_candidate

与早期玩具版的区别：工具不再自己写一套简化的 prompt，而是直接复用项目真实的
SQLite 存储层、Chroma 向量库与“BM25 + 向量 + RRF + rerank”混合检索，
结果和 Gradio 页面、seed 脚本完全一致。

传输方式：默认 stdio（本地 MCP 客户端各起一个进程、即插即用）；
跨网络的 Streamable HTTP 部署留到“部署上线”阶段开启（工具本身与传输无关）。
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 本文件在 recruit_mcp/ 下，上一级即仓库根目录
REPO_ROOT = Path(__file__).resolve().parent.parent

# 必须在 import recruit_graph 之前把根目录 .env 载入环境变量：
# 向量库 / 大模型相关模块在 import 时就会读取 API_KEY、BASE_URL。
load_dotenv(REPO_ROOT / ".env")
# 让独立进程也能 import recruit_graph 包
sys.path.insert(0, str(REPO_ROOT))

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

server = FastMCP("recruit_server")


# ===================== 岗位库（只读） =====================
@server.tool()
def list_positions() -> list[dict]:
    """列出岗位库中所有在招岗位（编号、岗位名称、录入时间），最新在前。
    当用户想知道“现在有哪些岗位 / 岗位列表”时使用，不返回 JD 全文。
    """
    try:
        return list_jds()
    except Exception as e:  # 兜底：读库失败也返回结构化错误，不让 MCP 调用崩
        return [{"error": "读取岗位列表失败", "detail": str(e)}]


@server.tool()
def get_position(jd_id: int) -> dict:
    """按岗位编号查看一份岗位的完整 JD 原文。
    当用户已经指定岗位编号、或要看某个岗位的详细要求时使用。
    :param jd_id: 岗位编号，来自 list_positions 返回的 id
    """
    try:
        jd = get_jd(jd_id)
    except Exception as e:
        return {"error": f"读取岗位 {jd_id} 失败", "detail": str(e)}
    if not jd:
        return {"error": f"未找到编号为 {jd_id} 的岗位"}
    return jd


@server.tool()
def search_jd_knowledge(
    query: str, top_k: int = 3, use_rerank: bool = True
) -> list[dict]:
    """用自然语言在岗位 JD 知识库里做混合检索（关键词 BM25 + 向量），
    返回最相关的若干段 JD 原文片段（含片段编号）。
    当用户问某项技术/要求在哪些岗位出现、或只给模糊需求找岗位时使用。
    :param query: 自然语言问题，如“哪个岗位要求会 RAG 和 LangChain”
    :param top_k: 返回片段数量，默认 3
    :param use_rerank: True=粗排后再用 bge-reranker 精排（默认，最准、稍慢）；
        False=只做 BM25+向量粗排（更快、不依赖精排服务，精排不可用时可关掉）
    """
    try:
        docs = hybrid_search(query, top_k=top_k, use_rerank=use_rerank)
        return [
            {"id": doc.metadata.get("id"), "text": doc.page_content} for doc in docs
        ]
    except Exception as e:  # BM25/向量/rerank 任一步网络失败，都降级为错误片段
        return [{"error": "知识库检索失败", "detail": str(e)}]


# ===================== 简历分析（只读，不写库） =====================
@server.tool()
def analyze_resume(resume_text: str) -> dict:
    """只解析一份简历、抽取结构化候选人画像（姓名、学历、技能、项目、
    目标岗位、期望薪资、电话），不会写数据库。
    当用户只想“分析这份简历 / 看看画像抽得准不准”，或想先人工确认再入库时使用。
    :param resume_text: 简历原文
    """
    try:
        return analyze_profile(resume_text)
    except Exception as e:
        return {"error": "简历分析失败", "detail": str(e)}


@server.tool()
def match_resume_with_jd(resume_text: str, jd_text: str) -> dict:
    """把一份简历与“用户直接给的一段 JD 原文”做临时对照，
    返回画像 + 匹配岗位、分数、理由；只读，不查岗位库、不写数据库。
    当用户手里拿着具体 JD、问“这个人配不配这个岗”时使用。
    :param resume_text: 简历原文
    :param jd_text: 岗位 JD 原文，由用户直接提供，不必在岗位库中
    """
    # 第一步：简历分析。它是匹配的刚需，失败就短路，绝不能把错误当画像喂给下一步。
    try:
        profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
    except Exception as e:
        return {"error": "简历分析失败，无法继续匹配", "detail": str(e)}
    # 第二步：拿着第一步的画像去匹配指定 JD。
    try:
        match_json = match_resume_by_jd.invoke(
            {"profile_json": profile_json, "jd_text": jd_text}
        )
        return {
            "profile": json.loads(profile_json),
            "match": json.loads(match_json),
        }
    except Exception as e:
        return {"error": "人岗匹配失败", "detail": str(e)}


# ===================== 业务闭环（写 SQLite） =====================
@server.tool()
def screen_candidate(resume_text: str) -> dict:
    """对一份简历做完整初筛：抽取画像 → 与岗位库全部 JD 横向匹配打分 →
    把候选人与匹配结果写入候选人库，返回画像和匹配结果。
    当用户要“把这个人收进系统初筛 / 入库并匹配岗位”时使用，会写数据库。
    :param resume_text: 简历原文
    """
    try:
        return save_candidate(resume_text)
    except Exception as e:
        return {"error": "候选人初筛入库失败", "detail": str(e)}


@server.tool()
def rank_candidates(min_score: int = 0) -> list[dict]:
    """查看候选人匹配排名（按匹配分从高到低），可只看某个分数线以上的人。
    当用户问“现在库里候选人怎么排 / 哪些人最匹配 / 给我排名”时使用。
    :param min_score: 最低匹配分，默认 0 表示返回全部候选人
    """
    try:
        return list_candidates(min_score=min_score)
    except Exception as e:
        return [{"error": "读取候选人排名失败", "detail": str(e)}]


@server.tool()
def delete_candidate(candidate_id: int) -> dict:
    """按候选人编号删除一名候选人，会写数据库且不可恢复。
    当用户明确要求删除某个候选人时使用。
    :param candidate_id: 候选人编号，来自 rank_candidates 返回的 id
    """
    try:
        deleted = delete_candidate_row(candidate_id)
        return {"candidate_id": candidate_id, "deleted": deleted}
    except Exception as e:
        return {"error": f"删除候选人 {candidate_id} 失败", "detail": str(e)}


if __name__ == "__main__":
    # 默认 stdio：本地 MCP 客户端（Claude Desktop / Agent）拉起一个进程即可调用。
    # 网络多客户端共享用 streamable HTTP，留到“部署上线”阶段开启。
    server.run()
