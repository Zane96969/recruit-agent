"""MCP 工具层测试：注册清单 + 新增写工具的行为。

要验证的不变式：
1. recruit_server 对外注册的工具清单稳定（新增不丢旧的）；
2. save_position / delete_position 把底层异常（超长 ValueError、
   向量库故障 RuntimeError）转成结构化 error 返回——外部 Agent 拿到的
   是"可读的失败信息"，而不是一次中断的 MCP 调用。

全程离线：monkeypatch 掉 jd_store 的写函数，不碰真实库。
"""

import asyncio

import pytest

import recruit_mcp.recruit_server as recruit_server
from recruit_mcp.recruit_server import server

EXPECTED_TOOLS = {
    "list_positions",
    "get_position",
    "search_jd_knowledge",
    "analyze_resume",
    "match_resume_with_jd",
    "screen_candidate",
    "rank_candidates",
    "delete_candidate",
    "save_position",
    "delete_position",
}


# ---------- 1. 注册清单 ----------

def test_registry_contains_all_expected_tools():
    """握手后外部 Agent 应能发现全部 10 个工具（新增不丢旧的）。"""
    tools = asyncio.run(server.list_tools())

    names = {tool.name for tool in tools}
    assert EXPECTED_TOOLS <= names


# ---------- 2. save_position：录岗（写双库） ----------

def test_save_position_ok(monkeypatch):
    """正常保存：透传 save_jd 的结果（含 jd_id）。"""
    monkeypatch.setattr(
        recruit_server,
        "save_jd",
        lambda name, text: {"jd_name": name, "jd_id": 1, "content_hash": "x"},
    )

    result = recruit_server.save_position("AI 工程师", "负责 RAG 系统。")

    assert result["jd_id"] == 1


def test_save_position_oversized_returns_structured_error(monkeypatch):
    """超长 JD（ValueError）：返回结构化 error，不把异常抛进 MCP 通道。"""

    def boom(name, text):
        raise ValueError("JD 原文过长（20001 字符，上限 20000），请精简后重试")

    monkeypatch.setattr(recruit_server, "save_jd", boom)

    result = recruit_server.save_position("超长岗", "字")

    assert "error" in result
    assert "请精简后重试" in result["detail"]


def test_save_position_vector_failure_returns_structured_error(monkeypatch):
    """向量库故障（RuntimeError）：同样转结构化 error，附可重试提示。"""

    def boom(name, text):
        raise RuntimeError("岗位「x」保存失败：向量库写入出错，数据库已回滚未落库")

    monkeypatch.setattr(recruit_server, "save_jd", boom)

    result = recruit_server.save_position("x", "正文")

    assert "error" in result
    assert "可重试" in result["error"]


# ---------- 3. delete_position：删岗（写双库） ----------

def test_delete_position_ok(monkeypatch):
    """正常删除：透传结果。"""
    monkeypatch.setattr(recruit_server, "delete_jd", lambda jd_id: True)

    result = recruit_server.delete_position(3)

    assert result == {"jd_id": 3, "deleted": True}


def test_delete_position_failure_returns_structured_error(monkeypatch):
    """向量清理失败（RuntimeError）：转结构化 error，说明岗位已保留。"""

    def boom(jd_id):
        raise RuntimeError("岗位 #3 删除失败：向量库清理出错，岗位已保留，可重试")

    monkeypatch.setattr(recruit_server, "delete_jd", boom)

    result = recruit_server.delete_position(3)

    assert "error" in result
    assert "岗位已保留" in result["detail"]
