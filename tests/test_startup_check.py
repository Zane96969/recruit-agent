"""阶段四（轻量版）：启动配置自检 + 界面错误兜底。

两件事：
1. 启动那一刻就发现 API_KEY 缺失，给出一句中文指引并退出，
   而不是等第一次 embedding/LLM 调用在深处抛 401；
2. Gradio 界面的保存/删除 handler 把底层异常转成界面上的
   一句可读消息，而不是把 traceback 甩给使用者。

全程离线，不碰真实 .env / chroma_db。
"""

import pytest

from recruit_graph.config_check import missing_keys

import web_jd_manager  # noqa: F401  # 模块级搭 Gradio 界面但不启动服务


# ---------- 1. 配置自检 ----------

def test_missing_keys_empty_when_set(monkeypatch):
    """API_KEY 在环境变量里：自检通过。"""
    monkeypatch.setenv("API_KEY", "sk-test")

    assert missing_keys() == []


def test_missing_keys_reports_api_key(monkeypatch):
    """API_KEY 缺失：返回缺失项清单，供启动器打印指引。"""
    monkeypatch.delenv("API_KEY", raising=False)

    assert missing_keys() == ["API_KEY"]


# ---------- 2. 界面 handler 兜底 ----------

@pytest.fixture()
def quiet_ui(monkeypatch):
    """让 handler 不碰真实数据库：下拉刷新返回空、底层函数可控替换。"""
    monkeypatch.setattr(web_jd_manager, "_refresh_choices", lambda: [])


def test_save_handler_shows_friendly_error(quiet_ui, monkeypatch):
    """底层保存抛系统错误（RuntimeError）：界面显示消息，不冒 traceback。"""

    def boom(name, text):
        raise RuntimeError("岗位「x」保存失败：向量库写入出错")

    monkeypatch.setattr(web_jd_manager, "save_jd", boom)

    _, msg = web_jd_manager.save_handler("x", "正文。")

    assert "未保存" in msg
    assert "向量库写入出错" in msg


def test_save_handler_rejects_oversized_text(quiet_ui, monkeypatch):
    """底层保存拒绝超长输入（ValueError）：界面同样转成可读消息。"""

    def boom(name, text):
        raise ValueError("JD 原文过长（20001 字符，上限 20000），请精简后重试")

    monkeypatch.setattr(web_jd_manager, "save_jd", boom)

    _, msg = web_jd_manager.save_handler("x", "正文。")

    assert "未保存" in msg
    assert "请精简后重试" in msg


def test_delete_handler_shows_friendly_error(quiet_ui, monkeypatch):
    """底层删除抛系统错误：岗位保留，界面显示消息。"""

    def boom(jd_id):
        raise RuntimeError("岗位 #3 删除失败：向量库清理出错，岗位已保留，可重试")

    monkeypatch.setattr(web_jd_manager, "delete_jd", boom)

    _, msg = web_jd_manager.delete_handler("x（#3）")

    assert "未删除" in msg
    assert "岗位已保留" in msg
