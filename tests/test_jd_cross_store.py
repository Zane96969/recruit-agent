"""跨存储一致性测试：SQLite（业务库）与向量库必须同生共死。

jd_store 的 save_jd / update_jd / delete_jd 写入横跨两个存储：
先写 SQLite、再重建向量块。若向量写失败时 SQLite 已经 commit，
就会出现"列表里看得到、检索却搜不到"的脏数据，且没有任何提示。

要验证的不变式：向量库写入失败 → SQLite 必须回滚到操作前的状态，
并抛出可读的错误；失败后重试一次即可恢复。

全程离线：数据库指到临时文件，向量层三个函数用假对象替换
（沿用 test_retrieve_fallback.py 的 monkeypatch 故障注入模式）。
"""

import pytest

from recruit_graph import db, jd_store


class FakeVectorOps:
    """假向量层：记录调用顺序，可按开关抛异常。"""

    def __init__(self):
        self.calls = []
        self.fail_upsert = False
        self.fail_delete = False
        self.fail_rename = False

    def upsert_jd(self, jd_id, jd_name, jd_text):
        self.calls.append(("upsert", jd_id, jd_name, jd_text))
        if self.fail_upsert:
            raise RuntimeError("模拟向量库写入失败")

    def delete_jd_vectors(self, jd_id):
        self.calls.append(("delete", jd_id))
        if self.fail_delete:
            raise RuntimeError("模拟向量库删除失败")

    def rename_jd_vectors(self, jd_id, jd_name):
        self.calls.append(("rename", jd_id, jd_name))
        if self.fail_rename:
            raise RuntimeError("模拟向量库改名失败")


@pytest.fixture()
def vectors(tmp_path, monkeypatch):
    """临时数据库 + 假向量层，不碰真实 recruit_real.db / chroma_db。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_jd.db")
    fake = FakeVectorOps()
    monkeypatch.setattr(jd_store, "upsert_jd", fake.upsert_jd)
    monkeypatch.setattr(jd_store, "delete_jd_vectors", fake.delete_jd_vectors)
    monkeypatch.setattr(jd_store, "rename_jd_vectors", fake.rename_jd_vectors)
    return fake


def _fetch_jds() -> list[dict]:
    """读当前 jds 表全部行（表由被测函数自己建）。"""
    conn = db.connect()
    try:
        conn.execute(jd_store._CREATE_JD_TABLE)
        return [dict(r) for r in conn.execute("SELECT * FROM jds").fetchall()]
    finally:
        conn.close()


def test_save_jd_vector_failure_rolls_back_sqlite(vectors):
    """向量写失败：SQLite 不能留下脏行（否则看得到、搜不到）。"""
    vectors.fail_upsert = True

    with pytest.raises(RuntimeError):
        jd_store.save_jd("Python 工程师", "负责 RAG 系统开发。")

    assert _fetch_jds() == []


def test_save_jd_success_writes_both_stores(vectors):
    """正常路径：SQLite 与向量层都写（基线回归）。"""
    result = jd_store.save_jd("Python 工程师", "负责 RAG 系统开发。")

    rows = _fetch_jds()
    assert len(rows) == 1
    assert rows[0]["jd_name"] == "Python 工程师"
    assert rows[0]["id"] == result["jd_id"]
    assert ("delete", result["jd_id"]) in vectors.calls
    assert ("upsert", result["jd_id"], "Python 工程师", "负责 RAG 系统开发。") in (
        vectors.calls
    )


def test_save_retry_after_failure_heals(vectors):
    """失败不留脏数据 → 直接重试一次就能成功（自愈）。"""
    vectors.fail_upsert = True
    with pytest.raises(RuntimeError):
        jd_store.save_jd("Python 工程师", "负责 RAG 系统开发。")

    vectors.fail_upsert = False
    result = jd_store.save_jd("Python 工程师", "负责 RAG 系统开发。")

    assert len(_fetch_jds()) == 1
    assert ("upsert", result["jd_id"], "Python 工程师", "负责 RAG 系统开发。") in (
        vectors.calls
    )


def test_update_jd_vector_failure_keeps_old_text(vectors):
    """更新时向量重建失败：SQLite 应回滚到旧文本，并尝试用旧文重建向量块。"""
    saved = jd_store.save_jd("旧岗位", "旧文本。")
    jd_id = saved["jd_id"]

    vectors.fail_upsert = True
    with pytest.raises(RuntimeError):
        jd_store.update_jd(jd_id, "新岗位", "新文本。")

    rows = _fetch_jds()
    assert len(rows) == 1
    assert rows[0]["jd_text"] == "旧文本。"
    # 补偿动作：曾尝试用旧岗位的原文重建向量块（重建本身也失败，但必须尝试）
    assert ("upsert", jd_id, "旧岗位", "旧文本。") in vectors.calls


def test_update_jd_rename_failure_keeps_old_name(vectors):
    """只改名时向量改名失败：SQLite 应回滚，旧岗位名不动。"""
    saved = jd_store.save_jd("旧岗位名", "同一段文本。")
    jd_id = saved["jd_id"]

    vectors.fail_rename = True
    with pytest.raises(RuntimeError):
        jd_store.update_jd(jd_id, "新岗位名", "同一段文本。")

    rows = _fetch_jds()
    assert rows[0]["jd_name"] == "旧岗位名"


def test_delete_jd_vector_failure_keeps_row(vectors):
    """删除时向量清理失败：SQLite 应回滚，岗位仍在。"""
    saved = jd_store.save_jd("Python 工程师", "负责 RAG 系统开发。")
    jd_id = saved["jd_id"]

    vectors.fail_delete = True
    with pytest.raises(RuntimeError):
        jd_store.delete_jd(jd_id)

    assert len(_fetch_jds()) == 1
