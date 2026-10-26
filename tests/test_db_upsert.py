"""候选人写入的回归测试：重复入库必须保持 id 不变。

这条不变式为什么值得单独测：`matchings.candidate_id` 引用候选人 id，
一旦写入退化成 `INSERT OR REPLACE`（删旧行 + 插新行），id 会变，
历史评分就全部悬空 —— 而且不会报错，属于最难发现的一类数据损坏。

本文件只依赖标准库与 `recruit_graph.db`，不需要 API_KEY / 向量库，可离线跑：
    pip install -e ".[dev]"
    pytest
"""

import pytest

from recruit_graph import db


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """把数据库指到临时文件，测试不碰真实的 recruit_real.db。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    return db


def candidate_row(**overrides) -> dict:
    row = {
        "name": "张三",
        "age": "23",
        "education": "本科",
        "skills": "Python,RAG",
        "projects": "招聘 Agent",
        "target_position": "AI Agent 实习",
        "expected_salary": "3K-5K",
        "phone": "13800000000",
        "content_hash": "hash_a",
        "matched_job": "AI Agent 开发",
        "match_score": 72,
    }
    row.update(overrides)
    return row


def test_insert_then_conflict_keeps_same_id(store):
    """同一份简历写两次：第二次是更新，id 不能变。"""
    conn = store.connect()
    try:
        with conn:
            first_id, is_update = store.upsert_candidate(conn, candidate_row())
        with conn:
            second_id, second_is_update = store.upsert_candidate(
                conn, candidate_row(match_score=88)
            )
    finally:
        conn.close()

    assert is_update is False
    assert second_id == first_id
    assert second_is_update is True


def test_conflict_by_phone_alone_keeps_same_id(store):
    """简历改写后 content_hash 变了，但手机号相同 → 仍认作同一个人。"""
    conn = store.connect()
    try:
        with conn:
            first_id, _ = store.upsert_candidate(conn, candidate_row())
        with conn:
            second_id, is_update = store.upsert_candidate(
                conn, candidate_row(content_hash="hash_b", match_score=91)
            )
    finally:
        conn.close()

    assert second_id == first_id
    assert is_update is True


def test_only_one_row_survives_repeat(store):
    """重复入库不能变成两条记录。"""
    conn = store.connect()
    try:
        with conn:
            store.upsert_candidate(conn, candidate_row())
            store.upsert_candidate(conn, candidate_row(match_score=88))
        rows = conn.execute("SELECT id, match_score FROM candidates").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["match_score"] == 88


def test_unknown_phone_does_not_merge_people(store):
    """手机号识别不到（NULL）时，两份不同简历不能被误判成同一个人。"""
    conn = store.connect()
    try:
        with conn:
            a_id, _ = store.upsert_candidate(
                conn, candidate_row(phone=None, content_hash="h1")
            )
            b_id, _ = store.upsert_candidate(
                conn, candidate_row(phone=None, content_hash="h2")
            )
    finally:
        conn.close()

    assert a_id != b_id


def test_failed_transaction_writes_nothing(store):
    """写库链路后半段报错时，整行不回滚就不能留脏数据 —— 事务必须整体回滚。"""
    conn = store.connect()
    try:
        with pytest.raises(RuntimeError):
            with conn:
                store.upsert_candidate(conn, candidate_row())
                raise RuntimeError("模拟后续步骤失败")
        rows = conn.execute("SELECT id FROM candidates").fetchall()
    finally:
        conn.close()

    assert rows == []
