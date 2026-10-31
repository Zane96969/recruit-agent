"""阶段三：边界输入防护测试。

检索与入库入口会收到调用方传来的"不合理但不算错"的参数与文本：
top_k / candidate_n 传 0 或负数、整段无句号的超长文本、超长 JD 原文。
这些输入不该让系统行为诡异（悄悄返回空、负数切掉尾部、生成超出
embedding token 上限的巨型块），而应被钳到安全下限或直接拒绝。

全程离线：假快照 + 假向量层，不碰真实 Chroma / API_KEY。
"""

import math

import pytest

from recruit_graph import db, hybrid_retriever, jd_store, jd_vector_store


# ---------- 1. 检索参数钳制 ----------

class FakeEmbeddings:
    """假 embedding：永远正常返回一个固定向量。"""

    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """假 Chroma collection：query 永远按固定顺序返回全部块 id。"""

    def __init__(self, ids):
        self.ids = ids

    def query(self, query_embeddings, n_results):
        return {"ids": [list(self.ids)]}


@pytest.fixture()
def retrieval(monkeypatch):
    """装一个 3 块的假检索环境（快照 / BM25 / 向量路全部离线可控）。"""
    ids = ["jd1_0", "jd1_1", "jd1_2"]
    fake = {
        "key": ("test", 3),
        "ids": ids,
        "id_to_doc": {
            "jd1_0": "Python 开发工程师。",
            "jd1_1": "负责 RAG 检索系统。",
            "jd1_2": "要求熟悉 Java 后端。",
        },
        "meta": {cid: {"jd_id": 1, "jd_name": "Python 工程师"} for cid in ids},
        "bm25": None,
    }
    monkeypatch.setattr(hybrid_retriever, "_snapshot", lambda: fake)
    monkeypatch.setattr(
        hybrid_retriever,
        "bm25_search",
        lambda query, top_n: [(cid, fake["id_to_doc"][cid]) for cid in ids],
    )
    monkeypatch.setattr(hybrid_retriever, "embeddings", FakeEmbeddings())
    monkeypatch.setattr(
        hybrid_retriever,
        "vectorstore",
        type(
            "FakeVectorStore",
            (),
            {"_collection": FakeCollection(ids)},
        )(),
    )
    return fake


def test_candidate_n_zero_clamped_to_one(retrieval):
    """candidate_n=0：应按 1 处理拿到结果，而不是悄悄返回空列表。"""
    results = hybrid_retriever.hybrid_search(
        "Python", candidate_n=0, top_k=2, use_rerank=False
    )

    assert len(results) == 1


def test_negative_top_k_clamped_to_one(retrieval):
    """top_k=-1：负切片会静默切掉尾部，应钳成 1。"""
    results = hybrid_retriever.hybrid_search(
        "Python", candidate_n=3, top_k=-1, use_rerank=False
    )

    assert len(results) == 1


# ---------- 2. 无句号超长文本的切块兜底 ----------

def test_split_sentences_normal():
    """基线：正常按句号切、补回句号。"""
    assert jd_vector_store.split_sentences("第一句。第二句。") == [
        "第一句。",
        "第二句。",
    ]


def test_split_sentences_long_no_period_chunked():
    """无句号超长文本：按 MAX_CHUNK_CHARS 硬切，不生成超出
    embedding token 上限的巨型块。"""
    text = "长" * 700
    chunks = jd_vector_store.split_sentences(text)

    limit = jd_vector_store.MAX_CHUNK_CHARS
    assert len(chunks) == math.ceil(len(text) / limit)
    assert all(len(c) <= limit for c in chunks)


# ---------- 3. JD 原文长度上限 ----------

class FakeVectorOps:
    """假向量层：正常成功，供入库路径走通。"""

    def upsert_jd(self, jd_id, jd_name, jd_text):
        pass

    def delete_jd_vectors(self, jd_id):
        pass

    def rename_jd_vectors(self, jd_id, jd_name):
        pass


@pytest.fixture()
def jds(tmp_path, monkeypatch):
    """临时数据库 + 假向量层，不碰真实 recruit_real.db / chroma_db。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_jd.db")
    fake = FakeVectorOps()
    monkeypatch.setattr(jd_store, "upsert_jd", fake.upsert_jd)
    monkeypatch.setattr(jd_store, "delete_jd_vectors", fake.delete_jd_vectors)
    monkeypatch.setattr(jd_store, "rename_jd_vectors", fake.rename_jd_vectors)
    return fake


def _rows() -> list[dict]:
    conn = db.connect()
    try:
        conn.execute(jd_store._CREATE_JD_TABLE)
        return [dict(r) for r in conn.execute("SELECT * FROM jds").fetchall()]
    finally:
        conn.close()


def test_save_jd_rejects_oversized_text(jds):
    """JD 原文超过上限：直接拒绝，不落库（防巨型切块与巨额 embedding 账单）。"""
    with pytest.raises(ValueError):
        jd_store.save_jd("超长岗位", "字" * (jd_store.MAX_JD_CHARS + 1))

    assert _rows() == []


def test_update_jd_rejects_oversized_text(jds):
    """更新时传入超长文本：拒绝，原内容保持不变。"""
    saved = jd_store.save_jd("岗位", "正常文本。")

    with pytest.raises(ValueError):
        jd_store.update_jd(saved["jd_id"], "岗位", "字" * (jd_store.MAX_JD_CHARS + 1))

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["jd_text"] == "正常文本。"
