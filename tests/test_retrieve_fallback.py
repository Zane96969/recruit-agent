"""检索层单路故障隔离的故障注入测试。

要验证的不变式：BM25 关键词路和向量语义路是两条独立的路，
其中一条（最常见是远程 embedding 服务）抛异常时，另一条要能兜底，
两路都失败才返回空列表——绝不能因为一次外部服务故障让整个检索崩掉。

全程离线：不碰真实 API_KEY / Chroma，用假快照 + monkeypatch 注入故障。
"""

import pytest

from recruit_graph import hybrid_retriever


class FakeEmbeddings:
    """假的 embedding（向量化）客户端：可控地正常返回或抛异常。"""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_query(self, query):
        if self.fail:
            raise RuntimeError("模拟 embedding 服务不可用")
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """假的 Chroma collection（向量集合）：query 可控地正常返回或抛异常。"""

    def __init__(self, ids, fail: bool = False):
        self._ids = ids
        self.fail = fail

    def query(self, query_embeddings, n_results):
        if self.fail:
            raise RuntimeError("模拟向量库查询失败")
        return {"ids": [list(self._ids)]}


def _fake_vectorstore(ids, fail: bool):
    """拼一个带 _collection 的假 vectorstore（向量库对象）。"""
    return type(
        "FakeVectorStore",
        (),
        {"_collection": FakeCollection(ids, fail=fail)},
    )()


@pytest.fixture()
def snapshot(monkeypatch):
    """装一个固定的假快照，让 hybrid_retrieve 不依赖真实 Chroma。"""
    ids = ["jd1_0", "jd1_1", "jd1_2"]
    fake = {
        "key": ("test", 3),
        "ids": ids,
        "id_to_doc": {
            "jd1_0": "Python 开发工程师。",
            "jd1_1": "负责 RAG 检索系统。",
            "jd1_2": "要求熟悉 Java 后端。",
        },
        "meta": {
            cid: {"jd_id": 1, "jd_name": "Python 工程师"} for cid in ids
        },
        "bm25": None,
    }
    monkeypatch.setattr(hybrid_retriever, "_snapshot", lambda: fake)
    return fake


def test_vector_path_fails_falls_back_to_bm25(snapshot, monkeypatch):
    """embedding 服务挂了：检索不崩，退回 BM25 结果。"""
    # BM25 路正常，给一个固定排序
    monkeypatch.setattr(
        hybrid_retriever,
        "bm25_search",
        lambda query, top_n: [
            (cid, snapshot["id_to_doc"][cid]) for cid in snapshot["ids"]
        ],
    )
    # 向量路挂掉
    monkeypatch.setattr(
        hybrid_retriever, "embeddings", FakeEmbeddings(fail=True)
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "vectorstore",
        _fake_vectorstore(snapshot["ids"], fail=True),
    )

    results = hybrid_retriever.hybrid_retrieve("Python", top_n=3)

    assert results  # 不崩、有结果
    assert [cid for cid, _ in results] == snapshot["ids"]


def test_bm25_path_fails_falls_back_to_vector(snapshot, monkeypatch):
    """BM25 路挂了：检索不崩，退回向量路结果。"""

    def boom(query, top_n):
        raise RuntimeError("模拟 BM25 失败")

    monkeypatch.setattr(hybrid_retriever, "bm25_search", boom)
    # 向量路正常，给一个固定排序
    monkeypatch.setattr(
        hybrid_retriever, "embeddings", FakeEmbeddings(fail=False)
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "vectorstore",
        _fake_vectorstore(["jd1_2", "jd1_0", "jd1_1"], fail=False),
    )

    results = hybrid_retriever.hybrid_retrieve("后端", top_n=3)

    assert results
    assert [cid for cid, _ in results] == ["jd1_2", "jd1_0", "jd1_1"]


def test_both_paths_fail_returns_empty(snapshot, monkeypatch):
    """两路都挂：返回空列表，由上层提示，而不是把异常抛到界面。"""

    def boom(query, top_n):
        raise RuntimeError("模拟 BM25 失败")

    monkeypatch.setattr(hybrid_retriever, "bm25_search", boom)
    monkeypatch.setattr(
        hybrid_retriever, "embeddings", FakeEmbeddings(fail=True)
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "vectorstore",
        _fake_vectorstore(snapshot["ids"], fail=True),
    )

    assert hybrid_retriever.hybrid_retrieve("查询", top_n=3) == []


def test_empty_store_returns_empty(monkeypatch):
    """空库：直接返回空，不查 Chroma。"""
    empty = {
        "key": ("test", 0),
        "ids": [],
        "id_to_doc": {},
        "meta": {},
        "bm25": None,
    }
    monkeypatch.setattr(hybrid_retriever, "_snapshot", lambda: empty)

    assert hybrid_retriever.hybrid_retrieve("查询", top_n=3) == []
