"""混合检索：BM25 关键词 + 向量语义 → RRF 融合 → bge-reranker 精排。

三道关卡的取舍：
1. BM25 抓"关键词命中"（专有名词、技术栈名），向量抓"语义相近"（换了说法的同义表达），
   两路各有所长，所以都要；
2. 两路名次用 RRF 融合（只比名次、不比绝对分数），不用调权重、天然抗量纲差异；
3. 融合后的粗排结果再交给精排模型，精排挂了就退回粗排（增强项，不是刚需）。

性能要点：BM25 必须对**全量块**分词建索引，是 O(全库) 的操作。如果每次查询都重建，
单次检索成本就随 JD 数量线性上升。所以这里带一个内容快照缓存，见 `_snapshot()`。
"""

import jieba
import requests
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 保留这几个名字的导入，老的 `from hybrid_retriever import embeddings/vectorstore` 仍可用
from recruit_graph import jd_vector_store
from recruit_graph.jd_vector_store import (  # noqa: F401
    API_KEY,
    BASE_URL,
    embeddings,
    vectorstore,
)

RRF_K = 60
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# 精排是"增强"不是"刚需"：给 15 秒上限，超时就走降级，绝不让请求无限挂住。
RERANK_TIMEOUT = 15

# 内容快照缓存：{"key": (版本号, 集合总数), "ids": [], "id_to_doc": {}, "meta": {}, "bm25": ...}
_SNAPSHOT = {"key": None, "ids": [], "id_to_doc": {}, "meta": {}, "bm25": None}


def _snapshot() -> dict:
    """取当前向量库全量块的快照，带缓存；内容没变就直接复用。

    缓存键 = (进程内内容版本号, 集合总数)：
    - 本进程增删向量时版本号会 +1，立刻失效（见 jd_vector_store._bump_version）；
    - 别的进程（如另一个脚本）写了库，版本号不变但集合总数会变，同样失效；
    - 极端情况（别的进程"删两块又加两块"数量恰好不变）需要手动 `clear_cache()`。
    """
    key = (
        jd_vector_store.index_version(),
        jd_vector_store.vectorstore._collection.count(),
    )
    if _SNAPSHOT["key"] != key:
        data = jd_vector_store.vectorstore._collection.get()
        ids = list(data.get("ids") or [])
        documents = list(data.get("documents") or [])
        metadatas = list(data.get("metadatas") or [])
        _SNAPSHOT.update(
            key=key,
            ids=ids,
            id_to_doc=dict(zip(ids, documents)),
            meta=dict(zip(ids, metadatas)),
            # 空库时不能建索引，留 None，调用处判空
            bm25=(
                BM25Okapi([list(jieba.cut(doc)) for doc in documents])
                if documents
                else None
            ),
        )
    return _SNAPSHOT


def clear_cache() -> None:
    """手动让检索缓存失效（用别的进程/工具改过向量库时调用）。"""
    _SNAPSHOT["key"] = None


def bm25_search(query: str, top_n: int = 5):
    """BM25 关键词检索，返回按相关度排序的若干块，每项为 (块id, 文本)。"""
    snapshot = _snapshot()
    if not snapshot["ids"]:
        return []

    scores = snapshot["bm25"].get_scores(list(jieba.cut(query)))
    ranked_index = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_n]
    return [
        (snapshot["ids"][i], snapshot["id_to_doc"][snapshot["ids"][i]])
        for i in ranked_index
    ]


def hybrid_retrieve(query: str, top_n: int = 5):
    """向量 + BM25 经 RRF 融合，返回粗排后的若干块，每项 (块id, 文本)。"""
    snapshot = _snapshot()
    ids = snapshot["ids"]
    if not ids:  # 空库直接返回，别去查 Chroma（n_results 不能为 0）
        return []

    # 第一组：BM25 路——拿到所有块按关键词排好的 id 顺序
    bm25_ranked_ids = [cid for cid, _ in bm25_search(query, top_n=len(ids))]

    # 第二组：向量路——语义召回，拿到按相似度排好的 id 顺序
    query_embedding = embeddings.embed_query(query)
    vec_result = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=len(ids),
    )
    vector_ranked_ids = vec_result["ids"][0]

    # 第三组：RRF —— 每个 id 先记 0 分，再按它在两路里的名次各加 1/(常数+名次)。
    # 名次从 1 开始（标准 RRF 公式），常数 60 用来压平"第一名与第二名"的差距。
    rrf_scores = {cid: 0.0 for cid in ids}
    for rank, cid in enumerate(bm25_ranked_ids, start=1):
        rrf_scores[cid] += 1 / (RRF_K + rank)
    for rank, cid in enumerate(vector_ranked_ids, start=1):
        rrf_scores[cid] += 1 / (RRF_K + rank)

    # 第四组：按融合分降序，取前 top_n 个 id，再用字典换回文本
    fused_ids = sorted(ids, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(cid, snapshot["id_to_doc"][cid]) for cid in fused_ids[:top_n]]


def rerank_candidates(query: str, candidates, top_k: int = 3):
    """用 bge-reranker 对粗排候选精排，返回 [(块id, 文本), ...]，长度 top_k。"""
    # 第一组：把候选里的文本抽成列表（rerank 接口只认文本，不认我们的块 id）
    documents = [doc for _, doc in candidates]

    # 第二组：发 POST 请求到 /rerank。timeout 必须有：没有它，服务端半死不活时
    # 这个请求会一直挂着，上层"失败就降级"的分支永远等不到异常。
    resp = requests.post(
        f"{BASE_URL}/rerank",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False,
        },
        timeout=RERANK_TIMEOUT,
    )
    # 4xx / 5xx 必须抛出：否则下面的 resp.json()["results"] 会抛 KeyError，
    # 把"服务不可用"伪装成"返回格式不对"，排查时误导人。
    resp.raise_for_status()
    results = resp.json()["results"]

    # 第三组：results 已按相关度从高到低排好，每项 item["index"] 是"它在你传入 documents 里的局部位置"
    # 用这个局部位置去 candidates 里取，才能把 (块id, 文本) 拿回来
    return [candidates[item["index"]] for item in results]


def hybrid_search(
    query: str,
    candidate_n: int = 8,
    top_k: int = 3,
    use_rerank: bool = True,
):
    """混合检索总入口：BM25+向量 RRF 粗排 → bge-reranker 精排。
    返回 LangChain Document 列表（与 retriever.invoke 同类型）。
    use_rerank=True（默认）：粗排后再精排，结果最准、但多调一次模型、稍慢；
    use_rerank=False：只做 BM25+向量粗排，更快、且不依赖精排服务。"""
    # 第一组：粗排，多捞一些候选（candidate_n 个）
    candidates = hybrid_retrieve(query, top_n=candidate_n)
    if not candidates:  # 空库：直接返回空列表，让上层给出"请先录入岗位"的提示
        return []

    if use_rerank:
        # 第二组：精排压到 top_k。rerank 只是“增强”不是“刚需”，
        # 它挂了（超时/4xx/5xx/格式不对）就自动降级：直接用粗排前 top_k 条。
        try:
            ranked = rerank_candidates(query, candidates, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            print(f"[降级] rerank 精排失败（{type(e).__name__}），退回 BM25+向量粗排结果")
            ranked = candidates[:top_k]
    else:
        # 只要粗排：直接取 RRF 融合后的前 top_k，省掉一次模型调用，更快。
        ranked = candidates[:top_k]

    # 第三组：把每个 (块id, 文本) 包成 Document 盒子，返回类型和 retriever.invoke 一致。
    # metadata 里带上 jd_id / jd_name：调用方（问答出处、JD 生成范例、深度研究引用）
    # 直接读字段即可，不必再去解析 "jd3_5" 这种内部编号 —— 那是向量层的实现细节。
    snapshot = _snapshot()
    documents = []
    for cid, text in ranked:
        meta = snapshot["meta"].get(cid) or {}
        metadata = {
            "id": cid,
            "jd_id": meta.get("jd_id"),
            "jd_name": meta.get("jd_name", ""),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents
